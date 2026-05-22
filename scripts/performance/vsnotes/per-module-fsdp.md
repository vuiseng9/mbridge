
---

**Step 1 — Grouping params by FSDP unit at buffer construction time**

param_and_grad_buffer.py:

```python
for m in module.modules():
    if isinstance(m, tuple(policy.fsdp_unit_modules)):   # e.g. TransformerLayer
        fsdp_units.append([param_to_name[p] for p in m.parameters()])

# Then for every param:
for fsdp_unit_id, fsdp_unit in enumerate(fsdp_units):
    if name in fsdp_unit:
        param_attrs["fsdp_unit_id"] = fsdp_unit_id   # ← stamp every param with its module's ID
```

All parameters belonging to the same `TransformerLayer` get tagged with the same `fsdp_unit_id`. This is what makes it **per-module** — the unit of sharding/gathering is one whole layer, not individual parameters.

---

**Step 2 — All-gather is triggered per-module, not per-param**

Two PyTorch hooks are registered **on each FSDP unit module** (e.g. each `TransformerLayer`):

- **`register_forward_pre_hook` → `_pre_forward_param_unshard`** (line 968): fires before the module's `forward()`, calls `all_gather_and_wait_parameters_ready(list(module.parameters()))` — gathers **all params of that layer** at once
- **`register_multi_grad_hook` → `_pre_backward_param_unshard`** (line 1002): fires before that module's backward, calls the same all-gather for the backward pass

---

**Step 3 — Post-forward: shard (release) the whole module at once**

`_post_forward` hook on each FSDP unit (line 912): after layer's `forward()` returns, calls `release_module_parameters(module, ...)` which walks `module.parameters()` and releases their bucket — freeing the all-gathered memory for the **entire layer** in one shot.

---

**The key contrast with per-parameter FSDP (PyTorch FSDP1):**

| | PyTorch FSDP1 | Megatron FSDP |
|---|---|---|
| Gather unit | One `FlatParameter` per wrapped module | All params of a `TransformerLayer` bucket together by `fsdp_unit_id` |
| Release trigger | Post-forward of each `FSDP` wrapper | `_post_forward` hook keyed to `isinstance(module, fsdp_unit_modules)` |
| Prefetch granularity | Per `FlatParameter` | Per layer bucket group (`suggested_communication_unit_size ≈ 2 × layer_size`) |

The module-level grouping is established entirely by `fsdp_unit_id` stamping in `BucketingPolicy`, and the gather/release hooks are wired at the `TransformerLayer` boundary — never at individual parameter level.



---

## All-Gather Overlap in MegatronFSDP

The overlap comes from **three cooperating mechanisms** working together.

### 1. Dedicated CUDA side stream

In `_init_fsdp_param_and_grad_buffer` ([megatron_fsdp.py line 367](3rdparty/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py#L367)):
```python
self.side_stream_for_param_gather = torch.cuda.Stream()
```
All NCCL all-gather collectives are dispatched on this stream, not the main compute stream. The main stream can proceed with a layer's compute while the side stream gathers the *next* layer's params in the background.

---

### 2. Prefetch in `all_gather_params`

([param_and_grad_buffer.py line 3948](3rdparty/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py#L3948)) — when asked to gather layer N's bucket, the pipeline immediately looks ahead:

```python
if prefetch:
    # Walk forward (or backward) from bucket N.
    bucket_id = next_bucket_id(ag_buckets)   # layer N+1 (or N-1 in bwd)
    while prefetch_all_gather_size < suggested_AG_prefetch_size:
        ag_buckets.extend(bucket_group_for(bucket_id))
        bucket_id = next_bucket_id(ag_buckets)
```

`suggested_AG_prefetch_size = total_layer_params / num_layers` (one layer's worth of params). So when layer N's pre-forward hook fires, two NCCL collectives are launched on `ag_stream`: one for layer N (required), one for layer N+1 (prefetch).

The actual dispatch ([line 4086](3rdparty/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py#L4086)):
```python
all_gather_stream.wait_stream(torch.cuda.current_stream())   # side stream waits for main
with torch.cuda.stream(all_gather_stream):
    with _coalescing_manager(dp_group, async_ops=True) as coalescing_event:
        for bucket_id in buckets:
            self.async_bucket_gather(bucket_id, bwd)
# event stored — NOT waited on yet
self.param_gather_event_map[bucket_key] = (coalescing_event, mark_ready)
```

Both collectives are in-flight asynchronously.

---

### 3. Hook sequencing creates the overlap window

**Forward pass** — for each `TransformerLayer`:

```
pre-forward hook (layer N)
  → launch AG for layer N  (side stream, async)
  → launch AG for layer N+1 (side stream, prefetch, async)
  → wait_bucket_ready(layer N)   ← makes main stream wait for layer N's event only
layer N compute begins on main stream
                                  ← layer N+1's AG runs concurrently on side stream
post-forward hook (layer N)
  → release layer N's bucket

pre-forward hook (layer N+1)
  → layer N+1 bucket already READY_TO_USE → wait returns immediately
layer N+1 compute begins
```

**Backward pass** — same structure but `prefetch_order=BACKWARD_PASS_ORDER` (searches for `bucket_id - 1` instead of `+ 1`):

```
pre-backward hook (layer N, via register_multi_grad_hook on layer N's output)
  → launch AG for layer N  (side stream)
  → launch AG for layer N-1 (side stream, prefetch)
  → wait_bucket_ready(layer N)
layer N backward runs on main stream
                                  ← layer N-1's AG running on side stream
post-backward hook (layer N)
  → release_module_parameters(layer N, bwd=True)
  → _process_post_backward_gradients → reduce-scatter grads (also async)

pre-backward hook (layer N-1)
  → already gathered → immediate
```

---

### The sync between streams

`wait_bucket_ready` ([line 4118](3rdparty/Megatron-LM/megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py#L4118)):
```python
param_gather_event, mark_bucket_ready_to_use = self.param_gather_event_map.pop(bucket_key)
param_gather_event.wait()   # inserts a stream-level wait event — no CPU stall
mark_bucket_ready_to_use()
```
`param_gather_event.wait()` is a **GPU-side wait** (CUDA event) — it inserts a dependency into the main stream so the main stream stalls only until the NCCL collective on the side stream completes. The CPU thread is never blocked.

---

### Summary timeline (N layers)

```
side stream: [AG₀+AG₁] [AG₂] [AG₃] ... 
main stream:     wait₀  [fwd₀][fwd₁][fwd₂]...
                         ↑ready ↑ready  ↑ready (already prefetched)
```

The overlap is: while the main stream executes layer N, the side stream is already gathering layer N+1. `wait_bucket_ready` on layer N+1 is essentially free because by the time pre-forward for N+1 fires, the AG launched during N's pre-forward hook has had a full layer's compute time to finish.


# code traces

align_param_gather=False
average_in_collective=False
bucket_size=None
check_for_large_grads=False
check_for_nan_in_grad=False

delay_wgrad_compute=False
disable_symmetric_registration=False
fp8_param_gather=False
fsdp_all_gather_in_start_param_sync=True
fsdp_db_use_persist_buf_on_alloc_fail=False
fsdp_manual_registration=False
grad_reduce_in_fp32=False
gradient_reduce_div_fusion=True
keep_fp8_transpose_cache=False
megatron_fsdp_grad_comm_dtype=None
megatron_fsdp_main_grads_dtype=None
megatron_fsdp_main_params_dtype=torch.float32
nccl_ub=False
num_distributed_optimizer_instances=1
outer_dp_sharding_strategy=no_shard
pad_buckets_for_high_nccl_busbw=False
param_name_patterns_for_fp32_local_accumulation=()
reduce_scatter_with_fp32_accumulation=False
reuse_grad_buf_for_mxfp8_param_ag=False
suggested_communication_unit_size=None
use_custom_fsdp=False


use_distributed_optimizer=True
use_megatron_fsdp=True
data_parallel_sharding_strategy=optim_grads_params

overlap_param_gather=True
overlap_grad_reduce=True
fsdp_double_buffer=False - dynamic allocation stalls



┌─────────────────────┬──────────────────────┬─────────────────────────────────────────────────────┐
│ fsdp_double_buffer  │ nccl_ub              │ Outcome                                             │
├─────────────────────┼──────────────────────┼─────────────────────────────────────────────────────┤
│ False               │ False                │ ✅ Valid. Dynamic cudaMalloc per AG/RS call.        │
│                     │                      │ Works, but allocation latency can serialize         │
│                     │                      │ with compute, degrading overlap quality.            │
├─────────────────────┼──────────────────────┼─────────────────────────────────────────────────────┤
│ True                │ False                │ ✅ Valid. Pre-allocated FixedPoolAllocator (2×       │
│                     │                      │ layer-size persistent buffers). Eliminates alloc     │
│                     │                      │ stalls. NCCL uses standard ring algorithm.           │
│                     │                      │ Best option if nccl_ub is unavailable.              │
├─────────────────────┼──────────────────────┼─────────────────────────────────────────────────────┤
│ False               │ True                 │ ⛔ Impossible. Code forces                          │
│                     │                      │ fsdp_double_buffer = True when nccl_ub=True          │
│                     │                      │ (param_and_grad_buffer.py line 1670).               │
│                     │                      │ UBR requires non-dynamic persistent memory.          │
├─────────────────────┼──────────────────────┼─────────────────────────────────────────────────────┤
│ True                │ True                 │ ✅ Best performance. FixedPoolAllocator is           │
│                     │                      │ registered with NCCL's MemPool API so NCCL          │
│                     │                      │ uses SM-efficient userbuffer algorithm               │
│                     │                      │ (4–6 SMs on NVLink vs 16 without).                  │
└─────────────────────┴──────────────────────┴─────────────────────────────────────────────────────┘