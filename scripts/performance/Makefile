NSTEP := 10

torchrun_intra = torchrun --standalone --nproc-per-node
ngpu ?= $(shell nvidia-smi -L | wc -l)
MASTER_ADDR ?= localhost
MASTER_PORT ?= 29700
master ?= $(MASTER_ADDR)
port ?= $(MASTER_PORT)

dbg ?= 0
xargs ?= 
logs_dir ?= ./logs
dryrun ?= 0
benchcfg ?= 0
ENVVARS ?=

nsys_cmd ?=
nsys_args ?=
prof_dir ?= ./nsys-prof
bench_dir ?= ./nsys-bench
base_step ?= 10
end_step ?= 
# base_step=3, end_step=4, meaning only step id 3 appearing in nsys
# but it is 4th step of all training steps

ifeq ($(dryrun),1)
	dr := --dryrun
else
	dr :=
endif

ifneq ($(filter 1,$(prof) $(bench)),)
  ifeq ($(bench),1)
    _ := $(shell mkdir -p $(bench_dir))
    end_step = $(shell echo $$(($(base_step) + 25)))
    out_path = $(bench_dir)/$(rep_name)-step_$(base_step)-$(end_step)
  else
    _ := $(shell mkdir -p $(prof_dir))
    end_step = $(shell echo $$(($(base_step) + 1)))
    out_path = $(prof_dir)/$(rep_name)-step_$(base_step)
  endif
  NSTEP = $(shell echo $$(($(end_step) + 1)))
  define nsys_cmd
	nsys profile \
		--capture-range=cudaProfilerApi \
		--capture-range-end=stop \
		-t cuda,nvtx,cudnn,cublas \
		--cuda-graph-trace=node \
		-o $(out_path) \
		--force-overwrite true
  endef
  nsys_args = profiling.use_nsys_profiler=true profiling.record_shapes=false profiling.nvtx_ranges=true profiling.profile_step_start=$(base_step) profiling.profile_step_end=$(end_step)
endif


setup: login-hf

login-hf:
	if ! git config --global --get credential.helper | grep --color=auto -q store; then \
		echo "Warning: git config 'credential.helper' is not set to 'store'."; \
		echo "Executing ... git config --global credential.helper store"; \
		git config --global credential.helper store; \
	fi
	hf auth login

clear-output: purge-prof-dir purge-log-dir purge-bench-dir

purge-bench-dir:
	rm -rf $(bench_dir)

purge-prof-dir:
	rm -rf $(prof_dir)

purge-log-dir:
	rm -rf $(logs_dir)

open-run-cfg-dump:
	code /nemo_run/configs/ConfigContainer.yaml
			
# ── base ────────────────────────────────────────────────────────────────────────────────────────
__perf_run_intranode__:
	DBG_ATTACH=$(dbg) \
		BENCH_CFG=$(benchcfg) $(ENVVARS) \
		$(nsys_cmd) \
			$(torchrun_intra) $(ngpu) run_script.py \
				-m $(family) -mr $(model) -ng $(ngpu) -g $(gputype) -c $(recipe) --max_steps $(NSTEP) $(dr) $(xargs) \
		$(nsys_args)

__perf_run_intranode__mfsdp:
	$(MAKE) __perf_run_intranode__ xargs="--use_megatron_fsdp 1 --cuda_graph_impl none"


# ── h100 ────────────────────────────────────────────────────────────────────────────────────────
#    applicable for any 80GB gpu, e.g. a100.
h100-bf16-llama3_8b:
	$(MAKE) __perf_run_intranode__ family=llama model=llama3_8b gputype=h100 recipe=bf16

# fastrun
tiny-gbs-h100-bf16-llama3_8b:
	$(MAKE) h100-bf16-llama3_8b xargs="-gb 8"

prof-tiny-gbs-h100-bf16-llama3_8b:
	$(MAKE) tiny-gbs-h100-bf16-llama3_8b prof=1 rep_name=$@


# ── b200 ────────────────────────────────────────────────────────────────────────────────────────
b200-bf16-llama3_8b:
	$(MAKE) __perf_run_intranode__ family=llama model=llama3_8b gputype=b200 recipe=bf16

b200-bf16-llama3_8b-mfsdp:
	$(MAKE) __perf_run_intranode___megatron_fsdp family=llama model=llama3_8b gputype=b200 recipe=bf16

b200-fp8-cs-llama3_8b:
	$(MAKE) __perf_run_intranode__ family=llama model=llama3_8b gputype=b200 recipe=fp8_cs

b200-mxfp8-llama3_8b:
	$(MAKE) __perf_run_intranode__ family=llama model=llama3_8b gputype=b200 recipe=fp8_mx

b200-nvfp4-llama3_8b:
	$(MAKE) __perf_run_intranode__ family=llama model=llama3_8b gputype=b200 recipe=nvfp4

# ── dev on x48gb gpu ─────────────────────────────────────────────────────────────────────────────

dev-bf16-llama32_1b:
	$(MAKE) __perf_run_intranode__ family=dev model=dev_llama32_1b gputype=gpu48gb recipe=bf16

dev-bf16-olmoe-4-layer:
	$(MAKE) __perf_run_intranode__ family=dev model=dev_olmoe_4_layer gputype=gpu48gb recipe=bf16

# ── mcore-moe ─────────────────────────────────────────────────────────────────────────────
qw3-a3b-b200-mxfp8:
	$(MAKE) __perf_run_intranode__ family=qwen model=qwen3_30b_a3b gputype=b200 recipe=fp8_mx
	
dsv3-gb300-mxfp8:
	$(MAKE) __perf_run_intranode__ family=deepseek model=deepseek_v3 gputype=gb300 recipe=fp8_mx

# ── olmoe ─────────────────────────────────────────────────────────────────────────────

_h100-bf16-olmoe:
	$(MAKE) __perf_run_intranode__ family=olmoe model=olmoe_1b_7b gputype=h100 recipe=bf16 NSTEP=30

# closest to paper setup, mbs=4, fsdp only
b0-olmoe-dist-optimizer:
	$(MAKE) _h100-bf16-olmoe benchcfg=0

partial-cg-b0-olmoe-dist-optimizer:
	$(MAKE) _h100-bf16-olmoe benchcfg=0 xargs="--cuda_graph_impl transformer_engine --cuda_graph_scope attn,moe_router,moe_preprocess"

full-cg-b0-olmoe-dist-optimizer:
	$(MAKE) _h100-bf16-olmoe benchcfg=0 xargs="--cuda_graph_impl local --cuda_graph_scope full_iteration"


b1-olmoe-ep8:
	$(MAKE) _h100-bf16-olmoe benchcfg=1

partial-cg-b1-olmoe-ep8:
	$(MAKE) _h100-bf16-olmoe benchcfg=1 xargs="--cuda_graph_impl transformer_engine --cuda_graph_scope attn,moe_router,moe_preprocess"

b2-olmoe-ep8-act-ckpt:
	$(MAKE) _h100-bf16-olmoe benchcfg=2

partial-cg-b2-olmoe-ep8-act-ckpt:
	$(MAKE) _h100-bf16-olmoe benchcfg=2 xargs="--cuda_graph_impl transformer_engine --cuda_graph_scope attn,moe_router,moe_preprocess"

b3-olmoe-ep8-act-ckpt-offload:
	$(MAKE) _h100-bf16-olmoe benchcfg=3 ENVVARS="NVTE_CPU_OFFLOAD_V1=1"

x-partial-cg-b3-olmoe-ep8-act-ckpt-offload:
	$(MAKE) _h100-bf16-olmoe benchcfg=3 xargs="--cuda_graph_impl transformer_engine --cuda_graph_scope attn,moe_router,moe_preprocess" ENVVARS="NVTE_CPU_OFFLOAD_V1=1"