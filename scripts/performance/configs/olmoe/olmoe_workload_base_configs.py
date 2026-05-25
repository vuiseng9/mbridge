from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig()

OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    global_batch_size=64,
    use_megatron_fsdp=True,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
)

# must note the importance
    # cfg.model.moe_router_force_load_balancing = True  # required for token dropless
    # cfg.model.recompute_granularity = None

# peak_mem_gb = torch.cuda.device_memory_used() / 1024**3, closest to nvidia-smi

# OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
#     BASE_OLMOE_1B_7B_CONFIG,
#     num_gpus=8,
#     micro_batch_size=4,
#     global_batch_size=32,
#     use_megatron_fsdp=True,

# e00 - global baseline
# mfsdp, mb4,gb32
# gbs32: Step Time: 0.50s , 32493.29 tok/s/gpu, GPU utilization: 255.6 MODEL_TFLOP/s/GPU , Peak Mem: 62.5GB (69 GB)
# gbs48: Step Time: 0.62s , 33146.87 tok/s/gpu, GPU utilization: 260.7 MODEL_TFLOP/s/GPU , Peak Mem: 73.8GB (80570)

# e02 - mfsdp + ep=8, mbs4
    # expert_model_parallel_size=8,
    # expert_tensor_parallel_size=1,
# gbs32: Step Time: 0.47s , 35080.99 tok/s/gpu, GPU utilization: 275.9 MODEL_TFLOP/s/GPU , Peak Mem: 60.6GB (72~3GB)
# gbs48: Step Time: 0.57s , 36087.42 tok/s/gpu, GPU utilization: 283.9 MODEL_TFLOP/s/GPU , Peak Mem: 72.0GB (~80GB)
# without nccl_ub
# gbs32: Step Time: 0.47s , 35216.32 tok/s/gpu, GPU utilization: 272.5 MODEL_TFLOP/s/GPU , Peak Mem: 60.7GB (72~3GB)
# without nccl_ub, nccl alltoall (less performant)
# gbs32: Step Time: 0.53s , 31038.07 tok/s/gpu, GPU utilization: 244.1 MODEL_TFLOP/s/GPU , Peak Mem: 61.8GB
# without nccl_ub, deep ep (even less performant, maybe SM settings? slightly lower memory)
# gbs32: Step Time: 0.56s , 29126.23 tok/s/gpu, GPU utilization: 229.1 MODEL_TFLOP/s/GPU , Peak Mem: 63.5GB (70-1GB)

# e04 - from e02, selective recomputation
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = ['core_attn']
# Step Time: 0.48s , 34200.59 tok/s/gpu, GPU utilization: 269.0 MODEL_TFLOP/s/GPU , Peak Mem: 62.1GB (70GB) weird
    # is recomputation really happening? i think yes, why effect is low? because flashattention by design include
    # the same recomputation has been baked in the kernel.
    # so we turn it off.
    # also it is not output discarding, yes verified
    # TODO, are we sure it is really flash attention


# e05 - from e02, recomputation layernorm # Fine-grained Recomputation, CheckpointWithoutOutput
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = ['layernorm']
# gbs32: Step Time: 0.47s , 35023.16 tok/s/gpu, GPU utilization: 275.5 MODEL_TFLOP/s/GPU , Peak Mem: 59.6GB (70 GB)
# gbs40: Step Time: 0.57s , 36136.66 tok/s/gpu, GPU utilization: 284.2 MODEL_TFLOP/s/GPU , Peak Mem: 71.0GB (78.8GB)
# gbs 56: OOM

        # e06 - from e02, recompute core_attn, layernorm # no beneficial, no obvious gain in memory and slow donw
            # cfg.model.recompute_granularity = "selective"
            # cfg.model.recompute_modules = ['core_attn', 'layernorm']
        # Step Time: 0.48s , 34279.71 tok/s/gpu, GPU utilization: 269.6 MODEL_TFLOP/s/GPU , Peak Mem: 61.7GB (72 GB)

# e07 - from e02, recompute moe_act, 
# gbs32: Step Time: 0.50s , 33052.72 tok/s/gpu, GPU utilization: 260.0 MODEL_TFLOP/s/GPU , Peak Mem: 45.5GB (50GB)
# gbs48: Step Time: 0.72s , 34327.44 tok/s/gpu, GPU utilization: 270.0 MODEL_TFLOP/s/GPU , Peak Mem: 61.4GB (70GB)
# gbs56: Step Time: 0.82s , 34797.85 tok/s/gpu, GPU utilization: 273.7 MODEL_TFLOP/s/GPU , Peak Mem: 72.4GB (80GB)
# gbs64: oom

# e08 - from e02, recompute layer_norm + moe_act
# layer_norm + moe_act failed? buffer registration again!
# if we disable nccl_ub, we can get it running
# gbs32: Step Time: 0.51s , 32206.07 tok/s/gpu, GPU utilization: 253.3 MODEL_TFLOP/s/GPU , Peak Mem: 50.7GB
# gbs56: Step Time: 0.82s , 34758.07 tok/s/gpu, GPU utilization: 273.4 MODEL_TFLOP/s/GPU , Peak Mem: 70.7GB (78.7)

# e08 - from e07 ++ 
    # NVTE_CPU_OFFLOAD_V1=1 
    # cfg.model.fine_grained_activation_offloading = True
    # cfg.model.offload_modules = ['...']
# gbs32, 'mlp_norm': 960.00gb per gpu
#       Step Time: 0.51s , 32091.48 tok/s/gpu, GPU utilization: 252.4 MODEL_TFLOP/s/GPU , Peak Mem: 50.0GB
#
# gbs32, 'qkv_linear': 3840.00gb per gpu
#       Step Time: 0.54s , 30065.09 tok/s/gpu, GPU utilization: 236.5 MODEL_TFLOP/s/GPU , Peak Mem: 47.8GB
#    56 Step Time: 0.84s , 34155.77 tok/s/gpu, GPU utilization: 268.7 MODEL_TFLOP/s/GPU , Peak Mem: 76.5GB
#    64 Step Time: 0.95s , 34313.56 tok/s/gpu, GPU utilization: 269.9 MODEL_TFLOP/s/GPU , Peak Mem: 79.4GB
# gbs32, 'core_attn': 3840.00
#       Step Time: 0.55s , 29576.33 tok/s/gpu, GPU utilization: 232.6 MODEL_TFLOP/s/GPU , Peak Mem: 48.7GB
# gbs32, 'expert_fc1': 7681.27
#       Step Time: 0.69s , 23848.06 tok/s/gpu, GPU utilization: 187.6 MODEL_TFLOP/s/GPU , Peak Mem: 45.8GB
# gbs32, 'moe_act': 7682.11
#       Step Time: 0.73s , 22444.99 tok/s/gpu, GPU utilization: 176.5 MODEL_TFLOP/s/GPU , Peak Mem: 45.5GB

# Open
# how offload and recompute convention work, it is not clear

# extra

# nvjet is NVIDIA's JIT (just-in-time) compiled GEMM engine inside cuBLASLt (newer cuBLAS versions).
# nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN


# It's CPU wall-clock time, but bracketed by torch.cuda.synchronize(), so it does capture GPU work completion — not just CPU control flow.
# Here's what happens at each boundary:
# start(barrier=True) (timers.py:147):