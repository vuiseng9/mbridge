from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig()

OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus=8,
    micro_batch_size=5,
    global_batch_size=40,
    use_megatron_fsdp=True,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    # cpu_offloading_num_layers=8,
)




# mfsdp, mb4,gb32
# Step Time: 0.50s , 32680.36 tok/s/gpu, GPU utilization: 257.1 MODEL_TFLOP/s/GPU , Peak Mem: 64.5GB
# mfsdp, mb5,gb40
# Step Time: 0.59s , 34691.57 tok/s/gpu, GPU utilization: 272.9 MODEL_TFLOP/s/GPU , Peak Mem: 73.8GB


# mfsdp + ep=8
# Step Time: 0.49s , 33602.54 tok/s/gpu, GPU utilization: 264.3 MODEL_TFLOP/s/GPU , Peak Mem: 62.1GB
# cfg.model.recompute_granularity = None
# Step Time: 0.48s , 34445.13 tok/s/gpu, GPU utilization: 270.9 MODEL_TFLOP/s/GPU , Peak Mem: 61.6GB
# cfg.model.recompute_granularity = None + mbs5, gbs40
# Step Time: 0.58s , 35273.90 tok/s/gpu, GPU utilization: 277.5 MODEL_TFLOP/s/GPU , Peak Mem: 72.0GB



# offload activation: 8 layers
# Step Time: 1.42s , 14426.29 tok/s/gpu, GPU utilization: 113.5 MODEL_TFLOP/s/GPU , Peak Mem: 54.7GB
# offload activation: 15 layers
# Step Time: 3.33s , 6141.17 tok/s/gpu, GPU utilization: 48.3 MODEL_TFLOP/s/GPU , Peak Mem: 31.0GB




# extra

# nvjet is NVIDIA's JIT (just-in-time) compiled GEMM engine inside cuBLASLt (newer cuBLAS versions).
# nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN


# It's CPU wall-clock time, but bracketed by torch.cuda.synchronize(), so it does capture GPU work completion — not just CPU control flow.
# Here's what happens at each boundary:
# start(barrier=True) (timers.py:147):