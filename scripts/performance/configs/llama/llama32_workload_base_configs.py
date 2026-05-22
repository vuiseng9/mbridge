from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_LLAMA32_1B_CONFIG = WorkloadBaseConfig()

LLAMA32_1B_PRETRAIN_CONFIG_A6000_BF16_V1 = replace(
    BASE_LLAMA32_1B_CONFIG,
    num_gpus=4,
    micro_batch_size=3,
    global_batch_size=12,
    use_megatron_fsdp=True,
)