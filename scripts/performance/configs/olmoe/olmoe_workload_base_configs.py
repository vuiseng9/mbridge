from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig()

OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    global_batch_size=32,
    use_megatron_fsdp=True,
)
