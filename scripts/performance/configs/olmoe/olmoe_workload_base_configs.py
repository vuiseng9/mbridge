from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig(
    # num_gpus=8,
    # global_batch_size=2048,
    # expert_model_parallel_size=8,
    # recompute_granularity="selective",
)

OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = BASE_OLMOE_1B_7B_CONFIG