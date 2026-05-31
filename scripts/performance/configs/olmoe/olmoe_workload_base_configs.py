from dataclasses import replace

from utils.utils import WorkloadBaseConfig

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig()

NGPU=2
MBS=1
OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus=NGPU,
    micro_batch_size=MBS,
    global_batch_size=MBS*NGPU,
    use_megatron_fsdp=True,
)
