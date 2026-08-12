import os
from dataclasses import replace

from utils.utils import WorkloadBaseConfig

local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
if local_world_size is None:
    raise ValueError("LOCAL_WORLD_SIZE is not set; use torchrun to launch.")

NGPU = int(local_world_size)

BASE_OLMOE_1B_7B_CONFIG = WorkloadBaseConfig()

_mbs = 4
OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus = NGPU,
    micro_batch_size = _mbs,
    global_batch_size = _mbs * NGPU,
    use_megatron_fsdp = False,
)

OLMOE_1B_7B_PRETRAIN_CONFIG_H100_FP8_MX_V1 = replace(
    BASE_OLMOE_1B_7B_CONFIG,
    num_gpus = NGPU,
    micro_batch_size = _mbs,
    global_batch_size = _mbs * NGPU,
    use_megatron_fsdp = False,
)
