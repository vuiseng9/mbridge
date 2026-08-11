try:
    import megatron.bridge  # noqa: F401
    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .dev_llm_pretrain import (
        dev_llama32_1b_pretrain_config_gpu48gb,
        dev_olmoe_4_layer_pretrain_config_gpu48gb
    )

from .dev_workload_base_configs import (
    DEV_LLAMA32_1B_PRETRAIN_CONFIG_GPU48GB_BF16_V1,
    DEV_OLMOE_4_LAYER_PRETRAIN_CONFIG_GPU48GB_BF16_V1
)

__all__ = [
    "DEV_LLAMA32_1B_PRETRAIN_CONFIG_GPU48GB_BF16_V1",
    "DEV_OLMOE_4_LAYER_PRETRAIN_CONFIG_GPU48GB_BF16_V1"
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "dev_llama32_1b_pretrain_config_gpu48gb",
            "dev_olmoe_4_layer_pretrain_config_gpu48gb"
        ]
    )