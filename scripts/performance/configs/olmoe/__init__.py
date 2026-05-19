try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .olmoe_llm_pretrain import (
        olmoe_1b_7b_pretrain_config_h100
    )

from .olmoe_workload_base_configs import (
    OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1
)

__all__ = [
    "OLMOE_1B_7B_PRETRAIN_CONFIG_H100_BF16_V1"
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "olmoe_1b_7b_pretrain_config_h100"
        ]
    )