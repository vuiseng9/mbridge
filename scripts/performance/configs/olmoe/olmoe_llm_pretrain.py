import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config


from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
#     userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
#     userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
#     userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
#     userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)

def set_olmoe_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Olmoe configs."""
    # cfg.model.bias_activation_fusion = True
    # cfg.model.recompute_granularity = None
    # cfg.model.recompute_method = None
    # cfg.model.recompute_num_layers = None
    # cfg.model.moe_router_fusion = True

    # cfg.mixed_precision.grad_reduce_in_fp32 = False
    # cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True  # required for token dropless


def olmoe_1b_7b_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="olmoe",
        model_recipe_name="olmoe_1b_7b",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = olmoe_7b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_olmoe_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    
    cfg.model.recompute_granularity = None
    
    # cfg.model.recompute_granularity = "full"
    # cfg.model.recompute_method = "uniform"
    # cfg.model.recompute_num_layers = 4

    # cfg.model.cpu_offloading_weights = True
    # cfg.model.cpu_offloading_activations = False
    
    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.keep_fp8_transpose_cache = True

    if cfg.model.expert_model_parallel_size > 1:
        cfg.model.moe_flex_dispatcher_backend = "hybridep"
        cfg.model.moe_token_dispatcher_type = "flex"
        
    return cfg