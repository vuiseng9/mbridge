import os
import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.recipes.olmoe import olmoe_7b_pretrain_config

from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
#     userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
#     userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
#     userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
#     userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.core.transformer.enums import AttnBackend
logger = logging.getLogger(__name__)

def set_llama32_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3.2 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

def dev_llama32_1b_pretrain_config_gpu48gb(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:

    base_cfg = get_workload_base_config(
        model_family_name="dev",
        model_recipe_name="dev_llama32_1b",
        gpu="gpu48gb",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    precision_config = get_precision_config(precision)

    cfg = llama32_1b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_llama32_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    if cfg.ddp.use_megatron_fsdp is True:
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.keep_fp8_transpose_cache = True

    return cfg


def set_olmoe_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Olmoe configs."""
    # cfg.model.bias_activation_fusion = True
    # cfg.model.recompute_method = None
    # cfg.model.recompute_num_layers = None
    # cfg.model.moe_router_fusion = True

    # cfg.mixed_precision.grad_reduce_in_fp32 = False
    # cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.attention_backend = AttnBackend.auto
    cfg.model.moe_router_force_load_balancing = True  # required for token dropless
    cfg.model.recompute_granularity = None

def dev_olmoe_4_layer_pretrain_config_gpu48gb(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="dev",
        model_recipe_name="dev_olmoe_4_layer",
        gpu="gpu48gb",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    precision_config = get_precision_config(precision)

    cfg = olmoe_7b_pretrain_config()
    cfg.mixed_precision = precision_config
    set_olmoe_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.model.num_layers = 4 # 16 original
    # cfg.model.num_moe_experts = 32
        
    # cfg.train.micro_batch_size=5
    # cfg.train.global_batch_size=cfg.train.micro_batch_size*8
    
    cfg.model.expert_model_parallel_size=base_cfg.num_gpus
    cfg.model.expert_tensor_parallel_size=1

    if cfg.ddp.use_megatron_fsdp:
        cfg.ddp.nccl_ub = False
        cfg.model.gradient_accumulation_fusion = False  # Disabled to avoid functional errors
        cfg.ddp.keep_fp8_transpose_cache = True

    if cfg.model.expert_model_parallel_size > 1:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
        # cfg.model.moe_token_dispatcher_type = "flex"
        # cfg.model.moe_flex_dispatcher_backend = "hybridep"
        # cfg.model.moe_flex_dispatcher_backend = "deepep"

    return cfg