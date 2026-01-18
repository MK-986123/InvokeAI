# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 model utilities and parameter presets."""

from invokeai.backend.flux2.model import Flux2Params
from invokeai.backend.model_manager.taxonomy import AnyVariant, Flux2VariantType


# FLUX.2 uses the same VAE as FLUX.1 but with different latent channels
# The VAE is AutoencoderKLFlux2 with 32 latent channels
FLUX2_LATENT_CHANNELS = 32


# Parameter presets for FLUX.2-klein variants
_flux2_transformer_params: dict[AnyVariant, Flux2Params] = {
    Flux2VariantType.Klein4B: Flux2Params(
        in_channels=128,
        hidden_size=3072,  # 24 * 128
        num_attention_heads=24,
        attention_head_dim=128,
        num_layers=5,
        num_single_layers=20,
        mlp_ratio=3.0,
        joint_attention_dim=7680,  # Qwen3 text encoder output
        axes_dims_rope=[32, 32, 32, 32],
        rope_theta=2000,
        timestep_guidance_channels=256,
        guidance_embeds=False,  # Distilled model, no guidance needed
    ),
    Flux2VariantType.Klein9B: Flux2Params(
        in_channels=128,
        hidden_size=4096,  # 32 * 128
        num_attention_heads=32,
        attention_head_dim=128,
        num_layers=8,
        num_single_layers=24,
        mlp_ratio=3.0,
        joint_attention_dim=12288,  # Larger Qwen3 text encoder output
        axes_dims_rope=[32, 32, 32, 32],
        rope_theta=2000,
        timestep_guidance_channels=256,
        guidance_embeds=False,  # Distilled model
    ),
    Flux2VariantType.Klein9BFP8: Flux2Params(
        in_channels=128,
        hidden_size=4096,
        num_attention_heads=32,
        attention_head_dim=128,
        num_layers=8,
        num_single_layers=24,
        mlp_ratio=3.0,
        joint_attention_dim=12288,
        axes_dims_rope=[32, 32, 32, 32],
        rope_theta=2000,
        timestep_guidance_channels=256,
        guidance_embeds=False,
    ),
}


def get_flux2_transformer_params(variant: AnyVariant) -> Flux2Params:
    """Get FLUX.2 transformer parameters for a given variant."""
    try:
        return _flux2_transformer_params[variant]
    except KeyError:
        raise ValueError(f"Unknown FLUX.2 variant: {variant}")


# Maximum sequence lengths for FLUX.2 models
# These are the recommended values from the model card
_flux2_max_seq_lengths: dict[AnyVariant, int] = {
    Flux2VariantType.Klein4B: 512,
    Flux2VariantType.Klein9B: 512,
    Flux2VariantType.Klein9BFP8: 512,
}


def get_flux2_max_seq_length(variant: AnyVariant) -> int:
    """Get maximum text sequence length for a FLUX.2 variant."""
    try:
        return _flux2_max_seq_lengths[variant]
    except KeyError:
        raise ValueError(f"Unknown FLUX.2 variant for max seq len: {variant}")


# Default inference settings for FLUX.2-klein (distilled for fast inference)
FLUX2_KLEIN_DEFAULT_STEPS = 4
FLUX2_KLEIN_DEFAULT_GUIDANCE = 1.0  # Guidance scale (not used in distilled)
