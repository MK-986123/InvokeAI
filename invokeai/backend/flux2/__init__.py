# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 model support for InvokeAI.

FLUX.2-klein is a distinct architecture from FLUX.1, featuring:
- Different transformer structure (Flux2Transformer2DModel)
- Qwen3 text encoder instead of CLIP+T5
- Different VAE with 32 latent channels
- Different RoPE configuration
- Exponential time-shifting for flow matching (shift=3.0)
"""

from invokeai.backend.flux2.model import Flux2, Flux2Params
from invokeai.backend.flux2.sampling_utils import (
    clip_flux2_timestep_schedule,
    exponential_time_shift,
    generate_flux2_img_ids,
    generate_flux2_txt_ids,
    get_flux2_noise,
    get_flux2_schedule,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.flux2.util import (
    FLUX2_BASE_DEFAULT_STEPS,
    FLUX2_BASE_SHIFT,
    FLUX2_KLEIN_DEFAULT_GUIDANCE,
    FLUX2_KLEIN_DEFAULT_STEPS,
    FLUX2_KLEIN_SHIFT,
    FLUX2_KLEIN_USE_EXPONENTIAL_SHIFT,
    FLUX2_LATENT_CHANNELS,
    get_flux2_max_seq_length,
    get_flux2_transformer_params,
)

__all__ = [
    # Model
    "Flux2",
    "Flux2Params",
    # Sampling utilities
    "get_flux2_noise",
    "get_flux2_schedule",
    "exponential_time_shift",
    "clip_flux2_timestep_schedule",
    "pack_flux2",
    "unpack_flux2",
    "generate_flux2_img_ids",
    "generate_flux2_txt_ids",
    # Configuration
    "get_flux2_transformer_params",
    "get_flux2_max_seq_length",
    "FLUX2_LATENT_CHANNELS",
    "FLUX2_KLEIN_DEFAULT_STEPS",
    "FLUX2_KLEIN_DEFAULT_GUIDANCE",
    "FLUX2_KLEIN_SHIFT",
    "FLUX2_KLEIN_USE_EXPONENTIAL_SHIFT",
    "FLUX2_BASE_DEFAULT_STEPS",
    "FLUX2_BASE_SHIFT",
]
