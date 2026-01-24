# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 denoise loop implementation.

FLUX.2-klein uses a simpler denoise loop compared to FLUX.1:
- No ControlNet support
- No IP Adapter
- No CLIP embeddings (uses Qwen3 text encoder only)
- Supports BN normalization of latents (from FLUX.2 VAE stats)
- Distilled for few-step inference (default 4 steps)
"""

import math
from typing import Callable

import torch
from tqdm import tqdm

from invokeai.backend.flux2.model import Flux2
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    model: Flux2,
    # Model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    # Sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    guidance: float | None = None,
    cfg_scale: list[float] | None = None,
    # Negative conditioning (for CFG)
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    # BN normalization stats from VAE (optional)
    bn_mean: torch.Tensor | None = None,
    bn_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FLUX.2 denoising loop.

    Args:
        model: FLUX.2 transformer model.
        img: Noisy image latents [B, seq_len, in_channels].
        img_ids: Image position IDs [B, seq_len, n_axes].
        txt: Text embeddings from Qwen3 [B, txt_len, joint_attention_dim].
        txt_ids: Text position IDs [B, txt_len, n_axes].
        timesteps: List of timestep values (sigma schedule, descending from 1 to 0).
        step_callback: Callback for progress reporting.
        guidance: Guidance value (typically 1.0 for distilled klein models).
        cfg_scale: Per-step CFG scale values. If all 1.0, no negative pass is run.
        neg_txt: Negative text embeddings (required if any cfg_scale > 1.0).
        neg_txt_ids: Negative text position IDs.
        bn_mean: VAE batch norm running mean for latent normalization [in_channels].
        bn_std: VAE batch norm running std for latent normalization [in_channels].

    Returns:
        Denoised latents [B, seq_len, in_channels].
    """
    total_steps = len(timesteps) - 1

    if cfg_scale is None:
        cfg_scale = [1.0] * total_steps

    # Apply BN normalization if stats available
    if bn_mean is not None and bn_std is not None:
        img = (img - bn_mean) / bn_std

    for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # Prepare guidance vector (None for distilled models without guidance_embed)
        guidance_vec = None
        if guidance is not None and model.params.guidance_embeds:
            guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

        # Positive prediction
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        # Apply CFG if scale > 1.0
        step_cfg = cfg_scale[min(step_index, len(cfg_scale) - 1)]
        if not math.isclose(step_cfg, 1.0):
            if neg_txt is None:
                raise ValueError("Negative text conditioning is required when cfg_scale > 1.0.")

            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            pred = neg_pred + step_cfg * (pred - neg_pred)

        # Euler step
        preview_img = img - t_curr * pred
        img = img + (t_prev - t_curr) * pred

        # Progress callback with correct timestep scaling (0-1000 range)
        step_callback(
            PipelineIntermediateState(
                step=step_index + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(t_curr * 1000),
                latents=preview_img,
            ),
        )

    # Apply BN denormalization if stats available
    if bn_mean is not None and bn_std is not None:
        img = img * bn_std + bn_mean

    return img
