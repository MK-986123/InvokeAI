"""FLUX.2 Klein Sampling Utilities.

FLUX.2 Klein uses a 32-channel VAE (AutoencoderKLFlux2) instead of the 16-channel VAE
used by FLUX.1. This module provides sampling utilities adapted for FLUX.2.
"""

import math

import torch
from einops import rearrange


def get_noise_flux2(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate noise for FLUX.2 Klein (32 channels).

    FLUX.2 uses a 32-channel VAE, so noise must have 32 channels.
    The spatial dimensions are calculated to allow for packing.

    Args:
        num_samples: Batch size.
        height: Target image height in pixels.
        width: Target image width in pixels.
        device: Target device.
        dtype: Target dtype.
        seed: Random seed.

    Returns:
        Noise tensor of shape (num_samples, 32, latent_h, latent_w).
    """
    rand_device = "cpu"
    rand_dtype = torch.float16

    # FLUX.2 uses 32 latent channels
    # Latent dimensions: height/8, width/8 (from VAE downsampling)
    # Must be divisible by 2 for packing (patchify step)
    latent_h = 2 * math.ceil(height / 16)
    latent_w = 2 * math.ceil(width / 16)

    return torch.randn(
        num_samples,
        32,  # FLUX.2 uses 32 latent channels (vs 16 for FLUX.1)
        latent_h,
        latent_w,
        device=rand_device,
        dtype=rand_dtype,
        generator=torch.Generator(device=rand_device).manual_seed(seed),
    ).to(device=device, dtype=dtype)


def pack_flux2(x: torch.Tensor) -> torch.Tensor:
    """Pack latent image to flattened array of patch embeddings for FLUX.2.

    Patchify + pack in one step:
    For 32-channel input: (B, 32, H, W) -> (B, H/2*W/2, 128)

    Args:
        x: Latent tensor of shape (B, 32, H, W).

    Returns:
        Packed tensor of shape (B, H/2*W/2, 128).
    """
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_flux2(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack flat array of patch embeddings back to latent image for FLUX.2.

    Reverses pack_flux2: (B, H/2*W/2, 128) -> (B, 32, H, W)

    Args:
        x: Packed tensor of shape (B, H/2*W/2, 128).
        height: Target image height in pixels.
        width: Target image width in pixels.

    Returns:
        Latent tensor of shape (B, 32, H, W).
    """
    latent_h = 2 * math.ceil(height / 16)
    latent_w = 2 * math.ceil(width / 16)
    packed_h = latent_h // 2
    packed_w = latent_w // 2

    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_h,
        w=packed_w,
        ph=2,
        pw=2,
    )


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for FLUX.2 schedule shifting.

    Matches the diffusers Flux2Pipeline implementation.
    The mu value controls how much the schedule is shifted towards higher timesteps.

    Args:
        image_seq_len: Number of image tokens (packed_h * packed_w).
        num_steps: Number of denoising steps.

    Returns:
        The empirical mu value.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def get_schedule_flux2(
    num_steps: int,
    image_seq_len: int,
) -> list[float]:
    """Get linear timestep schedule for FLUX.2.

    Returns a linear sigma schedule from 1.0 to 1/num_steps.
    The actual schedule shifting is handled by the FlowMatchEulerDiscreteScheduler
    using the mu parameter and use_dynamic_shifting=True.

    Args:
        num_steps: Number of denoising steps.
        image_seq_len: Number of image tokens (packed_h * packed_w).

    Returns:
        List of linear sigmas from 1.0 to 1/num_steps, plus final 0.0.
    """
    import numpy as np

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    sigmas_list = [float(s) for s in sigmas]
    sigmas_list.append(0.0)

    return sigmas_list


def generate_img_ids_flux2(h: int, w: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate tensor of image position ids for FLUX.2.

    FLUX.2 uses 4D position coordinates (T, H, W, L) for its rotary position embeddings.
    Position IDs must use int64 (long) dtype to avoid NaN in rotary embeddings.

    Args:
        h: Height of image in latent space.
        w: Width of image in latent space.
        batch_size: Batch size.
        device: Device.

    Returns:
        Image position ids tensor of shape (batch_size, h/2*w/2, 4) with int64 dtype.
    """
    packed_h = h // 2
    packed_w = w // 2

    # 4D coordinates: (T, H, W, L)
    img_ids = torch.zeros(packed_h, packed_w, 4, device=device, dtype=torch.long)
    img_ids[..., 1] = torch.arange(packed_h, device=device, dtype=torch.long)[:, None]
    img_ids[..., 2] = torch.arange(packed_w, device=device, dtype=torch.long)[None, :]

    img_ids = img_ids.reshape(1, packed_h * packed_w, 4)
    img_ids = img_ids.expand(batch_size, -1, -1)

    return img_ids
