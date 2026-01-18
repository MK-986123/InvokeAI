# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 sampling utilities with exponential time-shifting.

FLUX.2-klein uses Rectified Flow with exponential time-shifting for the noise schedule.
The shift parameter controls the trajectory of the flow matching process:
- shift=3.0 is the default for FLUX.2-klein models
- Higher shift values bias the schedule towards high timesteps (semantic structure formation)
"""

import math
from typing import Callable

import torch
from einops import rearrange, repeat

from invokeai.backend.flux2.util import FLUX2_LATENT_CHANNELS


def get_flux2_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate initial noise tensor for FLUX.2.

    FLUX.2 uses 32 latent channels (vs FLUX.1's 16) and in_channels=128 (vs FLUX.1's 64).
    The latent representation is 8x compressed in spatial dimensions.

    Args:
        num_samples: Batch size
        height: Image height in pixels
        width: Image width in pixels
        device: Target device
        dtype: Target dtype
        seed: Random seed for reproducibility

    Returns:
        Noise tensor of shape [B, 32, H/8, W/8]
    """
    # Generate noise on CPU with fixed dtype for consistency, then cast
    rand_device = "cpu"
    rand_dtype = torch.float16

    # FLUX.2 uses 32 latent channels
    # Spatial dimensions are 8x compressed (not 16x like in FLUX.1)
    latent_h = height // 8
    latent_w = width // 8

    # Ensure dimensions are even for packing (2x2 patches)
    latent_h = 2 * math.ceil(latent_h / 2)
    latent_w = 2 * math.ceil(latent_w / 2)

    return torch.randn(
        num_samples,
        FLUX2_LATENT_CHANNELS,  # 32 channels for FLUX.2
        latent_h,
        latent_w,
        device=rand_device,
        dtype=rand_dtype,
        generator=torch.Generator(device=rand_device).manual_seed(seed),
    ).to(device=device, dtype=dtype)


def exponential_time_shift(t: torch.Tensor, shift: float = 3.0) -> torch.Tensor:
    """Apply exponential time-shifting to the timestep schedule.

    FLUX.2-klein uses exponential shifting with shift=3.0 to redistribute
    the discretization steps along the flow trajectory, concentrating them
    where visual changes are most significant.

    Formula: t_shifted = (e^(μ*t) - 1) / (e^μ - 1)
    where μ is the shift parameter (3.0 for FLUX.2-klein).

    Args:
        t: Original timestep values in [0, 1] where 1=noise, 0=clean
        shift: The shift parameter μ (default: 3.0 for FLUX.2-klein)

    Returns:
        Shifted timestep values
    """
    if shift == 0.0:
        return t

    # Handle edge cases to avoid numerical issues
    t = torch.clamp(t, 0.0, 1.0)

    # Apply exponential shift: t_shifted = (e^(μ*t) - 1) / (e^μ - 1)
    exp_shift = math.exp(shift)
    numerator = torch.exp(shift * t) - 1.0
    denominator = exp_shift - 1.0

    return numerator / denominator


def get_flux2_schedule(
    num_steps: int,
    shift: float = 3.0,
    use_exponential_shift: bool = True,
) -> list[float]:
    """Generate the timestep schedule for FLUX.2 with exponential shifting.

    Args:
        num_steps: Number of denoising steps (4 for distilled, 20-50 for base)
        shift: The shift parameter for exponential time-shifting (default: 3.0)
        use_exponential_shift: Whether to apply exponential shifting (default: True)

    Returns:
        List of timestep values from 1.0 (noise) to 0.0 (clean)
    """
    # Generate linear timesteps from 1 to 0 (extra step for final value)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

    if use_exponential_shift and shift > 0:
        # Apply exponential time-shifting
        timesteps = exponential_time_shift(timesteps, shift)

    return timesteps.tolist()


def _find_last_index_ge_val(timesteps: list[float], val: float, eps: float = 1e-6) -> int:
    """Find the last index in timesteps that is >= val.

    Uses epsilon-close equality to avoid potential floating point errors.
    """
    idx = len(list(filter(lambda t: t >= (val - eps), timesteps))) - 1
    assert idx >= 0
    return idx


def clip_flux2_timestep_schedule(
    timesteps: list[float], denoising_start: float, denoising_end: float
) -> list[float]:
    """Clip the timestep schedule to the denoising range with fractional support.

    Args:
        timesteps: The original timestep schedule [1.0, ..., 0.0]
        denoising_start: A value in [0, 1] specifying the start (0.2 means start at t=0.8)
        denoising_end: A value in [0, 1] specifying the end (0.8 means end at t=0.2)

    Returns:
        The clipped timestep schedule
    """
    assert 0.0 <= denoising_start <= 1.0
    assert 0.0 <= denoising_end <= 1.0
    assert denoising_start <= denoising_end

    t_start_val = 1.0 - denoising_start
    t_end_val = 1.0 - denoising_end

    t_start_idx = _find_last_index_ge_val(timesteps, t_start_val)
    t_end_idx = _find_last_index_ge_val(timesteps, t_end_val)

    clipped_timesteps = timesteps[t_start_idx : t_end_idx + 1]

    # Replace first timestep with exact start value
    clipped_timesteps[0] = t_start_val

    # Add final step if needed
    eps = 1e-6
    if clipped_timesteps[-1] > t_end_val + eps:
        clipped_timesteps.append(t_end_val)

    return clipped_timesteps


def pack_flux2(x: torch.Tensor) -> torch.Tensor:
    """Pack FLUX.2 latent image to flattened array of patch embeddings.

    FLUX.2 uses 2x2 patches, resulting in 128 channels per patch (32 * 4).

    Args:
        x: Latent tensor of shape [B, 32, H, W]

    Returns:
        Packed tensor of shape [B, (H/2 * W/2), 128]
    """
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_flux2(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack flat array of patch embeddings to FLUX.2 latent image.

    Args:
        x: Packed tensor of shape [B, L, 128] where L = (H/16) * (W/16)
        height: Original image height in pixels
        width: Original image width in pixels

    Returns:
        Latent tensor of shape [B, 32, H/8, W/8]
    """
    # Calculate latent dimensions
    latent_h = height // 8
    latent_w = width // 8

    # Ensure even dimensions for unpacking
    latent_h = 2 * math.ceil(latent_h / 2)
    latent_w = 2 * math.ceil(latent_w / 2)

    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h // 2,
        w=latent_w // 2,
        ph=2,
        pw=2,
    )


def generate_flux2_img_ids(
    h: int, w: int, batch_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate tensor of image position IDs for FLUX.2.

    FLUX.2 uses 4 axes for RoPE (unlike FLUX.1's 3), with axes_dims_rope=[32, 32, 32, 32].

    Args:
        h: Height of image in latent space (after 8x compression)
        w: Width of image in latent space
        batch_size: Batch size
        device: Target device
        dtype: Target dtype

    Returns:
        Image position IDs of shape [B, L, 4] where L = (H/2) * (W/2)
    """
    if device.type == "mps":
        orig_dtype = dtype
        dtype = torch.float16

    # FLUX.2 uses 4 axes for RoPE: [batch_offset, y, x, extra]
    # After packing (2x2), we have (h/2) * (w/2) patches
    packed_h = h // 2
    packed_w = w // 2

    img_ids = torch.zeros(packed_h, packed_w, 4, device=device, dtype=dtype)

    # Axis 0: batch offset (0 for all image tokens)
    img_ids[..., 0] = 0

    # Axis 1: y position
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_h, device=device, dtype=dtype)[:, None]

    # Axis 2: x position
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_w, device=device, dtype=dtype)[None, :]

    # Axis 3: extra dimension (0 for all image tokens, may be used for temporal)
    img_ids[..., 3] = 0

    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

    if device.type == "mps":
        img_ids = img_ids.to(orig_dtype)

    return img_ids


def generate_flux2_txt_ids(seq_len: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate tensor of text position IDs for FLUX.2.

    Args:
        seq_len: Text sequence length
        batch_size: Batch size
        device: Target device
        dtype: Target dtype

    Returns:
        Text position IDs of shape [B, seq_len, 4]
    """
    if device.type == "mps":
        orig_dtype = dtype
        dtype = torch.float16

    # FLUX.2 uses 4 axes for RoPE
    txt_ids = torch.zeros(seq_len, 4, device=device, dtype=dtype)

    # For text, we typically use sequential positions on axis 0
    txt_ids[:, 0] = torch.arange(seq_len, device=device, dtype=dtype)

    txt_ids = repeat(txt_ids, "l c -> b l c", b=batch_size)

    if device.type == "mps":
        txt_ids = txt_ids.to(orig_dtype)

    return txt_ids
