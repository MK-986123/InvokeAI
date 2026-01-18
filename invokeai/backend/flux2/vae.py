# Copyright (c) 2024, InvokeAI Development Team
"""VAE for FLUX.2 models with 32 latent channels and tiled processing."""

from typing import Optional

import torch
import torch.nn as nn


class Flux2VAE(nn.Module):
    """Wrapper for AutoencoderKLFlux2 with 32 latent channels and tiled decoding.

    FLUX.2 VAE specifications:
    - Latent channels: 32 (strict, vs FLUX.1's 16)
    - Scaling factor: 1.0 (internalized normalization)
    - Tiled decoding: Enabled by default for <24GB VRAM
    - Memory optimized: Can process large images via tiling

    Args:
        model_path: Path to VAE model (HF diffusers directory or single file).
        enable_tiling: Enable tiled encode/decode for memory efficiency. Default: True.
        tile_size: Tile size for tiled processing. Default: 512.
        device: Torch device for model placement. Default: cuda if available.
        dtype: Torch dtype for model. Default: torch.bfloat16.
    """

    def __init__(
        self,
        model_path: str,
        enable_tiling: bool = True,
        tile_size: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if dtype is None:
            dtype = torch.bfloat16
        self.dtype = dtype

        # Store configuration
        self.enable_tiling = enable_tiling
        self.tile_size = tile_size
        self.latent_channels = 32  # FLUX.2 specific
        self.scaling_factor = 1.0  # FLUX.2 internalized normalization

        # Lazy import to avoid hard dependency on diffusers
        try:
            from diffusers import AutoencoderKLFlux2
        except ImportError:
            raise ImportError(
                "diffusers is required for Flux2VAE. Install via: pip install diffusers"
            )

        # Load VAE model
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

        # Verify latent channels
        if self.vae.latent_channels != self.latent_channels:
            raise ValueError(
                f"VAE has {self.vae.latent_channels} latent channels, "
                f"but FLUX.2 requires exactly 32. Check model compatibility."
            )

        # Enable tiling if requested
        if enable_tiling:
            self.vae.enable_tiling()
            self.vae.tile_latent_channels = 16  # Process 16 channels per tile

        # Set to eval mode
        self.vae.eval()

    def encode(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Encode image to latent distribution.

        Args:
            image: Image tensor [B, 3, H, W] in [-1, 1] range.

        Returns:
            Latent distribution parameters [B, 8, H//8, W//8].
            When using VAE.decode(), use latents[:, :4] for mean.
        """
        with torch.no_grad():
            # Encode to latent distribution
            distribution = self.vae.encode(image).latent_dist
            # Sample from distribution
            latents = distribution.sample()
            # FLUX.2 uses latents directly (32 channels internally managed)
            return latents

    def decode(
        self,
        latents: torch.Tensor,
        num_inference_steps: int = 1,
    ) -> torch.Tensor:
        """Decode latents to image.

        Args:
            latents: Latent tensor [B, 32, H//8, W//8].
            num_inference_steps: Number of decoding steps (usually 1 for VAE). Default: 1.

        Returns:
            Decoded image [B, 3, H, W] in [0, 1] range.
        """
        with torch.no_grad():
            # Decode from latents
            image = self.vae.decode(
                latents,
                num_inference_steps=num_inference_steps,
            ).sample

            # Normalize to [0, 1]
            image = (image + 1.0) / 2.0
            image = image.clamp(0, 1)

            return image

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Move model to device and/or dtype.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Self for chaining.
        """
        if device is not None:
            self.device = device
            self.vae = self.vae.to(device)

        if dtype is not None:
            self.dtype = dtype
            self.vae = self.vae.to(dtype=dtype)

        return self

    def enable_tiling(self) -> None:
        """Enable tiled encode/decode for memory efficiency."""
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
            self.enable_tiling = True

    def disable_tiling(self) -> None:
        """Disable tiled encode/decode (uses more memory but faster)."""
        if hasattr(self.vae, "disable_tiling"):
            self.vae.disable_tiling()
            self.enable_tiling = False

    @property
    def latent_shape_factor(self) -> int:
        """Factor by which image dimensions are reduced in latent space.

        For VAE with 4x downsampling: factor = 8 (image H//8, W//8).
        """
        return 8

    @property
    def config(self) -> dict:
        """Return VAE configuration."""
        return {
            "latent_channels": self.latent_channels,
            "scaling_factor": self.scaling_factor,
            "tile_size": self.tile_size,
            "enable_tiling": self.enable_tiling,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
