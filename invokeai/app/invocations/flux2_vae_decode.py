# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 VAE decode invocation for 32-channel latent space.

Decodes 32-channel latents from FLUX.2 back to images. Supports tiled
decoding for memory efficiency on hardware with limited VRAM.
"""

import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.util import FLUX2_LATENT_CHANNELS
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice

# FLUX.2 uses scaling factor of 1.0 (internalized)
FLUX2_VAE_SCALING_FACTOR = 1.0


@invocation(
    "flux2_vae_decode",
    title="VAE Decode - FLUX.2",
    tags=["vae", "decode", "flux2", "image"],
    category="image",
    version="1.0.0",
)
class Flux2VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Decode 32-channel FLUX.2 latents back to an image.

    Uses tiled decoding by default for memory efficiency on consumer hardware.
    The VAE has internalized normalization (scaling factor = 1.0).
    """

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    tiled: bool = InputField(
        default=True,
        description="Use tiled decoding for memory efficiency. Recommended for <24GB VRAM.",
    )
    tile_size: int = InputField(
        default=0,
        description="Tile size in pixels (image space). 0 = use VAE default. Larger = better quality but more VRAM.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)
        vae_info = context.models.load(self.vae.vae)

        image = self.vae_decode(
            vae_info=vae_info,
            latents=latents,
            tiled=self.tiled,
            tile_size=self.tile_size,
        )

        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)

    @classmethod
    def vae_decode(
        cls,
        vae_info: LoadedModel,
        latents: torch.Tensor,
        tiled: bool = True,
        tile_size: int = 0,
    ) -> Image.Image:
        """Decode FLUX.2 latents to an image.

        Args:
            vae_info: Loaded VAE model info
            latents: Latent tensor of shape [B, 32, H/8, W/8]
            tiled: Whether to use tiled decoding for memory efficiency
            tile_size: Tile size in pixels. 0 = use VAE default

        Returns:
            PIL Image
        """
        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        # Validate latent channels
        if latents.shape[1] != FLUX2_LATENT_CHANNELS:
            raise ValueError(
                f"Expected {FLUX2_LATENT_CHANNELS} latent channels for FLUX.2, got {latents.shape[1]}. "
                f"Ensure you are decoding FLUX.2 latents with a compatible VAE."
            )

        latents = latents.to(device=device, dtype=dtype)

        # Apply inverse scaling factor (1.0 for FLUX.2, effectively a no-op)
        latents = latents / FLUX2_VAE_SCALING_FACTOR

        with vae_info as vae:
            # Enable tiling for memory efficiency
            if tiled and hasattr(vae, "enable_tiling"):
                if tile_size > 0:
                    vae.enable_tiling(tile_sample_min_size=tile_size // 8)
                else:
                    vae.enable_tiling()
            elif hasattr(vae, "disable_tiling"):
                vae.disable_tiling()

            # Decode latents to image
            image_tensor = vae.decode(latents).sample

        # Convert tensor to PIL Image
        # Output tensor is in range [-1, 1], convert to [0, 255]
        image_tensor = (image_tensor + 1) / 2
        image_tensor = image_tensor.clamp(0, 1)
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
        image_tensor = image_tensor.permute(1, 2, 0)  # CHW -> HWC
        image_array = (image_tensor.float().cpu().numpy() * 255).round().astype("uint8")

        return Image.fromarray(image_array, mode="RGB")
