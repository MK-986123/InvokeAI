# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 VAE encode invocation for 32-channel latent space.

FLUX.2-klein uses a 32-channel VAE (AutoencoderKLFlux2) which differs from:
- Standard SD VAEs (4 channels)
- FLUX.1 VAE (16 channels)

Key features:
- 32 latent channels for higher fidelity reconstruction
- Scaling factor of 1.0 (internalized normalization)
- 8x spatial compression
"""

import einops
import torch
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.util import FLUX2_LATENT_CHANNELS
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice

# FLUX.2 uses scaling factor of 1.0 (internalized)
FLUX2_VAE_SCALING_FACTOR = 1.0


@invocation(
    "flux2_vae_encode",
    title="VAE Encode - FLUX.2",
    tags=["vae", "encode", "flux2", "image"],
    category="image",
    version="1.0.0",
)
class Flux2VaeEncodeInvocation(BaseInvocation):
    """Encode an image into 32-channel latent space for FLUX.2.

    FLUX.2 uses a 32-channel VAE with scaling factor 1.0 (internalized normalization).
    The spatial dimensions are compressed by 8x.
    """

    image: ImageField = InputField(description="The image to encode.")
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        image = context.images.get_pil(self.image.image_name)
        vae_info = context.models.load(self.vae.vae)

        latents = self.vae_encode(vae_info=vae_info, image=image)

        latents = latents.to("cpu")
        name = context.tensors.save(tensor=latents)

        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    @classmethod
    def vae_encode(cls, vae_info: LoadedModel, image: Image.Image) -> torch.Tensor:
        """Encode an image to FLUX.2 32-channel latent space.

        Args:
            vae_info: Loaded VAE model info
            image: PIL Image to encode

        Returns:
            Latent tensor of shape [1, 32, H/8, W/8]
        """
        # Convert image to tensor
        image = image.convert("RGB")
        image_tensor = image_resized_to_grid_as_tensor(image, multiple_of=8)

        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        return cls.vae_encode_tensor(vae_info=vae_info, image_tensor=image_tensor)

    @classmethod
    def vae_encode_tensor(cls, vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode an image tensor to FLUX.2 latent space.

        Args:
            vae_info: Loaded VAE model info
            image_tensor: Image tensor of shape [B, 3, H, W] in range [-1, 1]

        Returns:
            Latent tensor of shape [B, 32, H/8, W/8]
        """
        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        image_tensor = image_tensor.to(device=device, dtype=dtype)

        with vae_info as vae:
            # Check if VAE supports tiling
            if hasattr(vae, "enable_tiling"):
                # Enable tiling for memory efficiency
                vae.enable_tiling()

            # Encode image
            latent_dist = vae.encode(image_tensor).latent_dist
            latents = latent_dist.sample()

            # Apply scaling factor (1.0 for FLUX.2, effectively a no-op)
            latents = latents * FLUX2_VAE_SCALING_FACTOR

        # Validate channel count
        if latents.shape[1] != FLUX2_LATENT_CHANNELS:
            raise RuntimeError(
                f"Expected {FLUX2_LATENT_CHANNELS} latent channels for FLUX.2 VAE, got {latents.shape[1]}. "
                f"Ensure you are using a compatible FLUX.2 VAE."
            )

        return latents
