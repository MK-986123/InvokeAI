"""FLUX.2 VAE Decode Invocation.

Decodes FLUX.2 latents to images using AutoencoderKLFlux2.

Key differences from FLUX.1 VAE:
- 32 latent channels (vs 16)
- patch_size=[2, 2]
- Uses quant_conv/post_quant_conv
- Output range is [-1, 1]
"""

import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
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
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_vae_decode",
    title="Latents to Image - FLUX.2",
    tags=["latents", "image", "vae", "l2i", "flux2"],
    category="latents",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Flux2VaeDecodeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Generates an image from FLUX.2 latents using AutoencoderKLFlux2."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    vae: VAEField = InputField(
        description="FLUX.2 VAE model (AutoencoderKLFlux2).",
        input=Input.Connection,
    )

    def _vae_decode(self, vae: torch.nn.Module, latents: torch.Tensor) -> Image.Image:
        """Decode latents using the FLUX.2 VAE.

        The FLUX.2 VAE (AutoencoderKLFlux2) expects:
        - Input: [B, 32, H/8, W/8] latent tensor
        - Output: [B, 3, H, W] image tensor in [-1, 1] range
        """
        vae_dtype = next(iter(vae.parameters())).dtype
        device = TorchDevice.choose_torch_device()
        latents = latents.to(device=device, dtype=vae_dtype)

        # Decode using the VAE
        # Handle both diffusers API (.decode()) and custom API
        if hasattr(vae, "decode"):
            decoded = vae.decode(latents, return_dict=False)[0]
        else:
            decoded = vae(latents)

        # Convert from [-1, 1] to [0, 1] then to [0, 255] PIL image
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = rearrange(img[0], "c h w -> h w c")
        img_pil = Image.fromarray((img * 255).byte().cpu().numpy())
        return img_pil

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.tensors.load(self.latents.latents_name)

        vae_info = context.models.load(self.vae.vae)
        with vae_info.model_on_device() as (_, vae):
            context.util.signal_progress("Running FLUX.2 VAE Decode")
            image = self._vae_decode(vae=vae, latents=latents)

        TorchDevice.empty_cache()
        image_dto = context.images.save(image=image)
        return ImageOutput.build(image_dto)
