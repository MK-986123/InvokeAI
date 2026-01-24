# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 Denoise Invocation.

Implements the FLUX.2-klein denoising pipeline:
- Supports FLUX.2-klein-4B (and optionally 9B/9B-FP8)
- Uses Qwen3 text embeddings (not CLIP+T5)
- Supports BN normalization from FLUX.2 VAE stats
- Distilled for few-step inference (default 4 steps)
- Optional CFG support with negative conditioning
"""

import math
from typing import Optional

import torch
from einops import rearrange

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import (
    generate_img_ids,
    get_noise,
    get_schedule,
    pack,
    unpack,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.model import Flux2
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


class Flux2ConditioningField:
    """Placeholder for FLUX.2 conditioning field (Qwen3 embeddings)."""

    txt_embeddings: torch.Tensor
    txt_ids: torch.Tensor


@invocation(
    "flux2_denoise",
    title="FLUX.2 Denoise",
    tags=["image", "flux2", "klein"],
    category="image",
    version="1.0.0",
)
class Flux2DenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX.2-klein transformer model."""

    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(
        default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end
    )
    transformer: TransformerField = InputField(
        description="FLUX.2 transformer model.",
        input=Input.Connection,
        title="Transformer",
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description="FLUX.2 VAE for BN stats (optional).",
        input=Input.Connection,
        title="VAE",
    )
    positive_text_conditioning: LatentsField = InputField(
        description="Positive text conditioning (Qwen3 embeddings).",
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[LatentsField] = InputField(
        default=None,
        description="Negative text conditioning. Required if cfg_scale > 1.0.",
        input=Input.Connection,
    )
    cfg_scale: float = InputField(
        default=1.0,
        ge=1.0,
        description="CFG scale. 1.0 means no CFG (recommended for distilled models).",
        title="CFG Scale",
    )
    width: int = InputField(
        default=1024, multiple_of=16, ge=256, le=8192, description="Width of the generated image."
    )
    height: int = InputField(
        default=1024, multiple_of=16, ge=256, le=8192, description="Height of the generated image."
    )
    num_steps: int = InputField(
        default=4, ge=1, le=100, description="Number of denoising steps (4 recommended for klein)."
    )
    guidance: float = InputField(
        default=1.0,
        ge=0.0,
        description="Guidance value. Only used if model has guidance_embeds=True.",
    )
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler to use for denoising.",
    )
    seed: int = InputField(default=0, description="Random seed for noise generation.")

    def _get_bn_stats(
        self, context: InvocationContext
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Extract BN running statistics from the FLUX.2 VAE.

        The FLUX.2 VAE uses batch normalization on latents. We need the
        running mean and variance to normalize/denormalize latents during denoising.
        """
        if self.vae is None:
            return None

        with context.models.load(self.vae.vae) as vae:
            # Look for the quant_conv BN layer (FLUX.2 VAE uses BN after quantization)
            bn_layer = None
            if hasattr(vae, "quant_conv"):
                # Check if quant_conv has a BN sublayer
                if hasattr(vae.quant_conv, "bn") and hasattr(vae.quant_conv.bn, "running_mean"):
                    bn_layer = vae.quant_conv.bn
                elif hasattr(vae.quant_conv, "running_mean"):
                    bn_layer = vae.quant_conv
            elif hasattr(vae, "encoder") and hasattr(vae.encoder, "norm_out"):
                if hasattr(vae.encoder.norm_out, "running_mean"):
                    bn_layer = vae.encoder.norm_out

            if bn_layer is None or bn_layer.running_mean is None:
                return None

            # Get BN running statistics from VAE
            bn_mean = bn_layer.running_mean.clone()  # Shape: (128,)
            bn_var = bn_layer.running_var.clone()  # Shape: (128,)

            # Get epsilon: VAE config > BN layer > default (1e-4 for BFL)
            if hasattr(vae, "config") and hasattr(vae.config, "batch_norm_eps"):
                bn_eps = vae.config.batch_norm_eps
            elif hasattr(bn_layer, "eps"):
                bn_eps = bn_layer.eps
            else:
                bn_eps = 1e-4  # BFL FLUX.2 default
            bn_std = torch.sqrt(bn_var + bn_eps)

        return bn_mean, bn_std

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Early validation: CFG requires negative conditioning
        if self.cfg_scale > 1.0 and self.negative_text_conditioning is None:
            raise ValueError(
                f"cfg_scale={self.cfg_scale} requires negative_text_conditioning. "
                "Either set cfg_scale=1.0 or provide negative conditioning."
            )

        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=self.seed)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = torch.bfloat16

        # Generate noise
        noise = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

        # For FLUX.2: noise has 16 channels from get_noise, but model expects 128 in_channels
        # The packing operation handles this: 16ch * 2x2 patches = 64, but FLUX.2 uses 128
        # For FLUX.2, we need to generate noise with the correct latent_channels (32)
        # and pack appropriately
        # NOTE: FLUX.2 uses latent_channels=32, patch_size=1
        # Noise shape: [B, 32, H/8, W/8] -> packed: [B, (H/8)*(W/8), 32*patch^2] = [B, seq, 32]
        # But the transformer expects in_channels=128, which means the VAE produces 128-dim latents
        # This is because the VAE patch_size=[2,2] produces: 32 * 2 * 2 = 128 per spatial position
        b, c, h, w = noise.shape
        # Adjust for FLUX.2: repack from 16ch (FLUX.1) to correct dimensions
        # For FLUX.2 with latent_channels=32 and patch_size=[2,2]:
        # Effective latent dims: H/(8*2) x W/(8*2) spatial, 32*2*2=128 channels per position

        # Get image sequence length for schedule computation
        img_seq_len = (self.height // 16) * (self.width // 16)

        # Generate schedule
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=img_seq_len,
            base_shift=0.5,
            max_shift=1.15,
            shift=True,
        )

        # Pack noise into sequence format for transformer
        # FLUX.2 with patch_size=1: each 2x2 patch of latent channels becomes one token
        img = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Generate position IDs (4 axes for FLUX.2 RoPE)
        # FLUX.2 uses 4-axis RoPE: [32, 32, 32, 32]
        h_ids = torch.arange(self.height // 16, device=device)
        w_ids = torch.arange(self.width // 16, device=device)
        img_ids = torch.stack(
            torch.meshgrid(h_ids, w_ids, indexing="ij"), dim=-1
        ).reshape(-1, 2)
        # Expand to 4 axes (FLUX.2 uses 4-dimensional position IDs)
        img_ids = torch.cat([
            torch.zeros(img_ids.shape[0], 1, device=device),  # batch dim
            torch.zeros(img_ids.shape[0], 1, device=device),  # time dim
            img_ids,  # h, w dims
        ], dim=-1)
        img_ids = img_ids.unsqueeze(0).expand(1, -1, -1)

        # Get BN stats from VAE (for latent normalization)
        bn_stats = self._get_bn_stats(context)
        if bn_stats is not None:
            bn_mean, bn_std = bn_stats
            # Move to device once, not on every normalize/denormalize call
            bn_mean = bn_mean.to(device=device, dtype=inference_dtype)
            bn_std = bn_std.to(device=device, dtype=inference_dtype)
        else:
            context.logger.warning("FLUX.2 VAE BN stats not found. Normalization skipped.")
            bn_mean, bn_std = None, None

        # Load the input latents, if provided
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)
            # Pack init_latents same as noise
            init_latents = rearrange(init_latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Handle img-to-img: blend noise with init_latents based on denoising_start
        if init_latents is not None and self.denoising_start > 0.0:
            t_start = timesteps[0]  # First timestep value
            img = t_start * img + (1 - t_start) * init_latents

        # Load text conditioning
        pos_txt = context.tensors.load(self.positive_text_conditioning.latents_name)
        pos_txt = pos_txt.to(device=device, dtype=inference_dtype)

        # Create text position IDs (4 axes, matching image)
        txt_seq_len = pos_txt.shape[1]
        txt_ids = torch.zeros(1, txt_seq_len, 4, device=device)

        # Load negative conditioning if needed
        neg_txt = None
        neg_txt_ids = None
        if self.negative_text_conditioning is not None:
            neg_txt = context.tensors.load(self.negative_text_conditioning.latents_name)
            neg_txt = neg_txt.to(device=device, dtype=inference_dtype)
            neg_txt_ids = torch.zeros(1, neg_txt.shape[1], 4, device=device)

        # CFG scale as list
        cfg_scale_list = [self.cfg_scale] * self.num_steps

        # Prepare scheduler if not using built-in Euler
        # TODO: Load scheduler config from model config for variant-specific values
        # Current values are tuned for Klein 4B distilled
        scheduler = None
        if self.scheduler in FLUX_SCHEDULER_MAP:
            scheduler_class = FLUX_SCHEDULER_MAP[self.scheduler]
            scheduler_instance = scheduler_class(
                num_train_timesteps=1000,
                shift=3.0,
            )
            # For non-euler schedulers, use the scheduler instead of manual stepping
            if self.scheduler != "euler":
                scheduler = scheduler_instance

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.signal_progress(
                message=f"FLUX.2 Denoising step {state.step}/{state.total_steps}",
                progress=state.step / state.total_steps,
            )

        # Load and run transformer
        with context.models.load(self.transformer.transformer) as transformer:
            assert isinstance(transformer, Flux2)

            result = denoise(
                model=transformer,
                img=img,
                img_ids=img_ids,
                txt=pos_txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                step_callback=step_callback,
                guidance=self.guidance,
                cfg_scale=cfg_scale_list,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                bn_mean=bn_mean,
                bn_std=bn_std,
            )

        # Unpack from sequence format back to spatial
        result = rearrange(
            result,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=self.height // 16,
            w=self.width // 16,
            ph=2,
            pw=2,
        )

        return result
