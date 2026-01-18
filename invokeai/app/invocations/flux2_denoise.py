# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 denoise invocation with Rectified Flow and exponential time-shifting.

FLUX.2-klein uses Rectified Flow with exponential time-shifting for the noise schedule:
- shift=3.0 is the default for optimal flow trajectory
- Distilled models use 4 steps, base models use 20-50 steps
- The exponential shift formula: t_shifted = (e^(μ*t) - 1) / (e^μ - 1)
"""

from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Flux2ConditioningField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import LoRAField, TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2 import (
    FLUX2_KLEIN_DEFAULT_STEPS,
    FLUX2_KLEIN_SHIFT,
    FLUX2_LATENT_CHANNELS,
    Flux2,
    clip_flux2_timestep_schedule,
    generate_flux2_img_ids,
    generate_flux2_txt_ids,
    get_flux2_noise,
    get_flux2_schedule,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Flux2ConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# LoRA prefix for FLUX.2 transformer patching
FLUX2_LORA_TRANSFORMER_PREFIX = ""


@invocation(
    "flux2_denoise",
    title="FLUX.2 Denoise",
    tags=["image", "flux2", "denoise"],
    category="image",
    version="1.0.0",
)
class Flux2DenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX.2-klein transformer model.

    Uses Rectified Flow with exponential time-shifting (shift=3.0) for optimal
    flow trajectory during generation.
    """

    # If latents is provided, this means we are doing image-to-image
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )

    # denoise_mask is used for image-to-image inpainting
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )

    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )

    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)

    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")

    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux2_model,
        input=Input.Connection,
        title="Transformer",
    )

    positive_conditioning: Flux2ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )

    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")

    num_steps: int = InputField(
        default=FLUX2_KLEIN_DEFAULT_STEPS,
        description=f"Number of diffusion steps. Default is {FLUX2_KLEIN_DEFAULT_STEPS} for distilled models.",
    )

    shift: float = InputField(
        default=FLUX2_KLEIN_SHIFT,
        description="Exponential time-shift parameter. 3.0 is default for FLUX.2-klein.",
    )

    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        # Load initial latents if provided (image-to-image)
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Prepare input noise with 32 channels for FLUX.2
        noise = get_flux2_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

        b, _c, latent_h, latent_w = noise.shape

        # Validate noise channels
        if _c != FLUX2_LATENT_CHANNELS:
            raise RuntimeError(
                f"Expected {FLUX2_LATENT_CHANNELS} latent channels for FLUX.2, got {_c}. "
                "This may indicate an incompatible VAE or noise generation issue."
            )

        # Load conditioning data
        pos_conditioning = self._load_conditioning(context, self.positive_conditioning, inference_dtype, device)

        # Calculate timestep schedule with exponential shifting
        timesteps = get_flux2_schedule(
            num_steps=self.num_steps,
            shift=self.shift,
            use_exponential_shift=True,
        )

        # Clip schedule based on denoising range
        timesteps = clip_flux2_timestep_schedule(timesteps, self.denoising_start, self.denoising_end)

        # Prepare input latent
        if init_latents is not None:
            # Image-to-image
            if self.add_noise:
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            # Text-to-image (pure noise)
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            x = noise

        # Short-circuit if only noising
        if len(timesteps) <= 1:
            return x

        # Generate position IDs for FLUX.2 (4 axes for RoPE)
        img_ids = generate_flux2_img_ids(h=latent_h, w=latent_w, batch_size=b, device=device, dtype=inference_dtype)

        txt_ids = generate_flux2_txt_ids(
            seq_len=pos_conditioning.shape[1], batch_size=b, device=device, dtype=inference_dtype
        )

        # Pack latents and noise for transformer
        init_latents_packed = pack_flux2(init_latents) if init_latents is not None else None
        noise_packed = pack_flux2(noise)
        x_packed = pack_flux2(x)

        # Prepare inpaint extension if mask provided
        inpaint_mask = self._prep_inpaint_mask(context, x)
        inpaint_mask_packed = pack_flux2(inpaint_mask) if inpaint_mask is not None else None

        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask_packed is not None:
            assert init_latents_packed is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents_packed,
                inpaint_mask=inpaint_mask_packed,
                noise=noise_packed,
            )

        step_callback = self._build_step_callback(context)

        with ExitStack() as exit_stack:
            # Load transformer
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            assert isinstance(transformer, Flux2)

            # Get model config for quantization check
            transformer_config = context.models.get_config(self.transformer.transformer)
            model_is_quantized = transformer_config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]

            # Apply LoRA patches
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=FLUX2_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Run denoising loop
            x_packed = self._denoise_loop(
                transformer=transformer,
                x=x_packed,
                img_ids=img_ids,
                txt=pos_conditioning,
                txt_ids=txt_ids,
                timesteps=timesteps,
                inpaint_extension=inpaint_extension,
                step_callback=step_callback,
            )

        # Unpack and return
        x = unpack_flux2(x_packed.float(), self.height, self.width)
        return x

    def _denoise_loop(
        self,
        transformer: Flux2,
        x: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        inpaint_extension: RectifiedFlowInpaintExtension | None,
        step_callback: Callable[[PipelineIntermediateState], None],
    ) -> torch.Tensor:
        """Run the FLUX.2 denoising loop with Rectified Flow.

        This implements the Euler method for solving the flow matching ODE.
        """
        total_steps = len(timesteps) - 1

        for step_idx in tqdm(range(total_steps), desc="FLUX.2 Denoise"):
            t_curr = timesteps[step_idx]
            t_next = timesteps[step_idx + 1]

            # Create timestep tensor
            timestep_tensor = torch.full((x.shape[0],), t_curr * 1000, device=x.device, dtype=x.dtype)

            # Forward pass through transformer
            # FLUX.2 predicts the velocity field for flow matching
            velocity = transformer(
                img=x,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timestep_tensor,
                guidance=None,  # FLUX.2-klein distilled models don't use guidance
            )

            # Euler step: x_{t+dt} = x_t + dt * v(x_t, t)
            dt = t_next - t_curr
            x = x + dt * velocity

            # Apply inpainting mask if provided
            if inpaint_extension is not None:
                x = inpaint_extension.merge_intermediate_latents_with_init_latents(x, t_next)

            # Report progress
            step_callback(
                PipelineIntermediateState(
                    step=step_idx + 1,
                    order=1,
                    total_steps=total_steps,
                    timestep=int(t_curr * 1000),
                    latents=x,
                )
            )

        return x

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: Flux2ConditioningField,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Load FLUX.2 conditioning data."""
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        assert len(cond_data.conditionings) == 1
        flux2_conditioning = cond_data.conditionings[0]
        assert isinstance(flux2_conditioning, Flux2ConditioningInfo)
        flux2_conditioning = flux2_conditioning.to(dtype=dtype, device=device)
        return flux2_conditioning.qwen3_embeds

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare inpaint mask for FLUX.2."""
        if self.denoise_mask is None:
            return None

        import torchvision.transforms as tv_transforms
        from torchvision.transforms.functional import resize as tv_resize

        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = 1.0 - mask  # Invert: 0=denoise, 1=preserve

        _, _, latent_h, latent_w = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_h, latent_w],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )

        mask = mask.to(device=latents.device, dtype=latents.dtype)
        return mask.expand_as(latents)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.sd_step_callback(state, BaseModelType.Flux2)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models for the transformer."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
