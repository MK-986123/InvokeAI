"""Latent resampling invocation to reduce banding artifacts for FLUX outputs."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.model import UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES

if TYPE_CHECKING:  # pragma: no cover - imported lazily to avoid heavy dependencies at import time
    from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp ``value`` into the inclusive range [minimum, maximum]."""

    return max(minimum, min(value, maximum))


def _compute_window(strength: float, steps: int) -> Tuple[int, float]:
    """Compute the starting step index and normalized denoising start from a Comfy-style strength."""

    if steps <= 0:
        raise ValueError("steps must be greater than zero")

    clamped_strength = _clamp(strength, 0.0, 1.0)
    # Mirror ComfyUI's behaviour where strength 1.0 covers all steps and strength 0.0 performs no denoise.
    start_step = int(round(steps * (1.0 - clamped_strength)))

    if start_step >= steps:
        start_step = steps - 1

    denoising_start = start_step / steps
    return start_step, denoising_start


@invocation(
    "resample_banding_fix",
    title="Latent Resample Banding Fix",
    tags=["latents", "post-processing", "flux"],
    category="latents",
    version="1.0.0",
    classification=Classification.Beta,
)
class ResampleBandingFixInvocation(BaseInvocation):
    """Run a short denoising pass to resample latents and reduce vertical/horizontal banding."""

    positive_conditioning: Union[ConditioningField, List[ConditioningField]] = InputField(
        description=FieldDescriptions.positive_cond,
        input=Input.Connection,
        ui_order=0,
    )
    negative_conditioning: Union[ConditioningField, List[ConditioningField], None] = InputField(
        default=None,
        description=FieldDescriptions.negative_cond,
        input=Input.Connection,
        ui_order=1,
    )
    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
        ui_order=2,
    )
    unet: UNetField = InputField(
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
        ui_order=3,
    )
    scheduler: SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description=FieldDescriptions.scheduler,
        ui_type=UIType.Scheduler,
        ui_order=4,
    )
    steps: int = InputField(
        default=14,
        ge=1,
        description=FieldDescriptions.steps,
        title="Denoise Steps",
        ui_order=5,
    )
    strength: float = InputField(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Percentage of the denoising window to cover (0 disables, 1 runs the full window).",
        ui_order=6,
    )
    seed: Optional[int] = InputField(
        default=None,
        description=FieldDescriptions.seed,
        ui_order=7,
    )

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Compute the partial denoise window and guard against degenerate configurations.
        if self.strength <= 0:
            context.logger.info("Resample Banding Fix strength is 0; returning input latents unchanged.")
            latents_tensor = context.tensors.load(self.latents.latents_name)
            return LatentsOutput.build(latents_name=self.latents.latents_name, latents=latents_tensor)

        start_step, denoising_start = _compute_window(self.strength, self.steps)
        if start_step < 0:
            raise ValueError("Computed start_step was negative; this should never happen")

        # Delay import until invocation to avoid importing heavy torch dependencies when the node is unused.
        from invokeai.app.invocations.denoise_latents import DenoiseLatentsInvocation

        latents_field = LatentsField(
            latents_name=self.latents.latents_name,
            seed=self.seed if self.seed is not None else self.latents.seed,
        )

        denoise_node = DenoiseLatentsInvocation(
            positive_conditioning=self.positive_conditioning,
            negative_conditioning=self.negative_conditioning,
            noise=None,
            steps=self.steps,
            cfg_scale=1.0,
            denoising_start=denoising_start,
            denoising_end=1.0,
            scheduler=self.scheduler,
            unet=self.unet,
            control=None,
            ip_adapter=None,
            t2i_adapter=None,
            cfg_rescale_multiplier=0.0,
            latents=latents_field,
            denoise_mask=None,
            is_intermediate=self.is_intermediate,
            use_cache=False,
        )

        return denoise_node.invoke(context)


__all__ = ["ResampleBandingFixInvocation", "_compute_window"]
