"""FLUX.2 Klein denoise loop implementation.

Supports:
- Manual Euler stepping (for inpainting/img2img with exact timestep control)
- Diffusers schedulers (Euler, Heun, LCM) with dynamic shifting via mu parameter
- CFG with negative conditioning
- Inpainting via RectifiedFlowInpaintExtension
- Qwen3 text conditioning (no CLIP embeddings)
"""

import inspect
import math
from typing import Callable

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState


def denoise(
    model: torch.nn.Module,
    # Model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    # Sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    cfg_scale: list[float] | None = None,
    # Negative conditioning (for CFG)
    neg_txt: torch.Tensor | None = None,
    neg_txt_ids: torch.Tensor | None = None,
    # Scheduler support
    scheduler: SchedulerMixin | None = None,
    mu: float | None = None,
    # Inpainting
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
) -> torch.Tensor:
    """Run FLUX.2 Klein denoising loop.

    FLUX.2 Klein uses guidance_embeds=False (distilled), so no guidance parameter is used.
    CFG is applied externally through negative conditioning.

    Args:
        model: FLUX.2 transformer model (Flux2 or diffusers Flux2Transformer2DModel).
        img: Packed noisy latents [B, seq_len, 128].
        img_ids: Image position IDs [B, seq_len, 4] (int64).
        txt: Qwen3 text embeddings [B, txt_len, context_in_dim].
        txt_ids: Text position IDs [B, txt_len, 4] (int64).
        timesteps: Sigma schedule (descending from 1 to 0, length = num_steps + 1).
        step_callback: Progress callback.
        cfg_scale: Per-step CFG scale values.
        neg_txt: Negative text embeddings for CFG.
        neg_txt_ids: Negative text position IDs.
        scheduler: Optional diffusers scheduler for alternative sampling.
        mu: Schedule shifting parameter for dynamic shifting schedulers.
        inpaint_extension: Optional inpainting extension.

    Returns:
        Denoised packed latents [B, seq_len, 128].
    """
    use_scheduler = scheduler is not None
    total_steps = len(timesteps) - 1

    if cfg_scale is None:
        cfg_scale = [1.0] * total_steps

    # FLUX.2 Klein: guidance_embeds=False, so guidance_vec is not used
    # The model forward pass uses timesteps and vec (from pooled text) only

    if use_scheduler:
        return _denoise_with_scheduler(
            model=model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            step_callback=step_callback,
            cfg_scale=cfg_scale,
            neg_txt=neg_txt,
            neg_txt_ids=neg_txt_ids,
            scheduler=scheduler,
            mu=mu,
            inpaint_extension=inpaint_extension,
            total_steps=total_steps,
        )
    else:
        return _denoise_euler(
            model=model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            step_callback=step_callback,
            cfg_scale=cfg_scale,
            neg_txt=neg_txt,
            neg_txt_ids=neg_txt_ids,
            inpaint_extension=inpaint_extension,
            total_steps=total_steps,
        )


def _run_model(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    t_vec: torch.Tensor,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the FLUX.2 transformer model.

    Supports both InvokeAI Flux2 model and diffusers Flux2Transformer2DModel.
    """
    # Check if this is a diffusers model (has different API)
    if hasattr(model, "config") and hasattr(model.config, "_class_name"):
        # Diffusers Flux2Transformer2DModel
        output = model(
            hidden_states=img,
            encoder_hidden_states=txt,
            timestep=t_vec,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )
        return output[0] if isinstance(output, tuple) else output
    else:
        # InvokeAI Flux2 model or FLUX.1 Flux model with Klein params
        # The Flux model uses: img, img_ids, txt, txt_ids, timesteps, y, guidance
        if hasattr(model, "params"):
            # InvokeAI model (Flux or Flux2)
            return model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
                y=y if y is not None else torch.zeros(img.shape[0], 1, device=img.device, dtype=img.dtype),
                guidance=None,
            )
        else:
            # Fallback: try generic forward
            return model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t_vec,
            )


def _denoise_euler(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    cfg_scale: list[float],
    neg_txt: torch.Tensor | None,
    neg_txt_ids: torch.Tensor | None,
    inpaint_extension: RectifiedFlowInpaintExtension | None,
    total_steps: int,
) -> torch.Tensor:
    """Manual Euler denoising (exact timestep control, used for inpainting/img2img)."""
    for step_index, (t_curr, t_prev) in tqdm(list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)))):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        pred = _run_model(model, img, img_ids, txt, txt_ids, t_vec)

        # CFG
        step_cfg = cfg_scale[min(step_index, len(cfg_scale) - 1)]
        if not math.isclose(step_cfg, 1.0):
            if neg_txt is None:
                raise ValueError("Negative text conditioning is required when cfg_scale > 1.0.")
            neg_pred = _run_model(model, img, img_ids, neg_txt, neg_txt_ids, t_vec)
            pred = neg_pred + step_cfg * (pred - neg_pred)

        # Euler step
        preview_img = img - t_curr * pred
        img = img + (t_prev - t_curr) * pred

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)
            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

        step_callback(
            PipelineIntermediateState(
                step=step_index + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(t_curr * 1000),
                latents=preview_img,
            ),
        )

    return img


def _denoise_with_scheduler(
    model: torch.nn.Module,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    cfg_scale: list[float],
    neg_txt: torch.Tensor | None,
    neg_txt_ids: torch.Tensor | None,
    scheduler: SchedulerMixin,
    mu: float | None,
    inpaint_extension: RectifiedFlowInpaintExtension | None,
    total_steps: int,
) -> torch.Tensor:
    """Denoising with diffusers scheduler (Euler, Heun, LCM) and dynamic shifting."""
    # Initialize scheduler with sigmas
    is_lcm = scheduler.__class__.__name__ == "FlowMatchLCMScheduler"
    set_timesteps_sig = inspect.signature(scheduler.set_timesteps)

    if not is_lcm and "sigmas" in set_timesteps_sig.parameters:
        # Pass mu for dynamic shifting if supported
        kwargs = {"sigmas": timesteps, "device": img.device}
        if mu is not None and "mu" in set_timesteps_sig.parameters:
            kwargs["mu"] = mu
        scheduler.set_timesteps(**kwargs)
    else:
        num_inference_steps = len(timesteps) - 1
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=img.device)

    num_scheduler_steps = len(scheduler.timesteps)
    user_step = 0

    pbar = tqdm(total=total_steps, desc="FLUX.2 Denoising")
    for step_index in range(num_scheduler_steps):
        timestep = scheduler.timesteps[step_index]
        t_curr = timestep.item() / scheduler.config.num_train_timesteps
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        is_heun = hasattr(scheduler, "state_in_first_order")
        in_first_order = scheduler.state_in_first_order if is_heun else True

        pred = _run_model(model, img, img_ids, txt, txt_ids, t_vec)

        # CFG
        step_cfg = cfg_scale[min(user_step, len(cfg_scale) - 1)]
        if not math.isclose(step_cfg, 1.0):
            if neg_txt is None:
                raise ValueError("Negative text conditioning is required when cfg_scale > 1.0.")
            neg_pred = _run_model(model, img, img_ids, neg_txt, neg_txt_ids, t_vec)
            pred = neg_pred + step_cfg * (pred - neg_pred)

        # Scheduler step
        step_output = scheduler.step(model_output=pred, timestep=timestep, sample=img)
        img = step_output.prev_sample

        # Inpainting merge
        if inpaint_extension is not None:
            if step_index + 1 < len(scheduler.sigmas):
                t_prev = scheduler.sigmas[step_index + 1].item()
            else:
                t_prev = 0.0
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)

        # Progress callback (handle Heun's double steps)
        if is_heun:
            if not in_first_order:
                user_step += 1
                if user_step <= total_steps:
                    pbar.update(1)
                    preview_img = img - t_curr * pred
                    step_callback(
                        PipelineIntermediateState(
                            step=user_step,
                            order=2,
                            total_steps=total_steps,
                            timestep=int(t_curr * 1000),
                            latents=preview_img,
                        ),
                    )
        else:
            user_step += 1
            if user_step <= total_steps:
                pbar.update(1)
                preview_img = img - t_curr * pred
                step_callback(
                    PipelineIntermediateState(
                        step=user_step,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t_curr * 1000),
                        latents=preview_img,
                    ),
                )

    pbar.close()
    return img
