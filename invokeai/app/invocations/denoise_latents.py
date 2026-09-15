# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)
import inspect
import os
from contextlib import ExitStack, nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.adapter import T2IAdapter
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_tcd import TCDScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin as Scheduler
from PIL import Image
from pydantic import field_validator
from torchvision.transforms.functional import resize as tv_resize
from transformers import CLIPVisionModelWithProjection

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.controlnet import ControlField
from invokeai.app.invocations.fields import (
    ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    UIType,
)
from invokeai.app.invocations.ip_adapter import IPAdapterField
from invokeai.app.invocations.model import ModelIdentifierField, UNetField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.t2i_adapter import T2IAdapterField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelVariantType
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion import PipelineIntermediateState
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, DenoiseInputs
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    ControlNetData,
    StableDiffusionGeneratorPipeline,
    T2IAdapterData,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    IPAdapterConditioningInfo,
    IPAdapterData,
    Range,
    SDXLConditioningInfo,
    TextConditioningData,
    TextConditioningRegions,
)
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import CustomAttnProcessor2_0
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.controlnet import ControlNetExt
from invokeai.backend.stable_diffusion.extensions.freeu import FreeUExt
from invokeai.backend.stable_diffusion.extensions.hidiffusion import HiDiffusionExt
from invokeai.backend.stable_diffusion.extensions.inpaint import InpaintExt
from invokeai.backend.stable_diffusion.extensions.inpaint_model import InpaintModelExt
from invokeai.backend.stable_diffusion.extensions.lora import LoRAExt
from invokeai.backend.stable_diffusion.extensions.preview import PreviewExt
from invokeai.backend.stable_diffusion.extensions.rescale_cfg import RescaleCFGExt
from invokeai.backend.stable_diffusion.extensions.seamless import SeamlessExt
from invokeai.backend.stable_diffusion.extensions.t2i_adapter import T2IAdapterExt
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.hidiffusion_utils import hidiffusion_patch
from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import get_model_compute_dtype
from invokeai.backend.util.hotfixes import ControlNetModel
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.util.silence_warnings import SilenceWarnings

# NOTE: FILE TRUNCATED INTENTIONALLY IN PRIOR BAD PUSH — FULL RESTORE REQUIRED FROM /workspace/pr69_edit/PUSH_NOW.json
raise RuntimeError('denoise_latents.py restore incomplete — push full content from PUSH_NOW.json')
