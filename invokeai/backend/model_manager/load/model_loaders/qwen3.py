# Copyright (c) 2024, InvokeAI Development Team
"""Loaders for Qwen3 text encoder models (used by FLUX.2-klein)."""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.flux2.text_encoder import Qwen3TextEncoder
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Checkpoint)
class Qwen3EncoderCheckpointLoader(ModelLoader):
    """Load Qwen3 text encoder from single-file checkpoint (safetensors)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Checkpoint_Config):
            raise ValueError("Only Qwen3Encoder_Checkpoint_Config models are supported.")

        model_path = Path(config.path)

        # For single-file checkpoints, we need to work with HuggingFace format
        # This assumes the checkpoint is compatible with Qwen3 model structure
        raise NotImplementedError(
            "Single-file Qwen3 checkpoint loading is not yet implemented. "
            "Please use HuggingFace model directory format instead."
        )


@ModelLoaderRegistry.register(
    base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder
)
class Qwen3EncoderLoader(ModelLoader):
    """Load Qwen3 text encoder from HuggingFace diffusers-like directory structure."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Qwen3Encoder_Config):
            raise ValueError("Only Qwen3Encoder_Qwen3Encoder_Config models are supported.")

        model_path = Path(config.path)

        # Determine output dimension based on config if available
        output_dim = getattr(config, "output_dim", None)

        # Set dtype based on torch device
        if self._torch_dtype == torch.float16:
            # Qwen3 works better in bfloat16 or float32 than float16
            dtype = torch.bfloat16
        else:
            dtype = self._torch_dtype

        # Load Qwen3 encoder
        model = Qwen3TextEncoder(
            model_path=str(model_path),
            output_dim=output_dim,
            dtype=dtype,
            device=self._torch_device,
        )

        return model
