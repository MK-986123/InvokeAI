# Copyright (c) 2024, InvokeAI Development Team
"""Model loaders for FLUX.2 models (e.g., FLUX.2-klein-4B, FLUX.2-klein-9B)."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file

from invokeai.backend.flux2.model import Flux2
from invokeai.backend.flux2.util import get_flux2_transformer_params
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX2_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Flux2CheckpointModel(ModelLoader):
    """Loader for FLUX.2 transformer checkpoints (safetensors format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_FLUX2_Config):
            raise ValueError("Only Main_Checkpoint_FLUX2_Config models are supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_transformer(config)
            case _:
                raise ValueError(
                    f"Only Transformer submodels are currently supported for FLUX.2. "
                    f"Received: {submodel_type.value if submodel_type else 'None'}"
                )

    def _load_transformer(self, config: Main_Checkpoint_FLUX2_Config) -> AnyModel:
        """Load FLUX.2 transformer from safetensors checkpoint."""
        model_path = Path(config.path)

        # Get transformer parameters for this variant
        params = get_flux2_transformer_params(config.variant)

        # Initialize model with empty weights
        with accelerate.init_empty_weights():
            model = Flux2(params)

        # Load state dict
        sd = load_file(model_path)

        # Compute memory requirements and ensure space in cache
        new_sd_size = sum(ten.nelement() * torch.bfloat16.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)

        # Cast all tensors to bfloat16 (FLUX.2 requires bf16 for inference)
        for k in sd.keys():
            sd[k] = sd[k].to(torch.bfloat16)

        # Load weights
        model.load_state_dict(sd, assign=True)

        return model
