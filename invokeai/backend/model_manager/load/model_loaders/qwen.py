from typing import Optional

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Qwen, type=ModelType.Main, format=ModelFormat.Diffusers)
class QwenDiffusersModel(GenericDiffusersLoader):
    """Class to load Qwen main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Qwen models.")

        # If submodel_type is None, it means we are loading the whole pipeline (or using a logic that doesn't need submodels)
        # But GenericDiffusersLoader expects to load either a submodel or the whole pipeline if get_hf_load_class handles it.
        # Let's check how CogView4 does it.
        # CogView4 raises exception if submodel_type is None.

        # However, for Qwen Image, it seems to be a single transformer model inside a diffusers structure?
        # The config.json has _class_name: QwenImageTransformer2DModel.

        # If I look at GenericDiffusersLoader.get_hf_load_class:
        # It checks if submodel_type is provided. If not, it checks config.json for _class_name.
        # If _class_name is present, it loads that class.

        # So GenericDiffusersLoader might actually work out of the box if we just register it?
        # But CogView4 needed a custom loader to force bfloat16.
        # Let's assume for now we don't need to force anything and just use GenericDiffusersLoader logic.
        # But since I need to register it for Qwen base, I need a class.

        return super()._load_model(config, submodel_type)
