# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 model loader invocation.

Loads a FLUX.2-klein transformer model along with its required components:
- FLUX.2 Transformer (4B or 9B variant)
- Qwen3 Text Encoder (for text conditioning)
- VAE (32-channel latent space)
"""

from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.util import get_flux2_max_seq_length
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("flux2_model_loader_output")
class Flux2ModelLoaderOutput(BaseInvocationOutput):
    """FLUX.2 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    max_seq_len: Literal[256, 512] = OutputField(
        description="The max sequence length for the Qwen3 encoder (512 for FLUX.2-klein).",
        title="Max Seq Length",
    )


@invocation(
    "flux2_model_loader",
    title="Main Model - FLUX.2",
    tags=["model", "flux2"],
    category="model",
    version="1.0.0",
)
class Flux2ModelLoaderInvocation(BaseInvocation):
    """Loads a FLUX.2-klein transformer model with its required submodels.

    FLUX.2-klein requires:
    - Transformer: The main FLUX.2 flow matching model (4B or 9B)
    - Qwen3 Encoder: Text encoder using Qwen3 architecture
    - VAE: 32-channel variational autoencoder for latent encoding/decoding
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux2_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux2,
        ui_model_type=ModelType.Main,
    )

    qwen3_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Direct,
        title="Qwen3 Encoder",
        ui_model_type=ModelType.Qwen3Encoder,
    )

    vae_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.vae_model,
        title="VAE",
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> Flux2ModelLoaderOutput:
        # Validate that all models exist
        for key in [self.model.key, self.qwen3_encoder_model.key, self.vae_model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        # Get transformer config to determine variant
        transformer_config = context.models.get_config(self.model)
        if not isinstance(transformer_config, Main_Checkpoint_FLUX2_Config):
            raise ValueError(
                f"Expected FLUX.2 model config, got {type(transformer_config).__name__}. "
                "Please select a valid FLUX.2-klein model."
            )

        # Create model identifier fields
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Create Qwen3 encoder field with tokenizer and text_encoder submodels
        tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        # Get max sequence length for this variant
        max_seq_len = get_flux2_max_seq_length(transformer_config.variant)

        return Flux2ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
            max_seq_len=max_seq_len,
        )
