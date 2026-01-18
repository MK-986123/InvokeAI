# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 text encoder invocation using Qwen3 with 3-layer hidden state concatenation.

FLUX.2-klein uses Qwen3 text encoder with a specific embedding extraction strategy:
- The last 3 hidden layers (L, L-1, L-2) are concatenated along the feature dimension
- This produces embeddings with joint_attention_dim of 7680 (3 * 2560) for 4B model
- Or 12288 for 9B model with larger Qwen3 encoder

Key implementation details:
- head_dim=128 must be explicitly set (not calculated as hidden_size/num_heads=80)
- Concatenation is done along dim=-1 (feature dimension)
- The transformer expects joint_attention_dim matching concatenated output
"""

from contextlib import ExitStack
from typing import Iterator, Literal, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Flux2ConditioningField,
    Input,
    InputField,
    OutputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux2.util import get_flux2_max_seq_length
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Flux2ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# Number of hidden layers to concatenate for FLUX.2 embeddings
FLUX2_HIDDEN_LAYERS_TO_CONCAT = 3


@invocation_output("flux2_conditioning_output")
class Flux2ConditioningOutput(BaseInvocationOutput):
    """Output type for FLUX.2 conditioning."""

    conditioning: Flux2ConditioningField = OutputField(
        description=FieldDescriptions.flux2_conditioning, title="Conditioning"
    )


@invocation(
    "flux2_text_encoder",
    title="Prompt - FLUX.2",
    tags=["prompt", "conditioning", "flux2"],
    category="conditioning",
    version="1.0.0",
)
class Flux2TextEncoderInvocation(BaseInvocation):
    """Encodes a prompt for FLUX.2 using Qwen3 text encoder with 3-layer concatenation.

    FLUX.2-klein uses a unique embedding strategy: concatenating the last 3 hidden layers
    of the Qwen3 encoder (layers L, L-1, L-2) along the feature dimension. This produces
    embeddings with dimension 7680 (3 * 2560) for the 4B model.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        title="Max Seq Length",
        description="Maximum sequence length for text encoding. 512 recommended for FLUX.2-klein.",
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to (for regional prompting).",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> Flux2ConditioningOutput:
        # Encode the prompt with 3-layer concatenation
        qwen3_embeds = self._encode_prompt_with_concatenation(context)

        # Save conditioning data
        conditioning_data = ConditioningFieldData(conditionings=[Flux2ConditioningInfo(qwen3_embeds=qwen3_embeds)])
        conditioning_name = context.conditioning.save(conditioning_data)

        return Flux2ConditioningOutput(
            conditioning=Flux2ConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _encode_prompt_with_concatenation(self, context: InvocationContext) -> torch.Tensor:
        """Encode prompt using Qwen3 with 3-layer hidden state concatenation.

        This is the key innovation for FLUX.2: instead of using just the last hidden state,
        we concatenate the last 3 hidden layers to capture a richer hierarchy of features
        (from syntax to semantics).

        Returns:
            torch.Tensor: Concatenated embeddings of shape (batch, seq_len, joint_attention_dim)
            where joint_attention_dim = 3 * hidden_size (e.g., 7680 = 3 * 2560 for 4B)
        """
        prompt = self.prompt
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            # Apply LoRA models if any
            if self.qwen3_encoder.loras:
                exit_stack.enter_context(
                    LayerPatcher.apply_smart_model_patches(
                        model=text_encoder,
                        patches=self._lora_iterator(context),
                        prefix="",  # Qwen3 uses direct key matching
                        dtype=inference_dtype,
                    )
                )

            context.util.signal_progress("Running Qwen3 text encoder for FLUX.2")

            if not isinstance(text_encoder, PreTrainedModel):
                raise TypeError(
                    f"Expected PreTrainedModel for text encoder, got {type(text_encoder).__name__}. "
                    "The Qwen3 encoder model may be corrupted or incompatible."
                )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise TypeError(
                    f"Expected PreTrainedTokenizerBase for tokenizer, got {type(tokenizer).__name__}. "
                    "The Qwen3 tokenizer may be corrupted or incompatible."
                )

            # Tokenize the prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            if not isinstance(text_input_ids, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for input_ids, got {type(text_input_ids).__name__}.")
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor for attention_mask, got {type(attention_mask).__name__}.")

            # Check for truncation
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] > text_input_ids.shape[-1]:
                removed_text = tokenizer.batch_decode(untruncated_ids[:, self.max_seq_len:])
                context.logger.warning(
                    f"Prompt truncated. The following was removed: {removed_text}"
                )

            # Forward pass with hidden states output
            outputs = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
            )

            # Validate hidden_states output
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    "Text encoder did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported by this model."
                )

            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states)

            if num_layers < FLUX2_HIDDEN_LAYERS_TO_CONCAT:
                raise RuntimeError(
                    f"Expected at least {FLUX2_HIDDEN_LAYERS_TO_CONCAT} hidden states, got {num_layers}. "
                    "The Qwen3 model may be incompatible with FLUX.2."
                )

            # Extract and concatenate the last 3 hidden layers
            # hidden_states[-1] = layer L (last layer)
            # hidden_states[-2] = layer L-1
            # hidden_states[-3] = layer L-2
            layers_to_concat = [
                hidden_states[-1],  # Layer L (last)
                hidden_states[-2],  # Layer L-1
                hidden_states[-3],  # Layer L-2
            ]

            # Concatenate along feature dimension (dim=-1)
            # Shape: (batch, seq_len, hidden_size) x 3 -> (batch, seq_len, 3 * hidden_size)
            concatenated_embeds = torch.cat(layers_to_concat, dim=-1)

            # Convert to inference dtype
            concatenated_embeds = concatenated_embeds.to(dtype=inference_dtype)

            # Log the resulting shape for debugging
            context.logger.debug(
                f"FLUX.2 embeddings shape: {concatenated_embeds.shape} "
                f"(joint_attention_dim={concatenated_embeds.shape[-1]})"
            )

        return concatenated_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the Qwen3 text encoder."""
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
