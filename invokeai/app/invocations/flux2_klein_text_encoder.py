"""Flux2 Klein Text Encoder Invocation.

Flux2 Klein uses Qwen3 as the text encoder instead of CLIP+T5.
The key difference is that it extracts hidden states from layers (9, 18, 27)
and stacks them together for richer text representations.

This implementation matches the diffusers Flux2KleinPipeline exactly.
"""

from contextlib import ExitStack
from typing import Iterator, Literal, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import Qwen3EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# Qwen3 hidden state layers extracted for Klein-style stacking
KLEIN_EXTRACTION_LAYERS = (9, 18, 27)
KLEIN_MAX_SEQ_LEN = 512


@invocation(
    "flux2_klein_text_encoder",
    title="Prompt - Flux2 Klein",
    tags=["prompt", "conditioning", "flux", "klein", "qwen3"],
    category="conditioning",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Flux2KleinTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for Flux2 Klein image generation.

    Flux2 Klein uses Qwen3 as the text encoder, extracting hidden states from
    layers (9, 18, 27) and stacking them for richer text representations.
    This matches the diffusers Flux2KleinPipeline implementation exactly.
    """

    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    qwen3_encoder: Qwen3EncoderField = InputField(
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    max_seq_len: Literal[256, 512] = InputField(
        default=512,
        description="Max sequence length for the Qwen3 encoder.",
    )
    mask: Optional[TensorField] = InputField(
        default=None,
        description="A mask defining the region that this conditioning prompt applies to.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        qwen3_embeds, pooled_embeds = self._encode_prompt(context)

        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=pooled_embeds, t5_embeds=qwen3_embeds)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _encode_prompt(self, context: InvocationContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using Qwen3 text encoder with Klein-style layer extraction.

        This matches the diffusers Flux2KleinPipeline._get_qwen3_prompt_embeds() exactly.

        Returns:
            Tuple of (stacked_embeddings, pooled_embedding):
            - stacked_embeddings: Hidden states from layers (9, 18, 27) stacked together.
              Shape: (1, seq_len, hidden_size * 3)
            - pooled_embedding: Pooled representation for global conditioning.
              Shape: (1, hidden_size)
        """
        device = TorchDevice.choose_torch_device()

        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (cached_weights, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,
                    dtype=lora_dtype,
                    cached_weights=cached_weights,
                )
            )

            context.util.signal_progress("Running Qwen3 text encoder (Klein)")

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

            # Apply Qwen3 chat template for proper tokenization
            messages = [{"role": "user", "content": self.prompt}]
            text: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    "Text encoder did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported by this model."
                )

            num_hidden_layers = len(outputs.hidden_states)

            # Extract hidden states from specified layers and stack them
            hidden_states_list = []
            for layer_idx in KLEIN_EXTRACTION_LAYERS:
                # Clamp to available layers
                effective_idx = min(layer_idx, num_hidden_layers - 1)
                hidden_states_list.append(outputs.hidden_states[effective_idx])

            # Stack: (batch, num_layers, seq_len, hidden_dim)
            out = torch.stack(hidden_states_list, dim=1)
            out = out.to(dtype=text_encoder.dtype, device=device)

            # Reshape to (batch, seq_len, num_layers * hidden_dim)
            batch_size, num_channels, seq_len, hidden_dim = out.shape
            prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

            # Compute pooled embedding via mean pooling over non-padding tokens
            last_hidden_state = outputs.hidden_states[-1]
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
            sum_embeds = (last_hidden_state * expanded_mask).sum(dim=1)
            num_tokens = expanded_mask.sum(dim=1).clamp(min=1)
            pooled_embeds = sum_embeds / num_tokens

        return prompt_embeds, pooled_embeds

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply to the Qwen3 text encoder."""
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight)
            del lora_info
