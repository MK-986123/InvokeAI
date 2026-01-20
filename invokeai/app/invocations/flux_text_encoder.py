from contextlib import ExitStack
from typing import Iterator, Literal, Optional, Tuple, Union

import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import CLIPField, Qwen3EncoderField, T5EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.conditioner import HFEncoder
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_CLIP_PREFIX, FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_QWEN3_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice

QWEN3_MAX_SEQ_LEN = 512


@invocation(
    "flux_text_encoder",
    title="Prompt - FLUX",
    tags=["prompt", "conditioning", "flux"],
    category="conditioning",
    version="1.1.2",
)
class FluxTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a flux image."""

    clip: CLIPField = InputField(
        title="CLIP",
        description=FieldDescriptions.clip,
        input=Input.Connection,
    )
    t5_encoder: Optional[T5EncoderField] = InputField(
        default=None,
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    qwen3_encoder: Optional[Qwen3EncoderField] = InputField(
        default=None,
        title="Qwen3 Encoder",
        description=FieldDescriptions.qwen3_encoder,
        input=Input.Connection,
    )
    t5_max_seq_len: Literal[256, 512] = InputField(
        description="Max sequence length for the T5 encoder. Expected to be 256 for FLUX schnell models and 512 for FLUX dev models."
    )
    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    mask: Optional[TensorField] = InputField(
        default=None, description="A mask defining the region that this conditioning prompt applies to."
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        # Note: The T5 and CLIP encoding are done in separate functions to ensure that all model references are locally
        # scoped. This ensures that the T5 model can be freed and gc'd before loading the CLIP model (if necessary).
        qwen3_embeddings: torch.Tensor | None = None
        qwen3_txt_ids: torch.Tensor | None = None
        t5_embeddings: torch.Tensor | None = None
        if self.qwen3_encoder is not None:
            qwen3_embeddings, qwen3_txt_ids = self._qwen3_encode(context)
        elif self.t5_encoder is not None:
            t5_embeddings = self._t5_encode(context)
        else:
            raise ValueError("Either a T5 encoder or a Qwen3 encoder must be provided for FLUX conditioning.")
        clip_embeddings = self._clip_encode(context)
        conditioning_data = ConditioningFieldData(
            conditionings=[
                FLUXConditioningInfo(
                    clip_embeds=clip_embeddings,
                    t5_embeds=t5_embeddings,
                    qwen3_embeds=qwen3_embeddings,
                    qwen3_txt_ids=qwen3_txt_ids,
                )
            ]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _t5_encode(self, context: InvocationContext) -> torch.Tensor:
        prompt = [self.prompt]

        if self.t5_encoder is None:
            raise ValueError("T5 encoder is required for T5-based FLUX conditioning.")

        t5_encoder_info = context.models.load(self.t5_encoder.text_encoder)
        t5_encoder_config = t5_encoder_info.config
        assert t5_encoder_config is not None

        with (
            t5_encoder_info.model_on_device() as (cached_weights, t5_text_encoder),
            context.models.load(self.t5_encoder.tokenizer) as t5_tokenizer,
            ExitStack() as exit_stack,
        ):
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(t5_tokenizer, (T5Tokenizer, T5TokenizerFast))

            # Determine if the model is quantized.
            # If the model is quantized, then we need to apply the LoRA weights as sidecar layers. This results in
            # slower inference than direct patching, but is agnostic to the quantization format.
            if t5_encoder_config.format in [ModelFormat.T5Encoder, ModelFormat.Diffusers]:
                model_is_quantized = False
            elif t5_encoder_config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported model format: {t5_encoder_config.format}")

            # Apply LoRA models to the T5 encoder.
            # Note: We apply the LoRA after the encoder has been moved to its target device for faster patching.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=t5_text_encoder,
                    patches=self._t5_lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,
                    dtype=t5_text_encoder.dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            t5_encoder = HFEncoder(t5_text_encoder, t5_tokenizer, False, self.t5_max_seq_len)

            if context.config.get().log_tokenization:
                self._log_t5_tokenization(context, t5_tokenizer)

            context.util.signal_progress("Running T5 encoder")
            prompt_embeds = t5_encoder(prompt)

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds

    def _qwen3_encode(self, context: InvocationContext) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = self.prompt
        device = TorchDevice.choose_torch_device()

        assert self.qwen3_encoder is not None
        text_encoder_info = context.models.load(self.qwen3_encoder.text_encoder)
        tokenizer_info = context.models.load(self.qwen3_encoder.tokenizer)

        with ExitStack() as exit_stack:
            (_, text_encoder) = exit_stack.enter_context(text_encoder_info.model_on_device())
            (_, tokenizer) = exit_stack.enter_context(tokenizer_info.model_on_device())

            lora_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=text_encoder,
                    patches=self._qwen3_lora_iterator(context),
                    prefix=Z_IMAGE_LORA_QWEN3_PREFIX,
                    dtype=lora_dtype,
                )
            )

            context.util.signal_progress("Running Qwen3 text encoder")
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

            try:
                prompt_formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except (AttributeError, TypeError) as e:
                context.logger.warning(f"Chat template failed ({e}), using raw prompt.")
                prompt_formatted = prompt

            text_inputs = tokenizer(
                prompt_formatted,
                padding="max_length",
                max_length=QWEN3_MAX_SEQ_LEN,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            if not isinstance(text_input_ids, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for input_ids, got {type(text_input_ids).__name__}. "
                    "Tokenizer returned unexpected type."
                )
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(
                    f"Expected torch.Tensor for attention_mask, got {type(attention_mask).__name__}. "
                    "Tokenizer returned unexpected type."
                )

            prompt_mask = attention_mask.to(device).bool()
            outputs = text_encoder(
                text_input_ids.to(device),
                attention_mask=prompt_mask,
                output_hidden_states=True,
            )

            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise RuntimeError(
                    "Text encoder did not return hidden_states. "
                    "Ensure output_hidden_states=True is supported by this model."
                )
            if len(outputs.hidden_states) < 2:
                raise RuntimeError(
                    f"Expected at least 2 hidden states from text encoder, got {len(outputs.hidden_states)}. "
                    "This may indicate an incompatible model or configuration."
                )

            prompt_embeds = outputs.hidden_states[-2]
            prompt_embeds = prompt_embeds[0][prompt_mask[0]].unsqueeze(0)

        txt_seq_len = prompt_embeds.shape[1]
        txt_ids = torch.zeros((1, txt_seq_len, 3), device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        txt_ids[..., 0] = torch.arange(txt_seq_len, device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        return prompt_embeds, txt_ids

    def _clip_encode(self, context: InvocationContext) -> torch.Tensor:
        prompt = [self.prompt]

        clip_text_encoder_info = context.models.load(self.clip.text_encoder)
        clip_text_encoder_config = clip_text_encoder_info.config
        assert clip_text_encoder_config is not None

        with (
            clip_text_encoder_info.model_on_device() as (cached_weights, clip_text_encoder),
            context.models.load(self.clip.tokenizer) as clip_tokenizer,
            ExitStack() as exit_stack,
        ):
            assert isinstance(clip_text_encoder, CLIPTextModel)
            assert isinstance(clip_tokenizer, CLIPTokenizer)

            # Apply LoRA models to the CLIP encoder.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            if clip_text_encoder_config.format in [ModelFormat.Diffusers]:
                # The model is non-quantized, so we can apply the LoRA weights directly into the model.
                exit_stack.enter_context(
                    LayerPatcher.apply_smart_model_patches(
                        model=clip_text_encoder,
                        patches=self._clip_lora_iterator(context),
                        prefix=FLUX_LORA_CLIP_PREFIX,
                        dtype=clip_text_encoder.dtype,
                        cached_weights=cached_weights,
                    )
                )
            else:
                # There are currently no supported CLIP quantized models. Add support here if needed.
                raise ValueError(f"Unsupported model format: {clip_text_encoder_config.format}")

            clip_encoder = HFEncoder(clip_text_encoder, clip_tokenizer, True, 77)

            if context.config.get().log_tokenization:
                self._log_clip_tokenization(context, clip_tokenizer)

            context.util.signal_progress("Running CLIP encoder")
            pooled_prompt_embeds = clip_encoder(prompt)

        assert isinstance(pooled_prompt_embeds, torch.Tensor)
        return pooled_prompt_embeds

    def _clip_lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.clip.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _t5_lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.t5_encoder.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _qwen3_lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        if self.qwen3_encoder is None:
            return
        for lora in self.qwen3_encoder.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _log_t5_tokenization(
        self,
        context: InvocationContext,
        tokenizer: Union[T5Tokenizer, T5TokenizerFast],
    ) -> None:
        """Logs the tokenization of a prompt for a T5-based model like FLUX."""

        # Tokenize the prompt using the same parameters as the model's text encoder.
        # T5 tokenizers add an EOS token (</s>) and then pad to max_length.
        tokenized_output = tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.t5_max_seq_len,
            truncation=True,
            add_special_tokens=True,  # This is important for T5 to add the EOS token.
            return_tensors="pt",
        )

        input_ids = tokenized_output.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # The T5 tokenizer uses a space-like character ' ' (U+2581) to denote spaces.
        # We'll replace it with a regular space for readability.
        tokens = [t.replace("\u2581", " ") for t in tokens]

        tokenized_str = ""
        used_tokens = 0
        for token in tokens:
            if token == tokenizer.eos_token:
                tokenized_str += f"\x1b[0;31m{token}\x1b[0m"  # Red for EOS
                used_tokens += 1
            elif token == tokenizer.pad_token:
                # tokenized_str += f"\x1b[0;34m{token}\x1b[0m"  # Blue for PAD
                continue
            else:
                color = (used_tokens % 6) + 1  # Cycle through 6 colors
                tokenized_str += f"\x1b[0;3{color}m{token}\x1b[0m"
                used_tokens += 1

        context.logger.info(f">> [T5 TOKENLOG] Tokens ({used_tokens}/{self.t5_max_seq_len}):")
        context.logger.info(f"{tokenized_str}\x1b[0m")

    def _log_clip_tokenization(
        self,
        context: InvocationContext,
        tokenizer: CLIPTokenizer,
    ) -> None:
        """Logs the tokenization of a prompt for a CLIP-based model."""
        max_length = tokenizer.model_max_length

        tokenized_output = tokenizer(
            self.prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized_output.input_ids[0]
        attention_mask = tokenized_output.attention_mask[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # The CLIP tokenizer uses '</w>' to denote spaces.
        # We'll replace it with a regular space for readability.
        tokens = [t.replace("</w>", " ") for t in tokens]

        tokenized_str = ""
        used_tokens = 0
        for i, token in enumerate(tokens):
            if attention_mask[i] == 0:
                # Do not log padding tokens.
                continue

            if token == tokenizer.bos_token:
                tokenized_str += f"\x1b[0;32m{token}\x1b[0m"  # Green for BOS
            elif token == tokenizer.eos_token:
                tokenized_str += f"\x1b[0;31m{token}\x1b[0m"  # Red for EOS
            else:
                color = (used_tokens % 6) + 1  # Cycle through 6 colors
                tokenized_str += f"\x1b[0;3{color}m{token}\x1b[0m"
            used_tokens += 1

        context.logger.info(f">> [CLIP TOKENLOG] Tokens ({used_tokens}/{max_length}):")
        context.logger.info(f"{tokenized_str}\x1b[0m")
