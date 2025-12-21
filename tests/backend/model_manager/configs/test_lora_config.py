"""Tests for LoRA config detection utilities."""

import pytest

from invokeai.backend.model_manager.configs.lora import _looks_like_flux_lora


class TestLooksLikeFluxLoRA:
    """Tests for the _looks_like_flux_lora fallback detector."""

    def test_onetrainer_format_detected(self):
        """OneTrainer FLUX LoRA format should be detected."""
        state_dict = {
            "lora_transformer_transformer_blocks_0_attn_to_k.lora_down.weight": None,
            "lora_transformer_transformer_blocks_0_attn_to_k.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_diffusers_single_transformer_format_detected(self):
        """Diffusers FLUX LoRA format with single_transformer_blocks should be detected."""
        state_dict = {
            "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight": None,
            "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_diffusers_transformer_blocks_format_detected(self):
        """Diffusers FLUX LoRA format with transformer_blocks should be detected."""
        state_dict = {
            "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight": None,
            "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_peft_format_detected(self):
        """PEFT FLUX LoRA format should be detected."""
        state_dict = {
            "base_model.model.single_transformer_blocks.0.attn.to_q.lora_A.weight": None,
            "base_model.model.single_transformer_blocks.0.attn.to_q.lora_B.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_kohya_double_blocks_format_detected(self):
        """Kohya FLUX LoRA format with double_blocks should be detected."""
        state_dict = {
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": None,
            "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_kohya_single_blocks_format_detected(self):
        """Kohya FLUX LoRA format with single_blocks should be detected."""
        state_dict = {
            "lora_unet_single_blocks_0_linear1.lora_down.weight": None,
            "lora_unet_single_blocks_0_linear1.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_aitoolkit_format_detected(self):
        """AI-Toolkit FLUX LoRA format should be detected."""
        state_dict = {
            "diffusion_model.some_layer.lora_A.weight": None,
            "diffusion_model.some_layer.lora_B.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is True

    def test_sdxl_text_encoder_not_detected(self):
        """SDXL LoRA with only text encoder keys should NOT be detected as FLUX."""
        state_dict = {
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha": None,
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": None,
            "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": None,
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": None,
            "lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is False

    def test_sd1_lora_not_detected(self):
        """SD1 LoRA should NOT be detected as FLUX."""
        state_dict = {
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": None,
            "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is False

    def test_sdxl_unet_not_detected(self):
        """SDXL LoRA with unet keys (not FLUX-specific) should NOT be detected as FLUX."""
        state_dict = {
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": None,
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": None,
        }
        assert _looks_like_flux_lora(state_dict) is False

    def test_empty_state_dict_not_detected(self):
        """Empty state dict should NOT be detected as FLUX."""
        assert _looks_like_flux_lora({}) is False

    def test_keys_without_lora_suffix_not_detected(self):
        """FLUX-like keys without LoRA suffix should NOT be detected."""
        state_dict = {
            "lora_transformer_transformer_blocks_0_attn_to_k.weight": None,
            "lora_transformer_transformer_blocks_0_attn_to_k.bias": None,
        }
        assert _looks_like_flux_lora(state_dict) is False
