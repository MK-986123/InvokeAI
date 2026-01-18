# Copyright (c) 2024, InvokeAI Development Team
"""Tests for FLUX.2 model detection and loading."""

import pytest
import torch

from invokeai.backend.flux2.model import Flux2, Flux2Params
from invokeai.backend.flux2.util import get_flux2_transformer_params
from invokeai.backend.model_manager.configs.main import (
    _get_flux2_variant,
    _has_flux2_keys,
)
from invokeai.backend.model_manager.taxonomy import Flux2VariantType


class TestFlux2Detection:
    """Tests for FLUX.2 model detection from state dict."""

    def test_has_flux2_keys_positive(self):
        """Test that FLUX.2-specific keys are correctly detected."""
        state_dict = {
            "double_stream_modulation_img.lin.weight": torch.zeros(1),
            "double_stream_modulation_txt.lin.weight": torch.zeros(1),
            "single_stream_modulation.lin.weight": torch.zeros(1),
            "img_in.weight": torch.zeros(3072, 128),  # hidden_size=3072, in_channels=128
        }
        assert _has_flux2_keys(state_dict) is True

    def test_has_flux2_keys_negative_flux1(self):
        """Test that FLUX.1 keys are not detected as FLUX.2."""
        # FLUX.1 has per-block modulation, not shared modulation layers
        state_dict = {
            "double_blocks.0.img_mod.lin.weight": torch.zeros(1),
            "double_blocks.0.txt_mod.lin.weight": torch.zeros(1),
            "single_blocks.0.modulation.lin.weight": torch.zeros(1),
            "img_in.weight": torch.zeros(3072, 64),  # in_channels=64 for FLUX.1
        }
        assert _has_flux2_keys(state_dict) is False

    def test_get_flux2_variant_klein_4b(self):
        """Test detection of FLUX.2-klein-4B variant."""
        # Build a minimal state dict that matches klein-4B architecture
        # hidden_size=3072, 5 double blocks, 20 single blocks
        state_dict = {
            "img_in.weight": torch.zeros(3072, 128),
            "double_stream_modulation_img.lin.weight": torch.zeros(1),
        }
        # Add keys for 5 double blocks and 20 single blocks
        for i in range(5):
            state_dict[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.zeros(1)
        for i in range(20):
            state_dict[f"single_blocks.{i}.linear1.weight"] = torch.zeros(1)

        variant = _get_flux2_variant(state_dict)
        assert variant == Flux2VariantType.Klein4B

    def test_get_flux2_variant_klein_9b(self):
        """Test detection of FLUX.2-klein-9B variant."""
        # hidden_size=4096, 8 double blocks, 24 single blocks
        state_dict = {
            "img_in.weight": torch.zeros(4096, 128),
            "double_stream_modulation_img.lin.weight": torch.zeros(1),
        }
        for i in range(8):
            state_dict[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.zeros(1)
        for i in range(24):
            state_dict[f"single_blocks.{i}.linear1.weight"] = torch.zeros(1)

        variant = _get_flux2_variant(state_dict)
        assert variant == Flux2VariantType.Klein9B

    def test_get_flux2_variant_klein_9b_fp8(self):
        """Test detection of FLUX.2-klein-9B-FP8 variant."""
        state_dict = {
            "img_in.weight": torch.zeros(4096, 128),
            "double_stream_modulation_img.lin.weight": torch.zeros(1),
            # FP8 quantization markers
            "double_blocks.0.img_attn.qkv.input_scale": torch.zeros(1),
        }
        for i in range(8):
            state_dict[f"double_blocks.{i}.img_attn.qkv.weight"] = torch.zeros(1)
        for i in range(24):
            state_dict[f"single_blocks.{i}.linear1.weight"] = torch.zeros(1)

        variant = _get_flux2_variant(state_dict)
        assert variant == Flux2VariantType.Klein9BFP8

    def test_get_flux2_variant_not_flux2(self):
        """Test that non-FLUX.2 models return None."""
        # FLUX.1-style state dict (in_channels=64)
        state_dict = {
            "img_in.weight": torch.zeros(3072, 64),
        }
        variant = _get_flux2_variant(state_dict)
        assert variant is None


class TestFlux2Params:
    """Tests for FLUX.2 parameter configurations."""

    def test_klein_4b_params(self):
        """Test FLUX.2-klein-4B parameter preset."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein4B)

        assert params.in_channels == 128
        assert params.hidden_size == 3072
        assert params.num_attention_heads == 24
        assert params.attention_head_dim == 128
        assert params.num_layers == 5
        assert params.num_single_layers == 20
        assert params.mlp_ratio == 3.0
        assert params.joint_attention_dim == 7680
        assert params.guidance_embeds is False
        assert params.rope_theta == 2000
        assert params.axes_dims_rope == [32, 32, 32, 32]

    def test_klein_9b_params(self):
        """Test FLUX.2-klein-9B parameter preset."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein9B)

        assert params.in_channels == 128
        assert params.hidden_size == 4096
        assert params.num_attention_heads == 32
        assert params.attention_head_dim == 128
        assert params.num_layers == 8
        assert params.num_single_layers == 24
        assert params.joint_attention_dim == 12288

    def test_unknown_variant_raises(self):
        """Test that unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FLUX.2 variant"):
            get_flux2_transformer_params("invalid_variant")


class TestFlux2Model:
    """Tests for FLUX.2 transformer model instantiation."""

    def test_flux2_model_instantiation_klein_4b(self):
        """Test that Flux2 model can be instantiated with klein-4B params."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein4B)

        # Should not raise
        model = Flux2(params)

        # Verify architecture
        assert model.in_channels == 128
        assert model.out_channels == 128
        assert model.hidden_size == 3072
        assert len(model.double_blocks) == 5
        assert len(model.single_blocks) == 20
        assert model.guidance_in is None  # No guidance for distilled model

    def test_flux2_model_instantiation_klein_9b(self):
        """Test that Flux2 model can be instantiated with klein-9B params."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein9B)

        model = Flux2(params)

        assert model.hidden_size == 4096
        assert len(model.double_blocks) == 8
        assert len(model.single_blocks) == 24

    def test_flux2_model_state_dict_keys(self):
        """Test that Flux2 model produces expected state dict keys."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein4B)
        model = Flux2(params)

        state_dict = model.state_dict()

        # Check for FLUX.2-specific keys
        assert "double_stream_modulation_img.lin.weight" in state_dict
        assert "double_stream_modulation_txt.lin.weight" in state_dict
        assert "single_stream_modulation.lin.weight" in state_dict
        assert "img_in.weight" in state_dict
        assert "txt_in.weight" in state_dict
        assert "time_in.in_layer.weight" in state_dict
        assert "final_layer.linear.weight" in state_dict

        # Check block keys
        assert "double_blocks.0.img_attn.qkv.weight" in state_dict
        assert "double_blocks.4.txt_mlp.2.weight" in state_dict  # Last double block
        assert "single_blocks.0.linear1.weight" in state_dict
        assert "single_blocks.19.linear2.weight" in state_dict  # Last single block

    def test_flux2_model_tensor_shapes(self):
        """Test that Flux2 model tensors have expected shapes."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein4B)
        model = Flux2(params)

        state_dict = model.state_dict()

        # img_in: [hidden_size, in_channels]
        assert state_dict["img_in.weight"].shape == (3072, 128)

        # txt_in: [hidden_size, joint_attention_dim]
        assert state_dict["txt_in.weight"].shape == (3072, 7680)

        # double_stream_modulation: [6 * hidden_size, hidden_size]
        assert state_dict["double_stream_modulation_img.lin.weight"].shape == (6 * 3072, 3072)

        # single_stream_modulation: [3 * hidden_size, hidden_size]
        assert state_dict["single_stream_modulation.lin.weight"].shape == (3 * 3072, 3072)


class TestFlux2VsFlux1Distinction:
    """Tests to verify FLUX.2 is correctly distinguished from FLUX.1."""

    def test_flux2_in_channels_differs_from_flux1(self):
        """Verify FLUX.2 uses in_channels=128 vs FLUX.1's 64."""
        params_4b = get_flux2_transformer_params(Flux2VariantType.Klein4B)
        assert params_4b.in_channels == 128

        # FLUX.1 uses 64 in_channels (verify we're different)
        from invokeai.backend.flux.util import get_flux_transformers_params
        from invokeai.backend.model_manager.taxonomy import FluxVariantType

        flux1_params = get_flux_transformers_params(FluxVariantType.Dev)
        assert flux1_params.in_channels == 64

        # These should be different
        assert params_4b.in_channels != flux1_params.in_channels

    def test_flux2_block_structure_differs_from_flux1(self):
        """Verify FLUX.2 has different block counts than FLUX.1."""
        params_4b = get_flux2_transformer_params(Flux2VariantType.Klein4B)

        from invokeai.backend.flux.util import get_flux_transformers_params
        from invokeai.backend.model_manager.taxonomy import FluxVariantType

        flux1_params = get_flux_transformers_params(FluxVariantType.Dev)

        # FLUX.2-klein-4B: 5 double, 20 single
        # FLUX.1-dev: 19 double, 38 single
        assert params_4b.num_layers != flux1_params.depth
        assert params_4b.num_single_layers != flux1_params.depth_single_blocks

    def test_flux2_modulation_structure_is_shared(self):
        """Verify FLUX.2 uses shared modulation (not per-block like FLUX.1)."""
        params = get_flux2_transformer_params(Flux2VariantType.Klein4B)
        model = Flux2(params)

        # FLUX.2 has shared modulation layers at the top level
        assert hasattr(model, "double_stream_modulation_img")
        assert hasattr(model, "double_stream_modulation_txt")
        assert hasattr(model, "single_stream_modulation")

        # These should be single modules, not per-block
        assert not hasattr(model.double_blocks[0], "img_mod")
        assert not hasattr(model.single_blocks[0], "modulation")
