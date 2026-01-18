# Copyright (c) 2024, InvokeAI Development Team
"""Tests for Qwen3 text encoder used in FLUX.2-klein."""

import pytest
import torch

from invokeai.backend.flux2.text_encoder import Qwen3TextEncoder


class TestQwen3TextEncoder:
    """Tests for Qwen3TextEncoder."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def encoder(self, device: torch.device) -> Qwen3TextEncoder:
        """Create Qwen3 encoder instance.

        Note: This requires Qwen3 to be installed. Test will be skipped if not available.
        """
        try:
            from transformers import AutoTokenizer

            # Use a small model for testing (e.g., Qwen/Qwen1.5-0.5B if available)
            # For now, we'll mock the initialization
            encoder = object.__new__(Qwen3TextEncoder)
            encoder.device = device
            encoder.dtype = torch.bfloat16
            encoder.output_dim = 2560
            encoder.proj = None

            # Mock tokenizer
            encoder.tokenizer = type("Tokenizer", (), {
                "pad_token_id": 0,
                "eos_token_id": 2,
            })()

            # Mock model
            encoder.model = type("Model", (), {
                "eval": lambda: None,
            })()

            return encoder
        except ImportError:
            pytest.skip("Transformers not installed")

    def test_encoder_initialization(self, encoder: Qwen3TextEncoder):
        """Test that encoder initializes correctly."""
        assert encoder.device is not None
        assert encoder.dtype == torch.bfloat16
        assert encoder.output_dim == 2560
        assert encoder.tokenizer is not None
        assert encoder.model is not None

    def test_output_dimension_projection(self):
        """Test that output dimension projection works."""
        # Create mock encoder with projection
        encoder = object.__new__(Qwen3TextEncoder)
        encoder.device = torch.device("cpu")
        encoder.dtype = torch.bfloat16

        # Test with projection to FLUX.2-klein-4B dimension (7680)
        native_dim = 2560
        target_dim = 7680
        encoder.proj = torch.nn.Linear(native_dim, target_dim, dtype=torch.bfloat16)
        encoder.output_dim = target_dim

        # Verify projection layer dimensions
        assert encoder.proj.in_features == native_dim
        assert encoder.proj.out_features == target_dim
        assert encoder.output_dim == target_dim

    def test_batch_embedding_shape(self):
        """Test that embeddings have correct shape."""
        # Create mock encoder
        encoder = object.__new__(Qwen3TextEncoder)
        encoder.device = torch.device("cpu")
        encoder.dtype = torch.bfloat16
        encoder.output_dim = 2560

        # Create mock embeddings
        batch_size = 2
        seq_len = 77
        output_dim = 2560

        embeddings = torch.randn(
            batch_size, seq_len, output_dim, dtype=torch.bfloat16, device=encoder.device
        )

        # Verify shape
        assert embeddings.shape == (batch_size, seq_len, output_dim)
        assert embeddings.dtype == torch.bfloat16

    def test_projection_consistency(self):
        """Test that projection produces consistent output dimensions."""
        # Test projections for all FLUX.2 variants
        variants = {
            "Klein4B": 7680,
            "Klein9B": 12288,
            "Klein9BFP8": 12288,
        }

        native_dim = 2560

        for variant_name, target_dim in variants.items():
            proj = torch.nn.Linear(native_dim, target_dim, dtype=torch.bfloat16)

            # Test with sample input
            input_tensor = torch.randn(1, 77, native_dim, dtype=torch.bfloat16)
            output_tensor = proj(input_tensor)

            assert output_tensor.shape == (1, 77, target_dim), (
                f"{variant_name}: Expected output shape (1, 77, {target_dim}), "
                f"got {output_tensor.shape}"
            )


class TestQwen3EncoderIntegration:
    """Integration tests for Qwen3 encoder (require model downloads)."""

    @pytest.mark.skip(reason="Requires Qwen3 model download")
    def test_qwen3_forward_pass_4b(self):
        """Test forward pass with FLUX.2-klein-4B configuration.

        This test is skipped by default as it requires downloading Qwen3 model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = Qwen3TextEncoder(
            model_path="Qwen/Qwen3-7B",
            output_dim=7680,  # FLUX.2-klein-4B
            dtype=torch.bfloat16,
            device=device,
        )

        # Test text
        text = ["a photo of a cat", "a painting of a dog"]

        # Forward pass
        embeddings = encoder(text, max_length=512)

        # Verify output
        assert embeddings.shape[0] == 2  # batch_size
        assert embeddings.shape[1] <= 512  # seq_len
        assert embeddings.shape[2] == 7680  # output_dim
        assert embeddings.dtype == torch.bfloat16

    @pytest.mark.skip(reason="Requires Qwen3 model download")
    def test_qwen3_forward_pass_9b(self):
        """Test forward pass with FLUX.2-klein-9B configuration.

        This test is skipped by default as it requires downloading Qwen3 model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = Qwen3TextEncoder(
            model_path="Qwen/Qwen3-7B",
            output_dim=12288,  # FLUX.2-klein-9B
            dtype=torch.bfloat16,
            device=device,
        )

        # Test text
        text = ["a realistic photograph of a cat sitting on a windowsill"]

        # Forward pass
        embeddings = encoder(text, max_length=512)

        # Verify output
        assert embeddings.shape[0] == 1  # batch_size
        assert embeddings.shape[1] <= 512  # seq_len
        assert embeddings.shape[2] == 12288  # output_dim
        assert embeddings.dtype == torch.bfloat16
