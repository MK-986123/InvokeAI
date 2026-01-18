# Copyright (c) 2024, InvokeAI Development Team
"""Qwen3 text encoder for FLUX.2-klein models.

Qwen3 is a causal language model used as the text encoder for FLUX.2-klein,
replacing the CLIP+T5 combination used in FLUX.1.

Output embeddings have shape [batch, seq_len, hidden_size] where:
- hidden_size = 2560 (Qwen3-7B)
- mapped to joint_attention_dim for FLUX.2-klein (7680 for 4B, 12288 for 9B)
"""

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class Qwen3TextEncoder(nn.Module):
    """Wrapper for Qwen3 text encoder with projection to FLUX.2 embedding dimension.

    Attributes:
        model: Qwen3ForCausalLM transformer model
        tokenizer: Qwen tokenizer
        proj: Optional projection layer to joint_attention_dim
        device: Device the model is on
        dtype: Data type (torch.bfloat16 for FLUX.2)
    """

    def __init__(
        self,
        model_path: str,
        output_dim: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize Qwen3 text encoder.

        Args:
            model_path: Path to Qwen3 model (e.g., "Qwen/Qwen3-7B")
            output_dim: Target embedding dimension (joint_attention_dim for FLUX.2).
                        If None, uses native Qwen3 hidden_size (2560).
            dtype: Data type for model inference
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Get native hidden size
        native_hidden_size = config.hidden_size  # 2560 for Qwen3-7B

        # Optional projection to output_dim
        if output_dim is not None and output_dim != native_hidden_size:
            self.proj = nn.Linear(native_hidden_size, output_dim, dtype=dtype, device=device)
            self.output_dim = output_dim
        else:
            self.proj = None
            self.output_dim = native_hidden_size

    def forward(
        self,
        text: list[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            text: List of text strings to encode
            max_length: Maximum token sequence length

        Returns:
            Tensor of shape [batch_size, seq_len, output_dim]
                with dtype=torch.bfloat16
        """
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )

        # Use last hidden state
        # Shape: [batch_size, seq_len, hidden_size]
        embeddings = outputs.last_hidden_state

        # Project to output dimension if needed
        if self.proj is not None:
            embeddings = self.proj(embeddings)

        return embeddings.to(dtype=self.dtype)

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        if self.proj is not None:
            self.proj = self.proj.to(*args, **kwargs)
        return self
