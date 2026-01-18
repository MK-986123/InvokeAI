# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 transformer model implementation.

FLUX.2-klein uses a different architecture from FLUX.1:
- Gated SiLU (SwiGLU) activation in MLPs
- Different modulation structure (shared modulation layers)
- Fused QKV+MLP projections in single stream blocks
- Different RoPE configuration
"""

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn

from invokeai.backend.flux.math import attention, rope


@dataclass
class Flux2Params:
    """Parameters for FLUX.2 transformer model."""

    in_channels: int
    """Input latent channels (128 for FLUX.2)."""
    hidden_size: int
    """Hidden dimension of the transformer."""
    num_attention_heads: int
    """Number of attention heads."""
    attention_head_dim: int
    """Dimension per attention head."""
    num_layers: int
    """Number of double stream (joint attention) blocks."""
    num_single_layers: int
    """Number of single stream blocks."""
    mlp_ratio: float
    """MLP expansion ratio."""
    joint_attention_dim: int
    """Text encoder output dimension."""
    axes_dims_rope: list[int]
    """RoPE axis dimensions."""
    rope_theta: int
    """RoPE theta parameter."""
    timestep_guidance_channels: int = 256
    """Timestep embedding dimension."""
    guidance_embeds: bool = False
    """Whether to use guidance embeddings (False for distilled klein models)."""
    out_channels: Optional[int] = None
    """Output channels (defaults to in_channels)."""


class Flux2RMSNorm(nn.Module):
    """RMS normalization for FLUX.2."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.rms_norm(x, self.scale.shape, self.scale, eps=self.eps)


class Flux2QKNorm(nn.Module):
    """Query-Key normalization for FLUX.2."""

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = Flux2RMSNorm(dim)
        self.key_norm = Flux2RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Flux2EmbedND(nn.Module):
    """N-dimensional positional embedding using RoPE for FLUX.2."""

    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def flux2_timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Create sinusoidal timestep embeddings for FLUX.2."""
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t.dtype)


class Flux2MLPEmbedder(nn.Module):
    """MLP embedder for timestep/guidance embeddings in FLUX.2."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation for FLUX.2 MLPs."""

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.silu(gate)


class Flux2Modulation(nn.Module):
    """Shared modulation layer for FLUX.2.

    Unlike FLUX.1 which has per-block modulation, FLUX.2 uses shared
    modulation layers for all double/single stream blocks.
    """

    def __init__(self, hidden_size: int, num_params: int):
        super().__init__()
        self.lin = nn.Linear(hidden_size, num_params * hidden_size, bias=True)

    def forward(self, vec: Tensor) -> Tensor:
        return self.lin(nn.functional.silu(vec))


class Flux2DoubleStreamBlock(nn.Module):
    """Double stream (joint attention) block for FLUX.2.

    Key differences from FLUX.1:
    - Uses SwiGLU activation instead of GELU
    - Modulation is applied externally from shared modulation layers
    - Different tensor key naming convention
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        # Image attention
        self.img_attn = nn.ModuleDict({
            "qkv": nn.Linear(hidden_size, hidden_size * 3, bias=True),
            "proj": nn.Linear(hidden_size, hidden_size, bias=True),
            "norm": Flux2QKNorm(head_dim),
        })

        # Text attention
        self.txt_attn = nn.ModuleDict({
            "qkv": nn.Linear(hidden_size, hidden_size * 3, bias=True),
            "proj": nn.Linear(hidden_size, hidden_size, bias=True),
            "norm": Flux2QKNorm(head_dim),
        })

        # MLPs with SwiGLU
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden * 2, bias=True),  # *2 for gated
            Flux2SwiGLU(),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden * 2, bias=True),
            Flux2SwiGLU(),
            nn.Linear(mlp_hidden, hidden_size, bias=True),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        img_mod: Tensor,
        txt_mod: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Unpack modulation parameters (shift, scale, gate for norm1, norm2)
        img_mod = img_mod.chunk(6, dim=-1)
        txt_mod = txt_mod.chunk(6, dim=-1)

        # Image attention
        img_norm = nn.functional.layer_norm(img, (self.hidden_size,))
        img_modulated = (1 + img_mod[1]) * img_norm + img_mod[0]
        img_qkv = self.img_attn["qkv"](img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn["norm"](img_q, img_k, img_v)

        # Text attention
        txt_norm = nn.functional.layer_norm(txt, (self.hidden_size,))
        txt_modulated = (1 + txt_mod[1]) * txt_norm + txt_mod[0]
        txt_qkv = self.txt_attn["qkv"](txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn["norm"](txt_q, txt_k, txt_v)

        # Joint attention
        q = torch.cat([txt_q, img_q], dim=2)
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)
        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        # Apply attention and MLP with gating
        img = img + img_mod[2] * self.img_attn["proj"](img_attn)
        img_norm2 = nn.functional.layer_norm(img, (self.hidden_size,))
        img = img + img_mod[5] * self.img_mlp((1 + img_mod[4]) * img_norm2 + img_mod[3])

        txt = txt + txt_mod[2] * self.txt_attn["proj"](txt_attn)
        txt_norm2 = nn.functional.layer_norm(txt, (self.hidden_size,))
        txt = txt + txt_mod[5] * self.txt_mlp((1 + txt_mod[4]) * txt_norm2 + txt_mod[3])

        return img, txt


class Flux2SingleStreamBlock(nn.Module):
    """Single stream block for FLUX.2 with fused projections.

    Key differences from FLUX.1:
    - Fused QKV + MLP input projection (linear1)
    - Fused attention + MLP output projection (linear2)
    - SwiGLU activation
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)

        # Fused input: QKV (3 * hidden) + MLP gate+value (2 * mlp_hidden)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden * 2, bias=True)

        # Fused output: attention (hidden) + MLP (mlp_hidden) -> hidden
        self.linear2 = nn.Linear(hidden_size + mlp_hidden, hidden_size, bias=True)

        self.norm = Flux2QKNorm(head_dim)
        self.mlp_hidden = mlp_hidden

    def forward(self, x: Tensor, pe: Tensor, mod: Tensor) -> Tensor:
        # Unpack modulation (shift, scale, gate)
        mod = mod.chunk(3, dim=-1)

        # Apply pre-norm with modulation
        x_norm = nn.functional.layer_norm(x, (self.hidden_size,))
        x_mod = (1 + mod[1]) * x_norm + mod[0]

        # Fused projection
        qkv_mlp = self.linear1(x_mod)
        qkv, mlp = qkv_mlp.split([self.hidden_size * 3, self.mlp_hidden * 2], dim=-1)

        # Attention
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)

        # MLP with SwiGLU
        mlp_out = mlp.chunk(2, dim=-1)
        mlp_out = mlp_out[0] * nn.functional.silu(mlp_out[1])

        # Fused output projection with gating
        out = self.linear2(torch.cat([attn, mlp_out], dim=-1))
        return x + mod[2] * out


class Flux2FinalLayer(nn.Module):
    """Final output layer for FLUX.2."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)


class Flux2(nn.Module):
    """FLUX.2 transformer model for flow matching on sequences.

    This is a distinct architecture from FLUX.1, featuring:
    - Shared modulation layers (double_stream_modulation_img/txt, single_stream_modulation)
    - SwiGLU activation in MLPs
    - Different RoPE configuration
    - Qwen3 text encoder compatibility (joint_attention_dim matches Qwen3 output)
    """

    def __init__(self, params: Flux2Params):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels or params.in_channels
        self.hidden_size = params.hidden_size

        # Validate dimensions
        if params.hidden_size % params.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_attention_heads}"
            )

        # Positional embeddings
        self.pe_embedder = Flux2EmbedND(
            dim=params.attention_head_dim,
            theta=params.rope_theta,
            axes_dim=params.axes_dims_rope
        )

        # Input projections
        self.img_in = nn.Linear(params.in_channels, params.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.joint_attention_dim, params.hidden_size, bias=True)

        # Timestep embedding
        self.time_in = Flux2MLPEmbedder(params.timestep_guidance_channels, params.hidden_size)

        # Guidance embedding (only if enabled)
        if params.guidance_embeds:
            self.guidance_in = Flux2MLPEmbedder(params.timestep_guidance_channels, params.hidden_size)
        else:
            self.guidance_in = None

        # Shared modulation layers (key difference from FLUX.1)
        # Double stream: 6 params each for img and txt (shift, scale, gate for norm1 and norm2)
        self.double_stream_modulation_img = Flux2Modulation(params.hidden_size, 6)
        self.double_stream_modulation_txt = Flux2Modulation(params.hidden_size, 6)
        # Single stream: 3 params (shift, scale, gate)
        self.single_stream_modulation = Flux2Modulation(params.hidden_size, 3)

        # Transformer blocks
        self.double_blocks = nn.ModuleList([
            Flux2DoubleStreamBlock(params.hidden_size, params.num_attention_heads, params.mlp_ratio)
            for _ in range(params.num_layers)
        ])

        self.single_blocks = nn.ModuleList([
            Flux2SingleStreamBlock(params.hidden_size, params.num_attention_heads, params.mlp_ratio)
            for _ in range(params.num_single_layers)
        ])

        # Output layer
        self.final_layer = Flux2FinalLayer(params.hidden_size, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of FLUX.2 transformer.

        Args:
            img: Image latents [B, L_img, in_channels]
            img_ids: Image position IDs [B, L_img, n_axes]
            txt: Text embeddings from Qwen3 [B, L_txt, joint_attention_dim]
            txt_ids: Text position IDs [B, L_txt, n_axes]
            timesteps: Timestep values [B]
            guidance: Optional guidance values [B] (only if guidance_embeds=True)

        Returns:
            Denoised output [B, L_img, out_channels]
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # Project inputs
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # Timestep embedding
        vec = self.time_in(flux2_timestep_embedding(timesteps, self.params.timestep_guidance_channels))

        # Add guidance embedding if enabled
        if self.params.guidance_embeds:
            if guidance is None:
                raise ValueError("Model requires guidance but none was provided.")
            vec = vec + self.guidance_in(flux2_timestep_embedding(guidance, self.params.timestep_guidance_channels))

        # Compute positional embeddings
        ids = torch.cat([txt_ids, img_ids], dim=1)
        pe = self.pe_embedder(ids)

        # Compute shared modulation
        img_mod = self.double_stream_modulation_img(vec)[:, None, :]
        txt_mod = self.double_stream_modulation_txt(vec)[:, None, :]
        single_mod = self.single_stream_modulation(vec)[:, None, :]

        # Double stream blocks
        for block in self.double_blocks:
            img, txt = block(img, txt, pe, img_mod, txt_mod)

        # Concatenate for single stream
        x = torch.cat([txt, img], dim=1)

        # Single stream blocks
        for block in self.single_blocks:
            x = block(x, pe, single_mod)

        # Extract image portion
        img = x[:, txt.shape[1]:, :]

        # Final projection
        return self.final_layer(img, vec)
