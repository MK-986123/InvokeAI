# Copyright (c) 2024, InvokeAI Development Team
"""FLUX.2 model support for InvokeAI.

FLUX.2-klein is a distinct architecture from FLUX.1, featuring:
- Different transformer structure (Flux2Transformer2DModel)
- Qwen3 text encoder instead of CLIP+T5
- Different VAE with 32 latent channels
- Different RoPE configuration
"""
