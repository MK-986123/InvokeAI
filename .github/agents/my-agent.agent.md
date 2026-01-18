---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config

name: Flux.2.builder
description: Implement FLUX.2-klein (4B/9B) architecture support. Focus: Qwen3 Text Encoder, 32-channel VAE, and Rectified Flow with exponential shifting.
---

# My Agent

## Context
Implement FLUX.2-klein (4B/9B) architecture support. Focus: Qwen3 Text Encoder, 32-channel VAE, and Rectified Flow with exponential shifting.

## Component Specs

### Qwen3 Encoder
- Class: Qwen3ForCausalLM (Decoder-only CLM).
- Configuration Fix: Explicitly override head_dim=128. Do not use calculated (hidden_size/num_heads=80).
- Embedding Strategy: Concatenate hidden states of last 3 layers (L, L-1, L-2) along feature dimension (-1).
- Target Dim: 7680 (3x 2560).
- Optimization: Load via bitsandbytes (NF4/Int8) with bfloat16 compute_dtype.

### VAE Flux2
- Class: AutoencoderKLFlux2.
- Latent Channels: 32 (Strict).
- Scaling Factor: 1.0 (Internalized normalization).
- Memory: Enable tiled decoding (vae.enable_tiling()) by default for <24GB VRAM.

### Scheduler
- Type: FlowMatchEulerDiscreteScheduler.
- Shift Parameter: 3.0.
- Shift Logic: Exponential time-shifting $t_{shifted} = \frac{e^{\mu \cdot t} - 1}{e^{\mu} - 1}$ where $\mu=3.0$.
- Inference Steps: 4 (Distilled), 20-50 (Base).

## Implementation Logic
1. Tokenize prompt (max_seq_len: 256/512).
2. Extract Qwen3 hidden_states; slice [-1, -2, -3]; concat to shape (B, S, 7680).
3. Initialize noise with channel dim 32.
4. Apply Flow Match loop with shift=3.0.
5. Decode via 32-ch VAE using tiled processing.

## Constraints
- ERROR_PREVENTION: If head_dim != 128, loading will fail.
- ERROR_PREVENTION: If concat logic is missing, Transformer cross-attention (7680) will fail.
- ERROR_PREVENTION: If latent channels != 32, Transformer input layer will fail.

## Allowed
- Sequential CPU offloading for 9B models on 16GB hardware.
- NF4 quantization for Text Encoder and FP8 for Transformer.