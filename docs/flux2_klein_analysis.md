# FLUX.2-klein Model Analysis and InvokeAI Integration

## Executive Summary

FLUX.2-klein is a **completely new model architecture** from Black Forest Labs, distinct from FLUX.1. This document provides a comprehensive analysis of the architecture differences and the implementation approach for InvokeAI 6.10.0 support.

## Architecture Comparison

### FLUX.1 vs FLUX.2-klein-4B

| Component | FLUX.1 Dev/Schnell | FLUX.2-klein-4B |
|-----------|-------------------|-----------------|
| Pipeline class | FluxPipeline | Flux2KleinPipeline |
| Transformer class | Flux | Flux2Transformer2DModel |
| in_channels | 64 | 128 |
| hidden_size | 3072 | 3072 |
| num_attention_heads | 24 | 24 |
| num_double_blocks | 19 | 5 |
| num_single_blocks | 38 | 20 |
| mlp_ratio | 4.0 | 3.0 |
| context_in_dim | 4096 (T5) | 7680 (Qwen3) |
| guidance_embed | True (Dev) / False (Schnell) | False |
| RoPE axes_dim | [16, 56, 56] | [32, 32, 32, 32] |
| rope_theta | 10,000 | 2,000 |

### FLUX.2-klein Variants

| Variant | Parameters | Double Blocks | Single Blocks | Hidden Size |
|---------|------------|---------------|---------------|-------------|
| klein-4B | ~3.9B | 5 | 20 | 3072 |
| klein-9B | ~9.1B | 8 | 24 | 4096 |
| klein-9B-FP8 | ~9.1B | 8 | 24 | 4096 |

## Key Architectural Differences

### 1. Transformer Structure

FLUX.2 uses **shared modulation layers** at the model level, unlike FLUX.1 which has per-block modulation:

```
FLUX.1 structure:
- double_blocks.0.img_mod.lin.weight
- double_blocks.0.txt_mod.lin.weight
- single_blocks.0.modulation.lin.weight

FLUX.2 structure:
- double_stream_modulation_img.lin.weight  (shared)
- double_stream_modulation_txt.lin.weight  (shared)
- single_stream_modulation.lin.weight      (shared)
```

### 2. Text Encoder

| FLUX.1 | FLUX.2-klein |
|--------|--------------|
| CLIP-L/14 + T5-XXL | Qwen3ForCausalLM |
| 2 encoders, 2 tokenizers | 1 unified encoder |
| context_in_dim=4096 | joint_attention_dim=7680 (4B) / 12288 (9B) |

### 3. VAE

| FLUX.1 | FLUX.2-klein |
|--------|--------------|
| AutoEncoder | AutoencoderKLFlux2 |
| latent_channels=16 | latent_channels=32 |

### 4. MLP Activation

- FLUX.1: GELU (approximate="tanh")
- FLUX.2: SwiGLU (gated SiLU)

## Tensor Inventory Summary

### FLUX.2-klein-4B

- **Total parameters**: ~3.88B
- **Total tensors**: 238
- **File size (bf16)**: ~7.75 GB

| Component | Tensors | Parameters | % of Total |
|-----------|---------|------------|------------|
| single_blocks | 120 | 2,454M | 63.3% |
| double_blocks | 100 | 1,227M | 31.7% |
| modulation layers | 6 | 142M | 3.7% |
| input/output layers | 12 | 53M | 1.3% |

### Key Tensor Shapes (4B)

| Key | Shape | Description |
|-----|-------|-------------|
| img_in.weight | [3072, 128] | Latent input projection |
| txt_in.weight | [3072, 7680] | Text embedding projection |
| double_stream_modulation_img.lin.weight | [18432, 3072] | 6 modulation params |
| single_stream_modulation.lin.weight | [9216, 3072] | 3 modulation params |
| double_blocks.0.img_attn.qkv.weight | [9216, 3072] | QKV projection |
| single_blocks.0.linear1.weight | [27648, 3072] | Fused QKV+MLP input |

## Implementation Details

### Files Modified

1. **invokeai/backend/model_manager/taxonomy.py**
   - Added `BaseModelType.Flux2`
   - Added `Flux2VariantType` enum (Klein4B, Klein9B, Klein9BFP8)
   - Updated `AnyVariant` type alias

2. **invokeai/backend/model_manager/configs/main.py**
   - Added `_has_flux2_keys()` detection function
   - Added `_get_flux2_variant()` variant detection
   - Added `Main_Checkpoint_FLUX2_Config` class

3. **invokeai/backend/model_manager/configs/factory.py**
   - Imported and registered `Main_Checkpoint_FLUX2_Config`

### Files Created

1. **invokeai/backend/flux2/__init__.py**
   - Module initialization

2. **invokeai/backend/flux2/model.py**
   - `Flux2Params` dataclass
   - `Flux2` transformer model class
   - Supporting modules (Flux2DoubleStreamBlock, Flux2SingleStreamBlock, etc.)

3. **invokeai/backend/flux2/util.py**
   - Parameter presets for all variants
   - `get_flux2_transformer_params()` function
   - `get_flux2_max_seq_length()` function

4. **invokeai/backend/model_manager/load/model_loaders/flux2.py**
   - `Flux2CheckpointModel` loader class

5. **tests/backend/flux2/test_flux2_model.py**
   - Unit tests for detection and model instantiation

## Model Detection Logic

FLUX.2 models are detected by the presence of shared modulation layer keys:

```python
flux2_specific_keys = {
    "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.lin.weight",
}
```

Variant is determined by:
1. `img_in.weight.shape[0]` → hidden_size (3072 for 4B, 4096 for 9B)
2. `img_in.weight.shape[1]` → must be 128 (FLUX.2 signature)
3. Counting double_blocks and single_blocks
4. Presence of `.input_scale` keys → FP8 quantization

## Future Work

### Not Implemented (Out of Scope)

1. **Qwen3 Text Encoder Loader** - Requires new ModelType and loader
2. **FLUX.2 VAE (AutoencoderKLFlux2)** - Requires new VAE implementation
3. **Pipeline Invocations** - flux2_denoise.py, flux2_text_encoder.py, etc.
4. **Diffusers Format Support** - Loading from HuggingFace diffusers format

### Extension to 9B Models

The 9B variants require:
- Qwen3-8B text encoder (larger, possibly gated)
- More VRAM (~18GB for bf16)
- FP8 support for the quantized variant

## License

FLUX.2-klein-4B is released under **Apache 2.0** license (open weights, commercial use allowed).

## References

- [FLUX.2-klein-4B on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [FLUX.2 GitHub Repository](https://github.com/black-forest-labs/flux2)
- [BFL Blog Post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence)
