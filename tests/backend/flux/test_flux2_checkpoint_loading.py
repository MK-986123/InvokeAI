import sys

import pytest
import torch

from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.model import Flux, FluxParams
from invokeai.backend.flux.text_conditioning import FluxRegionalTextConditioning
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Range

FLUX2_KLEIN_4B_PARAMS = FluxParams(
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=2048,
    mlp_ratio=4.0,
    num_heads=16,
    depth=18,
    depth_single_blocks=36,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)

FLUX2_KLEIN_9B_PARAMS = FluxParams(
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=24,
    depth_single_blocks=48,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)


def _build_synthetic_state_dict(params: FluxParams) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        model = Flux(params)
    return model.state_dict()


def _make_regional_prompting_extension(params: FluxParams, batch_size: int, txt_tokens: int) -> RegionalPromptingExtension:
    t5_embeddings = torch.zeros(batch_size, txt_tokens, params.context_in_dim)
    t5_txt_ids = torch.zeros(batch_size, txt_tokens, 3)
    clip_embeddings = torch.zeros(batch_size, params.vec_in_dim)
    regional_text_conditioning = FluxRegionalTextConditioning(
        t5_embeddings=t5_embeddings,
        t5_txt_ids=t5_txt_ids,
        clip_embeddings=clip_embeddings,
        image_masks=[None],
        t5_embedding_ranges=[Range(start=0, end=txt_tokens)],
    )
    return RegionalPromptingExtension(regional_text_conditioning=regional_text_conditioning, restricted_attn_mask=None)


@pytest.mark.skipif(sys.platform == "darwin", reason="Skipping on macOS")
@pytest.mark.parametrize("params", [FLUX2_KLEIN_4B_PARAMS, FLUX2_KLEIN_9B_PARAMS])
def test_flux2_klein_checkpoint_strict_load(params: FluxParams) -> None:
    sd = _build_synthetic_state_dict(params)
    with torch.device("meta"):
        model = Flux(params)
    result = model.load_state_dict(sd, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_flux2_klein_forward_smoke() -> None:
    params = FLUX2_KLEIN_4B_PARAMS
    model = Flux(params)
    model.eval()

    batch_size = 1
    img_tokens = 4
    txt_tokens = 8

    img = torch.randn(batch_size, img_tokens, params.in_channels)
    img_ids = torch.zeros(batch_size, img_tokens, 3)
    txt = torch.randn(batch_size, txt_tokens, params.context_in_dim)
    txt_ids = torch.zeros(batch_size, txt_tokens, 3)
    timesteps = torch.zeros(batch_size)
    y = torch.randn(batch_size, params.vec_in_dim)
    guidance = torch.ones(batch_size)

    regional_prompting_extension = _make_regional_prompting_extension(params, batch_size, txt_tokens)

    with torch.no_grad():
        output = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
            timestep_index=0,
            total_num_timesteps=1,
            controlnet_double_block_residuals=None,
            controlnet_single_block_residuals=None,
            ip_adapter_extensions=[],
            regional_prompting_extension=regional_prompting_extension,
        )

    expected_channels = params.out_channels or params.in_channels
    assert output.shape == (batch_size, img_tokens, expected_channels)
    assert torch.isfinite(output).all()
