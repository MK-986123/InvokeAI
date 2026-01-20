import pytest
import torch

from invokeai.backend.flux.modules.fp8_linear import Fp8Linear


def _supports_fp8_storage() -> bool:
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    try:
        sample = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
        sample.to(torch.float32)
    except (TypeError, RuntimeError):
        return False
    return True


@pytest.mark.skipif(not _supports_fp8_storage(), reason="float8 support not available in this torch build")
def test_fp8_scale_mode_matches_reference() -> None:
    torch.manual_seed(0)
    in_features = 4
    out_features = 3
    input_tensor = torch.randn(2, in_features, dtype=torch.float32)
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    bias = torch.randn(out_features, dtype=torch.float32)

    input_scale = torch.tensor(0.5, dtype=torch.float32)
    weight_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_q = (weight / weight_scale).to(torch.float8_e4m3fn)

    reference = torch.nn.functional.linear(input_tensor, weight_q.to(torch.float32) * weight_scale, bias)

    direct = Fp8Linear(in_features, out_features, bias=True, scale_mode="direct")
    direct.weight_f8.data = weight_q
    direct.bias.data = bias
    direct.input_scale.copy_(input_scale)
    direct.weight_scale.copy_(weight_scale)
    direct_output = direct(input_tensor)

    inverse = Fp8Linear(in_features, out_features, bias=True, scale_mode="inverse")
    inverse.weight_f8.data = weight_q
    inverse.bias.data = bias
    inverse.input_scale.copy_(input_scale)
    inverse.weight_scale.copy_(weight_scale)
    inverse_output = inverse(input_tensor)

    direct_error = torch.mean(torch.abs(direct_output - reference)).item()
    inverse_error = torch.mean(torch.abs(inverse_output - reference)).item()

    assert direct_error < inverse_error
