import pytest
import torch
from torch import nn

from invokeai.backend.flux.modules.fp8_linear import Fp8Linear, Fp8LinearConfig, apply_fp8_linear_from_state_dict


def _supports_float8() -> bool:
    return hasattr(torch, "float8_e4m3fn")


@pytest.mark.skipif(not _supports_float8(), reason="float8 dtype not available")
def test_fp8_linear_dequantizes_with_scales():
    linear = Fp8Linear(4, 3, bias=False, config=Fp8LinearConfig(use_native_fp8=False))
    weight_f32 = torch.randn(3, 4, dtype=torch.float32)
    linear.weight.data = weight_f32.to(torch.float8_e4m3fn)
    linear.input_scale.copy_(torch.tensor(2.0))
    linear.weight_scale.copy_(torch.tensor(0.5))

    x = torch.randn(2, 4, dtype=torch.float32)
    expected = torch.nn.functional.linear(x * 2.0, weight_f32 * 0.5)
    output = linear(x)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not _supports_float8(), reason="float8 dtype not available")
def test_apply_fp8_linear_from_state_dict_replaces_module():
    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 3, bias=False)

    model = Dummy()
    state_dict = {
        "linear.weight": torch.randn(3, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
        "linear.input_scale": torch.tensor(1.0, dtype=torch.float32),
        "linear.weight_scale": torch.tensor(1.0, dtype=torch.float32),
    }
    apply_fp8_linear_from_state_dict(model, state_dict)
    assert isinstance(model.linear, Fp8Linear)
