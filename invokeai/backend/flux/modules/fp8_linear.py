from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

ScaleMode = Literal["direct", "inverse"]


def interpret_fp8_scales(input_scale: torch.Tensor, weight_scale: torch.Tensor, scale_mode: ScaleMode) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_mode == "inverse":
        return input_scale.reciprocal(), weight_scale.reciprocal()
    return input_scale, weight_scale


def _make_scalar(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if device.type == "meta":
        return torch.empty((), device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


class Fp8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_mode: ScaleMode = "direct",
        force_fp8_gemm: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("This PyTorch build does not support float8_e4m3fn tensors.")
        self.in_features = in_features
        self.out_features = out_features
        self.scale_mode = scale_mode
        self.force_fp8_gemm = force_fp8_gemm

        if device is None:
            device = torch.device("cpu")

        self.weight_f8 = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("input_scale", _make_scalar(1.0, device=device, dtype=torch.float32))
        self.register_buffer("weight_scale", _make_scalar(1.0, device=device, dtype=torch.float32))

    @staticmethod
    def _fp8_gemm_available(device: torch.device) -> bool:
        if device.type != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        if not hasattr(torch, "float8_e4m3fn"):
            return False
        if not hasattr(torch, "_scaled_mm"):
            return False
        try:
            major, _ = torch.cuda.get_device_capability(device)
        except RuntimeError:
            return False
        return major >= 9

    def _float8_linear(self, input: torch.Tensor) -> torch.Tensor:
        scale_input, scale_weight = interpret_fp8_scales(self.input_scale, self.weight_scale, self.scale_mode)
        scale_input = scale_input.to(device=input.device, dtype=torch.float32)
        scale_weight = scale_weight.to(device=input.device, dtype=torch.float32)
        input_f8 = (input / scale_input).to(torch.float8_e4m3fn)

        input_2d = input_f8.reshape(-1, self.in_features)
        output_dtype = input.dtype if input.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        output = torch._scaled_mm(
            input_2d,
            self.weight_f8.t(),
            scale_input,
            scale_weight,
            out_dtype=output_dtype,
        )
        output = output.reshape(*input.shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype, device=output.device)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._fp8_gemm_available(input.device):
            return self._float8_linear(input)
        if self.force_fp8_gemm:
            raise RuntimeError(
                "FP8 GEMM execution was requested, but the current PyTorch build or CUDA device does not "
                "support float8 GEMM. Ensure you are running on a supported CUDA device with float8 support."
            )
        scale_weight = interpret_fp8_scales(self.input_scale, self.weight_scale, self.scale_mode)[1]
        weight = self.weight_f8.to(dtype=input.dtype) * scale_weight.to(device=input.device, dtype=input.dtype)
        bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        weight_key = prefix + "weight"
        if weight_key in state_dict and prefix + "weight_f8" not in state_dict:
            state_dict[prefix + "weight_f8"] = state_dict.pop(weight_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
