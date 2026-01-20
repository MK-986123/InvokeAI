from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Fp8LinearConfig:
    use_native_fp8: bool = True
    dequantize_dtype: torch.dtype | None = None


class Fp8Linear(nn.Module):
    """Linear layer that loads FP8 weights with per-layer scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Fp8LinearConfig | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.config = config or Fp8LinearConfig()

    def _dequantize_weight(self, target_dtype: torch.dtype) -> Tensor:
        return self.weight.to(target_dtype) * self.weight_scale.to(target_dtype)

    def _scale_input(self, x: Tensor) -> Tensor:
        return x * self.input_scale.to(x.dtype)

    def _can_use_native_fp8(self, x: Tensor) -> bool:
        return (
            self.config.use_native_fp8
            and x.is_cuda
            and self.weight.is_cuda
            and hasattr(torch, "float8_e4m3fn")
        )

    def _native_fp8_linear(self, x: Tensor) -> Tensor | None:
        if not hasattr(torch, "_scaled_mm"):
            return None
        try:
            out = torch._scaled_mm(
                x,
                self.weight.t(),
                self.input_scale,
                self.weight_scale,
                out_dtype=x.dtype,
            )
        except RuntimeError:
            return None
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def forward(self, x: Tensor) -> Tensor:
        if self._can_use_native_fp8(x):
            out = self._native_fp8_linear(x)
            if out is not None:
                return out
        target_dtype = self.config.dequantize_dtype or x.dtype
        return F.linear(self._scale_input(x), self._dequantize_weight(target_dtype), self.bias)


def apply_fp8_linear_from_state_dict(model: nn.Module, state_dict: dict[str, Tensor]) -> None:
    """Replace Linear layers with Fp8Linear modules for fp8 checkpoints."""

    fp8_bases: set[str] = set()
    for key in state_dict:
        if key.endswith(".input_scale"):
            fp8_bases.add(key[: -len(".input_scale")])
        elif key.endswith(".weight_scale"):
            fp8_bases.add(key[: -len(".weight_scale")])

    for base in sorted(fp8_bases):
        weight_key = f"{base}.weight"
        if weight_key not in state_dict:
            continue
        weight = state_dict[weight_key]
        if not torch.is_floating_point(weight):
            continue
        out_features, in_features = weight.shape
        bias = f"{base}.bias" in state_dict
        fp8_linear = Fp8Linear(in_features=in_features, out_features=out_features, bias=bias)
        _set_submodule(model, base, fp8_linear)


def _set_submodule(model: nn.Module, module_path: str, module: nn.Module) -> None:
    parts = module_path.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module  # type: ignore[index]
    else:
        setattr(parent, last, module)
