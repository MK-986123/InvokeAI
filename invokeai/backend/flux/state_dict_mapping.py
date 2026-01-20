# Initially pulled from https://github.com/black-forest-labs/flux

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

FluxStateDict = dict[str | int, Any]


@dataclass(frozen=True)
class FluxStateDictMappingEntry:
    """Defines how a checkpoint key maps to a runtime module name."""

    checkpoint_key: str
    runtime_module: str
    description: str | None = None


FLUX_STATE_DICT_MAPPING: tuple[FluxStateDictMappingEntry, ...] = (
    FluxStateDictMappingEntry("img_in", "Flux.img_in"),
    FluxStateDictMappingEntry("time_in", "Flux.time_in"),
    FluxStateDictMappingEntry("vector_in", "Flux.vector_in"),
    FluxStateDictMappingEntry("guidance_in", "Flux.guidance_in"),
    FluxStateDictMappingEntry("txt_in", "Flux.txt_in"),
    FluxStateDictMappingEntry("double_blocks.*.img_mod", "DoubleStreamBlock.img_mod"),
    FluxStateDictMappingEntry("double_blocks.*.img_attn", "DoubleStreamBlock.img_attn"),
    FluxStateDictMappingEntry("double_blocks.*.img_mlp", "DoubleStreamBlock.img_mlp"),
    FluxStateDictMappingEntry("double_blocks.*.txt_mod", "DoubleStreamBlock.txt_mod"),
    FluxStateDictMappingEntry("double_blocks.*.txt_attn", "DoubleStreamBlock.txt_attn"),
    FluxStateDictMappingEntry("double_blocks.*.txt_mlp", "DoubleStreamBlock.txt_mlp"),
    FluxStateDictMappingEntry(
        "single_blocks.*.linear1",
        "SingleStreamBlock.linear1",
        "Fused QKV projection + gated-MLP input projection.",
    ),
    FluxStateDictMappingEntry(
        "single_blocks.*.linear2",
        "SingleStreamBlock.linear2",
        "Fused attention output + MLP hidden output projection.",
    ),
    FluxStateDictMappingEntry("single_blocks.*.modulation", "SingleStreamBlock.modulation"),
    FluxStateDictMappingEntry("final_layer.linear", "LastLayer.linear"),
    FluxStateDictMappingEntry("final_layer.adaLN_modulation", "LastLayer.adaLN_modulation"),
)


@dataclass(frozen=True)
class FluxStateDictValidationResult:
    hidden_size: int | None
    mlp_ratio: float | None
    mlp_hidden_dim: int | None
    gated_mlp: bool | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


_MODEL_PREFIX = "model.diffusion_model."
_SINGLE_LINEAR1_RE = re.compile(r"^single_blocks\.(\d+)\.linear1\.weight$")
_SINGLE_LINEAR2_RE = re.compile(r"^single_blocks\.(\d+)\.linear2\.weight$")
_DOUBLE_QKV_RE = re.compile(r"^double_blocks\.(\d+)\.(img_attn|txt_attn)\.qkv\.weight$")


def _normalize_key(key: str) -> str:
    return key[len(_MODEL_PREFIX) :] if key.startswith(_MODEL_PREFIX) else key


def infer_flux_hidden_size(state_dict: FluxStateDict) -> int | None:
    """Infer hidden size from a FLUX transformer state dict."""
    for key in ("img_in.weight", "txt_in.weight"):
        for candidate in (key, f"{_MODEL_PREFIX}{key}"):
            tensor = state_dict.get(candidate)
            if tensor is not None and hasattr(tensor, "shape"):
                return int(tensor.shape[0])
    return None


def infer_flux_mlp_hidden_dim(state_dict: FluxStateDict, hidden_size: int) -> int | None:
    """Infer the MLP hidden dim from a SingleStreamBlock linear2 weight."""
    for key, tensor in _iter_tensor_items(state_dict):
        if _SINGLE_LINEAR2_RE.match(key):
            return int(tensor.shape[1] - hidden_size)
    return None


def validate_flux_state_dict_shapes(
    state_dict: FluxStateDict,
    *,
    hidden_size: int | None = None,
    mlp_ratio: float | None = None,
    gated_mlp: bool | None = None,
) -> FluxStateDictValidationResult:
    errors: list[str] = []

    inferred_hidden_size = hidden_size or infer_flux_hidden_size(state_dict)
    if inferred_hidden_size is None:
        errors.append("Unable to infer hidden_size from state dict.")
        return FluxStateDictValidationResult(None, mlp_ratio, None, gated_mlp, tuple(errors))

    inferred_mlp_hidden_dim = infer_flux_mlp_hidden_dim(state_dict, inferred_hidden_size)
    expected_mlp_hidden_dim = (
        int(inferred_hidden_size * mlp_ratio) if mlp_ratio is not None else inferred_mlp_hidden_dim
    )

    if mlp_ratio is not None and inferred_mlp_hidden_dim is not None and expected_mlp_hidden_dim != inferred_mlp_hidden_dim:
        errors.append(
            "MLP hidden dim mismatch: "
            f"expected {expected_mlp_hidden_dim} from mlp_ratio={mlp_ratio}, "
            f"found {inferred_mlp_hidden_dim} from linear2 weights."
        )

    mlp_hidden_dim = expected_mlp_hidden_dim
    if mlp_hidden_dim is None:
        errors.append("Unable to infer mlp_hidden_dim from state dict.")
        return FluxStateDictValidationResult(inferred_hidden_size, mlp_ratio, None, gated_mlp, tuple(errors))

    inferred_gated_mlp: bool | None = gated_mlp

    for key, tensor in _iter_tensor_items(state_dict):
        if _DOUBLE_QKV_RE.match(key):
            expected_qkv_out = 3 * inferred_hidden_size
            if tensor.shape[0] != expected_qkv_out:
                errors.append(
                    f"{key} has out_dim={tensor.shape[0]} but expected {expected_qkv_out} for QKV projection."
                )
        if _SINGLE_LINEAR1_RE.match(key):
            expected_qkv_out = 3 * inferred_hidden_size
            mlp_out_dim = int(tensor.shape[0] - expected_qkv_out)
            gated_detected = None
            if mlp_out_dim == mlp_hidden_dim:
                gated_detected = False
            elif mlp_out_dim == 2 * mlp_hidden_dim:
                gated_detected = True

            if gated_detected is None:
                errors.append(
                    f"{key} has mlp_out_dim={mlp_out_dim}; expected {mlp_hidden_dim} (ungated) "
                    f"or {2 * mlp_hidden_dim} (gated)."
                )
            else:
                if inferred_gated_mlp is None:
                    inferred_gated_mlp = gated_detected
                elif inferred_gated_mlp != gated_detected:
                    errors.append(
                        f"{key} gating mismatch: expected gated_mlp={inferred_gated_mlp} but found {gated_detected}."
                    )

            if tensor.shape[0] < expected_qkv_out:
                errors.append(
                    f"{key} has out_dim={tensor.shape[0]} but expected at least {expected_qkv_out} for QKV."
                )
        if _SINGLE_LINEAR2_RE.match(key):
            expected_in = inferred_hidden_size + mlp_hidden_dim
            if tensor.shape[1] != expected_in:
                errors.append(
                    f"{key} has in_dim={tensor.shape[1]} but expected {expected_in} for attn+MLP concat."
                )
            if tensor.shape[0] != inferred_hidden_size:
                errors.append(
                    f"{key} has out_dim={tensor.shape[0]} but expected {inferred_hidden_size}."
                )

    resolved_mlp_ratio = mlp_ratio
    if resolved_mlp_ratio is None:
        resolved_mlp_ratio = mlp_hidden_dim / inferred_hidden_size

    return FluxStateDictValidationResult(
        inferred_hidden_size, resolved_mlp_ratio, mlp_hidden_dim, inferred_gated_mlp, tuple(errors)
    )


def _iter_tensor_items(state_dict: FluxStateDict) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not hasattr(tensor, "shape"):
            continue
        items.append((_normalize_key(key), tensor))
    return items
