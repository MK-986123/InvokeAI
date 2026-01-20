from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


@dataclass
class FluxStateDictReport:
    total_keys: int
    inferred_dims: dict[str, int | None]
    block_counts: dict[str, int]
    fp8_layers: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]


def audit_flux_safetensors(path: str | Path) -> FluxStateDictReport:
    path = Path(path)
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

        def get_shape(key: str) -> tuple[int, ...] | None:
            """Get shape for a key, trying both unprefixed and prefixed variants."""
            for variant in [key, f"model.diffusion_model.{key}"]:
                if variant in f.keys():
                    tensor = f.get_tensor(variant)
                    return tuple(tensor.shape)
            return None

        img_in_shape = get_shape("img_in.weight")
        txt_in_shape = get_shape("txt_in.weight")
        vector_in_shape = get_shape("vector_in.in_layer.weight")
        time_in_shape = get_shape("time_in.in_layer.weight")
        q_norm_shape = get_shape("double_blocks.0.img_attn.norm.query_norm.scale")
        mlp_double_shape = get_shape("double_blocks.0.img_mlp.0.weight")
        mlp_single_shape = get_shape("single_blocks.0.linear1.weight")

        hidden_size = img_in_shape[0] if img_in_shape else None
        in_channels = img_in_shape[1] if img_in_shape else None
        context_in_dim = txt_in_shape[1] if txt_in_shape else None
        vec_in_dim = vector_in_shape[1] if vector_in_shape else None
        time_in_dim = time_in_shape[1] if time_in_shape else None
        head_dim = q_norm_shape[0] if q_norm_shape else None
        mlp_hidden_dim = mlp_double_shape[0] if mlp_double_shape else None
        if mlp_hidden_dim is None and mlp_single_shape is not None and hidden_size is not None:
            mlp_hidden_dim = mlp_single_shape[0] - (3 * hidden_size)

        double_blocks = 0
        while (
            f"double_blocks.{double_blocks}.img_attn.qkv.weight" in keys
            or f"model.diffusion_model.double_blocks.{double_blocks}.img_attn.qkv.weight" in keys
        ):
            double_blocks += 1

        single_blocks = 0
        while (
            f"single_blocks.{single_blocks}.linear1.weight" in keys
            or f"model.diffusion_model.single_blocks.{single_blocks}.linear1.weight" in keys
        ):
            single_blocks += 1

        fp8_layers = _collect_fp8_layers(keys)
        required_keys = _required_flux_keys()
        missing_keys = sorted([k for k in required_keys if k not in keys and f"model.diffusion_model.{k}" not in keys])
        unexpected_keys = sorted([k for k in keys if not _is_expected_prefix(k)])

    return FluxStateDictReport(
        total_keys=len(keys),
        inferred_dims={
            "hidden_size": hidden_size,
            "in_channels": in_channels,
            "context_in_dim": context_in_dim,
            "vec_in_dim": vec_in_dim,
            "time_in_dim": time_in_dim,
            "mlp_hidden_dim": mlp_hidden_dim,
            "head_dim": head_dim,
        },
        block_counts={"double_blocks": double_blocks, "single_blocks": single_blocks},
        fp8_layers=fp8_layers,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def _required_flux_keys() -> set[str]:
    return {
        "img_in.weight",
        "txt_in.weight",
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "vector_in.in_layer.weight",
        "vector_in.out_layer.weight",
        "final_layer.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    }


def _collect_fp8_layers(keys: list[str]) -> list[str]:
    fp8_bases: set[str] = set()
    for key in keys:
        if key.endswith((".input_scale", ".weight_scale")):
            base_key = key.rsplit(".", 1)[0]
            # Normalize to unprefixed form for consistent reporting
            if base_key.startswith("model.diffusion_model."):
                base_key = base_key[len("model.diffusion_model.") :]
            fp8_bases.add(base_key)
    return sorted(fp8_bases)


def _is_expected_prefix(key: str) -> bool:
    prefixes = (
        "img_in.",
        "txt_in.",
        "time_in.",
        "vector_in.",
        "guidance_in.",
        "double_blocks.",
        "single_blocks.",
        "final_layer.",
        "double_stream_modulation_",
        "single_stream_modulation.",
    )
    # Check both unprefixed and prefixed variants
    if key.startswith(prefixes):
        return True
    if key.startswith("model.diffusion_model."):
        unprefixed_key = key[len("model.diffusion_model.") :]
        return unprefixed_key.startswith(prefixes)
    return False
