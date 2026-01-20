import argparse
import json
from typing import Any, Dict, Iterable, List, Tuple

from safetensors import safe_open

REQUIRED_TOKENS = (
    "img_in",
    "txt_in",
    "time_in",
    "double_blocks",
    "single_blocks",
    "modulation",
    "final_layer",
)


def _matches_token(key: str, token: str) -> bool:
    if key.startswith(f"{token}."):
        return True
    if f".{token}." in key:
        return True
    if key.endswith(f".{token}"):
        return True
    return False


def _collect_tensor_inventory(path: str) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            inventory[key] = {
                "dtype": str(tensor_slice.get_dtype()),
                "shape": list(tensor_slice.get_shape()),
            }
    return inventory


def _find_key_by_suffix(keys: Iterable[str], suffixes: Tuple[str, ...]) -> str | None:
    for key in keys:
        for suffix in suffixes:
            if key.endswith(suffix):
                return key
    return None


def _infer_dims(inventory: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = list(inventory.keys())
    hidden_size = None
    context_in_dim = None
    mlp_ratio = None
    num_heads = None
    head_dim = None

    img_in_key = _find_key_by_suffix(keys, ("img_in.weight",))
    if img_in_key:
        hidden_size = inventory[img_in_key]["shape"][0]

    txt_in_key = _find_key_by_suffix(keys, ("txt_in.weight",))
    if txt_in_key:
        context_in_dim = inventory[txt_in_key]["shape"][1]

    mlp_key = _find_key_by_suffix(keys, ("double_blocks.0.img_mlp.0.weight",))
    if mlp_key and hidden_size:
        mlp_hidden_dim = inventory[mlp_key]["shape"][0]
        if hidden_size != 0:
            mlp_ratio = mlp_hidden_dim // hidden_size

    head_dim_key = _find_key_by_suffix(keys, ("double_blocks.0.img_attn.norm.query_norm.scale",))
    if head_dim_key and hidden_size:
        head_dim = inventory[head_dim_key]["shape"][0]
        if head_dim != 0:
            num_heads = hidden_size // head_dim

    # Count blocks using suffix matching to handle prefixed keys
    # (e.g., "model.diffusion_model." or "diffusion_model." prefixes)
    double_blocks = 0
    while _find_key_by_suffix(keys, (f"double_blocks.{double_blocks}.img_attn.qkv.weight",)):
        double_blocks += 1

    single_blocks = 0
    while _find_key_by_suffix(keys, (f"single_blocks.{single_blocks}.linear1.weight",)):
        single_blocks += 1

    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "mlp_ratio": mlp_ratio,
        "context_in_dim": context_in_dim,
        "double_blocks": double_blocks,
        "single_blocks": single_blocks,
    }


def _find_missing_required(keys: Iterable[str]) -> List[str]:
    missing = []
    for token in REQUIRED_TOKENS:
        if not any(_matches_token(key, token) for key in keys):
            missing.append(token)
    return missing


def _find_unexpected_keys(keys: Iterable[str]) -> List[str]:
    unexpected = []
    for key in keys:
        if not any(_matches_token(key, token) for token in REQUIRED_TOKENS):
            unexpected.append(key)
    return unexpected


def _render_report(
    path: str,
    inventory: Dict[str, Dict[str, Any]],
    inferred_dims: Dict[str, Any],
    missing_keys: List[str],
    unexpected_keys: List[str],
) -> str:
    lines = [f"Flux tensor inventory: {path}", f"Total tensors: {len(inventory)}"]
    lines.append("Inferred dims:")
    for key, value in inferred_dims.items():
        lines.append(f"  - {key}: {value}")
    lines.append("Required key groups:")
    if missing_keys:
        lines.append(f"  - missing: {', '.join(missing_keys)}")
    else:
        lines.append("  - missing: none")
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:10])
        extra = "" if len(unexpected_keys) <= 10 else f" (and {len(unexpected_keys) - 10} more)"
        lines.append(f"  - unexpected: {preview}{extra}")
    else:
        lines.append("  - unexpected: none")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory a FLUX safetensors file without loading tensors.",
    )
    parser.add_argument("safetensors_file", type=str, help="Path to the safetensors file.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output for tests.")
    args = parser.parse_args()

    inventory = _collect_tensor_inventory(args.safetensors_file)
    inferred_dims = _infer_dims(inventory)
    missing_keys = _find_missing_required(inventory.keys())
    unexpected_keys = _find_unexpected_keys(inventory.keys())

    if args.json:
        payload = {
            "path": args.safetensors_file,
            "inferred_dims": inferred_dims,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "tensors": inventory,
        }
        print(json.dumps(payload, indent=2))
        return

    print(_render_report(args.safetensors_file, inventory, inferred_dims, missing_keys, unexpected_keys))


if __name__ == "__main__":
    main()
