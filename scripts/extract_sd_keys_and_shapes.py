import argparse
import json

from safetensors.torch import load_file

from invokeai.backend.flux.state_dict_mapping import validate_flux_state_dict_shapes


def extract_sd_keys_and_shapes(
    safetensors_file: str,
    *,
    validate_flux: bool = False,
    mlp_ratio: float | None = None,
    gated_mlp: bool | None = None,
) -> None:
    sd = load_file(safetensors_file)

    keys_to_shapes = {k: v.shape for k, v in sd.items()}

    out_file = "keys_and_shapes.json"
    with open(out_file, "w") as f:
        json.dump(keys_to_shapes, f, indent=4)

    print(f"Keys and shapes written to '{out_file}'.")

    if validate_flux:
        validation = validate_flux_state_dict_shapes(sd, mlp_ratio=mlp_ratio, gated_mlp=gated_mlp)
        if not validation.ok:
            print("Flux state dict validation failed:")
            for error in validation.errors:
                print(f"  - {error}")
            raise SystemExit(1)

        print(
            "Flux state dict validation passed "
            f"(hidden_size={validation.hidden_size}, mlp_ratio={validation.mlp_ratio}, gated_mlp={validation.gated_mlp})."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extracts the keys and shapes from the state dict in a safetensors file. Intended for creating "
        + "dummy state dicts for use in unit tests."
    )
    parser.add_argument("safetensors_file", type=str, help="Path to the safetensors file.")
    parser.add_argument(
        "--validate-flux",
        action="store_true",
        help="Validate FLUX transformer fused-layer shapes after extracting keys.",
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=None,
        help="Optional MLP ratio override used for FLUX fused-layer validation.",
    )
    gated_group = parser.add_mutually_exclusive_group()
    gated_group.add_argument(
        "--gated-mlp",
        action="store_true",
        help="Assume a gated MLP when validating FLUX fused-layer shapes.",
    )
    gated_group.add_argument(
        "--ungated-mlp",
        action="store_true",
        help="Assume an ungated MLP when validating FLUX fused-layer shapes.",
    )
    args = parser.parse_args()
    gated_mlp = None
    if args.gated_mlp:
        gated_mlp = True
    elif args.ungated_mlp:
        gated_mlp = False
    extract_sd_keys_and_shapes(
        args.safetensors_file,
        validate_flux=args.validate_flux,
        mlp_ratio=args.mlp_ratio,
        gated_mlp=gated_mlp,
    )


if __name__ == "__main__":
    main()
