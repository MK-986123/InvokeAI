#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from invokeai.backend.flux.state_dict_audit import audit_flux_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a FLUX safetensors checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to a FLUX safetensors checkpoint.")
    args = parser.parse_args()

    report = audit_flux_safetensors(args.checkpoint)
    print(json.dumps(report.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
