#!/usr/bin/env python3
"""Verify a retained M3 gate receipt and saved exact outputs without inference."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_metal_profile import verify_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify_gate(args.directory)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "manifest"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
