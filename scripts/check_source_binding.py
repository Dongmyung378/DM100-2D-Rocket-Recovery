"""Check that published experiment source hashes match the distributed files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocols", type=Path, nargs="+")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    checked = set()
    root = args.root.resolve()
    for protocol_path in args.protocols:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        for name, expected in protocol["source_hashes"].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source filename must stay inside the repository")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Distributed source differs from recorded source: {name}")
            checked.add(name)
    print(f"Verified {len(checked)} distributed source files against recorded experiment hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
