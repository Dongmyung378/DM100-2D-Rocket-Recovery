"""Check shipped font and license bytes in distributable wheels using the standard library."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path


def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = list(directory.glob("powered_landing_guidance-*.whl"))
    if not wheels:
        raise SystemExit("No DM100 wheel was found")
    source = Path(__file__).resolve().parents[1] / "src/powered_landing_guidance"
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as package:
            for name in ("NotoSansKR-Regular.otf", "OFL.txt", "NOTICE.txt"):
                data = package.read(f"powered_landing_guidance/fonts/{name}")
                assert data == (source / "fonts" / name).read_bytes(), name
                if name.endswith(".otf"):
                    assert hashlib.sha256(data).hexdigest() == (
                        "69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68"
                    )
            assert "powered_landing_guidance/runtime_support.py" in package.namelist()
        print(f"Verified Korean font, license and runtime module in {wheel.name}")


if __name__ == "__main__":
    main()
