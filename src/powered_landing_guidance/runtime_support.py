"""Application-local fonts and video encoder, independent of system installations."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path

FONT_SHA256 = "69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68"


@lru_cache(maxsize=1)
def replay_font_family() -> str:
    """Register the distributed font in this process without a system font install."""
    from matplotlib import font_manager

    resource = resources.files("powered_landing_guidance").joinpath("fonts/NotoSansKR-Regular.otf")
    with resources.as_file(resource) as path:
        if hashlib.sha256(path.read_bytes()).hexdigest() != FONT_SHA256:
            raise RuntimeError("The bundled Korean font is missing or damaged. Run the launcher.")
        font_manager.fontManager.addfont(str(path))
        return font_manager.FontProperties(fname=str(path)).get_name()


def video_encoder_path() -> str:
    """Prefer the executable shipped with the tested video dependency."""
    from imageio_ffmpeg import get_ffmpeg_exe

    # An unrelated IMAGEIO_FFMPEG_EXE must not replace the application's encoder.
    from imageio_ffmpeg._utils import _get_ffmpeg_exe

    executable = _get_ffmpeg_exe() or get_ffmpeg_exe()
    if not executable or not Path(executable).is_file():
        raise RuntimeError("The packaged FFmpeg executable is unavailable. Run the launcher.")
    return executable


@lru_cache(maxsize=1)
def check_runtime_assets() -> dict[str, str]:
    family = replay_font_family()
    executable = video_encoder_path()
    result = subprocess.run(
        [
            executable,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.04",
            "-c:v",
            "libx264",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=20,
        check=False,
        **({"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}),
    )
    if result.returncode:
        raise RuntimeError("The packaged FFmpeg could not encode H.264. Run the launcher.")
    return {"font": family, "ffmpeg": executable, "h264_encoder": "ok"}


def main() -> None:
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.system(),
                "assets": check_runtime_assets(),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
