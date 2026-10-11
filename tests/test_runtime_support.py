"""Installed assets must render Korean and encode video without system packages."""

import os
import warnings
from importlib import resources

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from powered_landing_guidance.runtime_support import (  # noqa: E402
    check_runtime_assets,
    replay_font_family,
    video_encoder_path,
)


def test_bundled_font_renders_korean_without_missing_glyph_warnings():
    with matplotlib.rc_context({"font.family": replay_font_family()}), warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .*missing from font")
        fig, ax = plt.subplots()
        try:
            ax.set_title("한국어 비행 기록과 착륙 검증")
            ax.set_xlabel("시간 (초)")
            ax.set_ylabel("고도 (m), 자세 (°)")
            fig.canvas.draw()
        finally:
            plt.close(fig)
    assert resources.files("powered_landing_guidance").joinpath("fonts/OFL.txt").is_file()


def test_packaged_encoder_ignores_unrelated_system_override(monkeypatch):
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", os.devnull)
    assert "imageio_ffmpeg" in video_encoder_path()
    assert check_runtime_assets()["h264_encoder"] == "ok"
