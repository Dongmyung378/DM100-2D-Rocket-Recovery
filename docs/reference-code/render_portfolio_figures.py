"""Apply DM100 presentation colors to the existing recorded-data plot functions.

No simulation, evaluation, model loading or report rewriting is performed.
The upstream plotting functions retain their data, axes, text and layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {
    "charcoal": "#374151",
    "slate": "#516B82",
    "efficiency": "#D47A28",
    "sensor": "#397EA4",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_snapshot(figure: Figure) -> str:
    """Bind plotted values and labels while excluding presentation colors."""
    records = []
    for axis in figure.axes:
        records.append(
            {
                "xlim": list(axis.get_xlim()),
                "ylim": list(axis.get_ylim()),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "title": axis.get_title(),
                "xticks": axis.get_xticks().tolist(),
                "yticks": axis.get_yticks().tolist(),
                "xticklabels": [item.get_text() for item in axis.get_xticklabels()],
                "yticklabels": [item.get_text() for item in axis.get_yticklabels()],
                "lines": [
                    {
                        "x": list(item.get_xdata()),
                        "y": list(item.get_ydata()),
                        "label": item.get_label(),
                        "linestyle": item.get_linestyle(),
                    }
                    for item in axis.lines
                ],
                "patches": [item.get_path().vertices.tolist() for item in axis.patches],
                "collections": [
                    item.get_segments()
                    for item in axis.collections
                    if hasattr(item, "get_segments")
                ],
                "text": [
                    {"text": item.get_text(), "position": list(item.get_position())}
                    for item in axis.texts
                ],
                "legend": [item.get_text() for item in axis.get_legend().get_texts()]
                if axis.get_legend()
                else None,
            }
        )

    def default(value):
        return value.tolist() if hasattr(value, "tolist") else float(value)

    raw = json.dumps(
        {"axes": records, "figure_text": [t.get_text() for t in figure.texts]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def apply_colors(figure: Figure, kind: str) -> None:
    if kind == "validation":
        for item, color in zip(
            figure.axes[0].patches,
            (PALETTE["charcoal"], PALETTE["slate"], PALETTE["sensor"]),
            strict=True,
        ):
            item.set_facecolor(color)
        for item in figure.axes[0].collections:
            item.set_color(PALETTE["charcoal"])
        for item, color in zip(
            figure.axes[1].patches, (PALETTE["slate"], PALETTE["efficiency"]), strict=True
        ):
            item.set_facecolor(color)
            item.set_edgecolor(PALETTE["charcoal"])
        for item in figure.axes[1].lines:
            item.set_color(PALETTE["charcoal"])
            item.set_markeredgecolor(PALETTE["charcoal"])
    else:
        roles = (("charcoal",), ("slate", "efficiency"), ("sensor",), ("slate", "efficiency"))
        for axis, names in zip(figure.axes, roles, strict=True):
            for item, name in zip(axis.lines, names, strict=True):
                item.set_color(PALETTE[name])
            if axis.get_legend():
                for item, name in zip(axis.get_legend().get_lines(), names, strict=True):
                    item.set_color(PALETTE[name])
    for axis in figure.axes:
        axis.title.set_color(PALETTE["charcoal"])
        axis.xaxis.label.set_color(PALETTE["charcoal"])
        axis.yaxis.label.set_color(PALETTE["charcoal"])
        for item in axis.texts:
            item.set_color(PALETTE["charcoal"])
    for item in figure.texts:
        item.set_color(PALETTE["charcoal"])


@contextmanager
def palette_save(kind: str, checks: list[dict]):
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        before = numerical_snapshot(figure)
        apply_colors(figure, kind)
        after = numerical_snapshot(figure)
        if before != after:
            raise ValueError("Presentation changed plotted values, axes or text")
        checks.append(
            {"plotted_data_axes_and_text_sha256": before, "unchanged_after_palette": True}
        )
        return original(figure, *args, **kwargs)

    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("validation", "trajectory"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-record", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        raise FileExistsError("Preserve an existing presentation receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.parent / (".palette-render-" + uuid.uuid4().hex + ".png")
    checks = []
    source = None
    plot_source = None
    try:
        with plt.rc_context(), palette_save(args.figure, checks):
            if args.figure == "validation":
                sys.path.insert(0, str(ROOT))
                from scripts.render_study_report import chart

                source = ROOT / "docs/results/performance.json"
                data = json.loads(source.read_text(encoding="utf-8"))
                summary = {
                    "holdout/full": data["new_conditions"],
                    "sensor-holdout/full": data["extended_sensors"],
                }
                plot_source = ROOT / "scripts/render_study_report.py"
                chart(summary, data["regression"], data["paired_fuel"], temporary)
            else:
                if args.trajectory_record is None:
                    parser.error("--trajectory-record is required for trajectory presentation")
                sys.path.insert(0, str(ROOT / "src"))
                from powered_landing_guidance.local_app import _save_plot

                source = args.trajectory_record
                raw = json.loads(source.read_text(encoding="utf-8"))
                plot_source = ROOT / "src/powered_landing_guidance/local_app.py"
                _save_plot(raw, temporary)
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        proof = {
            "schema_version": 1,
            "scope": "Presentation colors only, using existing recorded-data plot functions",
            "figure": args.figure,
            "palette": PALETTE,
            "matplotlib": matplotlib.__version__,
            "input_sha256": sha256(source),
            "upstream_plot_source_sha256": sha256(plot_source),
            "output_sha256": sha256(args.output),
            "checks": checks,
            "simulation_evaluation_or_training_run": False,
        }
        with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(proof, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
