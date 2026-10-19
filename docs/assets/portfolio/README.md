# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

The figures below use English labels. The [Korean guide](README.ko.md) links to Korean figures with the same layout refinements. Both versions use the retained measurements referenced, with checksums, in `visual-data.json`. Translation does not change plotted trajectories, measured values, evaluation populations or scientific reports.

Both renderers use Python's standard library. The first regenerates the preserved Korean display figures from embedded data; the second creates English and Korean reading copies and checks that plotted geometry is unchanged. Neither reruns a flight or verifies its scientific evaluation.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
python docs/assets/portfolio/render_localized.py
```

`visual-receipt.json` binds the preserved Korean SVG bytes to the display data. Each language directory contains `localization-receipt.json` with source and output hashes and geometry checks. Download ZIPs remain versioned snapshots; these documentation additions do not replace them or introduce new benchmark results.

## Figures

- [Launch and recovery overview](en/dm100-thumbnail.svg)
- [Separate evaluation populations](en/dm100-validation.svg)
- [Matched RCS propellant comparison](en/dm100-propellant.svg)
- [Recorded reference trajectory](en/dm100-trajectory.svg)
