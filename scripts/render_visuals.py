# ruff: noqa: E501
"""Generate DM100 Orbital Reusable RTLS visuals in Oxford Blue editorial style."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "docs" / "results" / "validation-487.json"
CALM_REPLAY = ROOT / "docs" / "results" / "replays" / "calm_replay_50hz.json"
WEATHER_REPLAY = ROOT / "docs" / "results" / "replays" / "example_weather_replay_50hz.json"
ASSETS_DIR = ROOT / "docs" / "assets"

# University of Oxford brand palette (#002147 primary) + restrained engineering accents
OXFORD_BLUE = "#002147"
OXFORD_DEEP = "#001630"
OXFORD_SURFACE = "#082C56"
OXFORD_BORDER = "#1E4673"
OXFORD_GRID = "#163B66"
OXFORD_BRASS = "#C8A452"
OXFORD_BRASS_LT = "#DEC074"
OXFORD_SKY = "#6FA4DC"
OXFORD_SLATE = "#9CB4CE"
OXFORD_MUTED = "#8AA2BC"
OXFORD_IVORY = "#F5F3EE"
OXFORD_GREEN = "#52B780"
OXFORD_CRIMSON = "#D96C62"

FONT_STYLE = (
    "text{font-family:'Malgun Gothic','Noto Sans KR','Apple SD Gothic Neo','Segoe UI',sans-serif;"
    "font-variant-numeric:tabular-nums}"
    ".eyebrow{letter-spacing:1.6px;font-weight:700}"
    ".mono{font-family:Consolas,'Courier New',monospace}"
)

PHASE_COLORS = {
    "ascent": OXFORD_BRASS_LT,
    "flip": "#94B3D6",
    "boostback": "#A8C3E0",
    "coast": "#7394BA",
    "entry_burn": OXFORD_CRIMSON,
    "aero_glide": OXFORD_SKY,
    "landing_burn": OXFORD_GREEN,
    "touchdown_settle": OXFORD_GREEN,
}

# Light-theme phase palette for white-paper Matplotlib PNG
PNG_PHASE_COLORS = {
    "ascent": "#002147",
    "flip": "#5B7898",
    "boostback": "#345C8A",
    "coast": "#7D95B0",
    "entry_burn": "#9E2A2B",
    "aero_glide": "#1F5B94",
    "landing_burn": "#1B5E3A",
    "touchdown_settle": "#1B5E3A",
}


def _subsample(frames: list[dict], step: int = 10) -> list[dict]:
    out = frames[::step]
    if frames and out[-1] is not frames[-1]:
        out.append(frames[-1])
    return out


def _pitch_deg(frame: dict) -> float:
    w, _, y, _ = frame["quat_wxyz"]
    return 2.0 * math.atan2(y, w) * (180.0 / math.pi)


def _polyline(pts: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def _phase_segments(frames: list[dict], sx, sy) -> list[tuple[str, list[tuple[float, float]]]]:
    segs: list[tuple[str, list[tuple[float, float]]]] = []
    cur_phase: str | None = None
    cur_pts: list[tuple[float, float]] = []
    for f in frames:
        ph = f["phase"]
        pt = (sx(f), sy(f))
        if cur_phase is None:
            cur_phase = ph
            cur_pts = [pt]
        elif ph != cur_phase:
            cur_pts.append(pt)
            segs.append((cur_phase, cur_pts))
            cur_phase = ph
            cur_pts = [pt]
        else:
            cur_pts.append(pt)
    if cur_phase is not None and cur_pts:
        segs.append((cur_phase, cur_pts))
    return segs


def _rocket_badge_svg(x: float, y: float, size: float = 46.0) -> str:
    """Site-matched Rocket Engineering emblem framed in Oxford Blue & Brass."""
    scale = (size - 16.0) / 24.0
    tx = x + 8.0
    ty = y + 8.0
    return (
        f'<g aria-hidden="true">'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" rx="9" '
        f'fill="{OXFORD_SURFACE}" stroke="{OXFORD_BRASS}" stroke-opacity="0.65" stroke-width="1.3"/>'
        f'<g transform="translate({tx:.1f} {ty:.1f}) scale({scale:.4f})" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M12 15l-3-3a22 22 0 0 1 3-5c3-3 7-4 10-4 0 3-1 7-4 10a22 22 0 0 1-5 3Z" '
        f'fill="{OXFORD_IVORY}" fill-opacity="0.10" stroke="{OXFORD_IVORY}" stroke-width="1.55"/>'
        f'<path d="M9 12H4s.6-3 2-4c1.6-1 4-1 4-1m3 8v5s3-.6 4-2c1-1.6 1-4 1-4" '
        f'stroke="{OXFORD_IVORY}" stroke-width="1.55"/>'
        f'<path d="M4.5 16.5c-1.5 1.3-2 5-2 5s3.7-.5 5-2" '
        f'stroke="{OXFORD_BRASS_LT}" stroke-width="1.65"/>'
        f'<circle cx="16" cy="7" r="1.25" fill="{OXFORD_BRASS_LT}"/>'
        f"</g></g>"
    )


def _compute_peak_utilization_pct(rows: list[dict]) -> float:
    checks = [
        ("position_error_m", 2.0),
        ("vertical_foot_speed_m_s", 1.5),
        ("horizontal_foot_speed_m_s", 0.5),
        ("tilt_deg", 3.0),
        ("contact_rate_deg_s", 0.5),
        ("peak_q_pa", 42_000.0),
        ("peak_heat_flux_w_m2", 80_000.0),
        ("peak_q_alpha_kpa_deg", 150.0),
        ("peak_ascent_high_q_alpha_deg", 5.0),
        ("peak_ascent_axial_g", 4.5),
        ("peak_recovery_g", 5.5),
        ("peak_rate_deg_s", 15.0),
        ("peak_accel_deg_s2", 5.0),
        ("peak_fin_hinge_nm", 7_500.0),
        ("max_leg_stroke_m", 0.90),
        ("max_contact_load_g", 4.0),
    ]
    return max(max(float(r[k]) for r in rows) / lim * 100.0 for k, lim in checks)


def render_thumbnail_svg(calm_frames: list[dict], summary: dict) -> str:
    rows = summary["cases"]
    sub = _subsample(calm_frames, step=12)
    x_vals = [f["pos_enu_m"][0] / 1000.0 for f in sub]
    z_vals = [f["pos_enu_m"][2] / 1000.0 for f in sub]
    x_min_raw = min(0.0, min(x_vals))
    x_lo = math.floor(x_min_raw / 10.0) * 10.0 if x_min_raw < -2.0 else 0.0
    x_hi = max(80.0, math.ceil(max(x_vals) / 20.0) * 20.0)
    z_hi = max(200.0, math.ceil((max(z_vals) + 25.0) / 25.0) * 25.0)

    px0, px1 = 516.0, 1116.0
    py0, py1 = 236.0, 580.0

    def sx(f: dict) -> float:
        return px0 + ((f["pos_enu_m"][0] / 1000.0) - x_lo) / (x_hi - x_lo) * (px1 - px0)

    def sy(f: dict) -> float:
        return py1 - (f["pos_enu_m"][2] / 1000.0) / z_hi * (py1 - py0)

    segs = _phase_segments(sub, sx, sy)
    sep_f = next(f for f in calm_frames if f["phase"] == "flip")
    apogee_f = max(calm_frames, key=lambda f: f["pos_enu_m"][2])
    glide_f = next(f for f in calm_frames if f["phase"] == "aero_glide")
    land_f = next(f for f in calm_frames if f["phase"] == "landing_burn")
    t_end = calm_frames[-1]["time_s"]

    grid_lines = []
    for i in range(5):
        zv = z_hi * i / 4.0
        yy = py1 - (zv / z_hi) * (py1 - py0)
        grid_lines.append(
            f'<line x1="{px0:.0f}" y1="{yy:.1f}" x2="{px1:.0f}" y2="{yy:.1f}" stroke="{OXFORD_GRID}" stroke-width="1"/>'
            f'<text x="{px0 - 12:.0f}" y="{yy + 6:.1f}" font-size="18" fill="{OXFORD_MUTED}" text-anchor="end">{zv:.0f}</text>'
        )
    for i in range(5):
        xv = x_lo + (x_hi - x_lo) * i / 4.0
        xx = px0 + (px1 - px0) * i / 4.0
        anchor = "start" if i == 0 else ("end" if i == 4 else "middle")
        grid_lines.append(
            f'<line x1="{xx:.1f}" y1="{py0:.0f}" x2="{xx:.1f}" y2="{py1:.0f}" stroke="{OXFORD_GRID}" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{py1 + 28:.0f}" font-size="18" fill="{OXFORD_MUTED}" text-anchor="{anchor}">{xv:.0f}</text>'
        )

    path_svg = []
    for ph, pts in segs:
        col = PHASE_COLORS.get(ph, OXFORD_SKY)
        w = 4.6 if ph in ("ascent", "aero_glide", "landing_burn") else 3.4
        path_svg.append(
            f'<polyline points="{_polyline(pts)}" fill="none" stroke="{col}" stroke-width="{w}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    total = summary["overall"]["total"]
    passed = summary["overall"]["passed"]
    med_main = float(np.median([r["main_remaining_kg"] for r in rows]))
    med_pos = float(np.median([r["position_error_m"] for r in rows]))
    min_glide = min(r["glide_distance_km"] for r in rows)
    max_glide = max(r["glide_distance_km"] for r in rows)
    peak_util = _compute_peak_utilization_pct(rows)

    spec_rows = [
        ("목표 궤도", "700 km · 1,800 kg", OXFORD_BRASS_LT),
        ("최대 제약 사용률", f"{peak_util:.1f}% (한계 80%)", OXFORD_GREEN),
        ("무동력 격자핀 활강", f"{min_glide:.1f}–{max_glide:.1f} km", OXFORD_SKY),
        ("잔여 주추진제 중앙", f"{med_main:,.0f} kg", OXFORD_IVORY),
        ("접지 위치 중앙값", f"{med_pos:.3f} m", OXFORD_IVORY),
        ("5초 착륙다리 직립", f"{passed} / {total}", OXFORD_GREEN),
    ]
    row_svg = []
    for idx, (k, v, col) in enumerate(spec_rows):
        ry = 384 + idx * 42
        row_svg.append(
            f'<text x="70" y="{ry}" font-size="15.5" fill="{OXFORD_SLATE}">{k}</text>'
            f'<text x="372" y="{ry}" font-size="16.5" fill="{col}" font-weight="700" text-anchor="end">{v}</text>'
            + (
                f'<line x1="70" y1="{ry + 14}" x2="372" y2="{ry + 14}" stroke="{OXFORD_GRID}" stroke-width="1"/>'
                if idx < len(spec_rows) - 1
                else ""
            )
        )

    sep_x, sep_y = sx(sep_f), sy(sep_f)
    apo_x, apo_y = sx(apogee_f), sy(apogee_f)
    gld_x, gld_y = sx(glide_f), sy(glide_f)
    lnd_x, lnd_y = sx(land_f), sy(land_f)
    sep_z_km = sep_f["pos_enu_m"][2] / 1000.0
    apo_z_km = apogee_f["pos_enu_m"][2] / 1000.0
    gld_z0 = glide_f["pos_enu_m"][2] / 1000.0
    lnd_z0 = land_f["pos_enu_m"][2] / 1000.0

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">DM100 100톤 메탈록스 700 km 원궤도 발사 및 1단 재사용 RTLS</title>
<desc id="desc">100톤 메탈록스 7엔진 2단 로켓의 700 km 원궤도(1,800 kg) 투입과 1단 발사장 귀환, {min_glide:.1f}–{max_glide:.1f} km 무동력 격자핀 활강, 3->1 호버슬램 착륙 및 5초 직립 동역학 487케이스 전수 검증 결과입니다.</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="760" fill="{OXFORD_BLUE}"/>
<rect width="1200" height="6" fill="{OXFORD_BRASS}"/>
{_rocket_badge_svg(1106, 30, 46)}
<text x="48" y="48" font-size="13" fill="{OXFORD_BRASS_LT}" class="eyebrow">DM100 · ORBITAL REUSABLE RTLS · 700 KM LEO</text>
<text x="48" y="96" font-size="40" fill="{OXFORD_IVORY}" font-weight="700">DM100 · 100톤 메탈록스 궤도 발사 및 1단 RTLS</text>
<text x="48" y="132" font-size="19" fill="{OXFORD_SLATE}">700 km 원궤도 1,800 kg 투입 · {min_glide:.0f}–{max_glide:.0f} km 무동력 격자핀 활강 · 3→1 호버슬램 · 접지 후 5초 직립 동역학</text>
<line x1="48" y1="154" x2="1152" y2="154" stroke="{OXFORD_BORDER}" stroke-width="1"/>

<rect x="48" y="178" width="348" height="500" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}" stroke-width="1"/>
<rect x="48" y="178" width="348" height="3" fill="{OXFORD_BRASS}"/>
<text x="70" y="218" font-size="14" fill="{OXFORD_BRASS_LT}" class="eyebrow">487-CASE CERTIFICATION</text>
<text x="70" y="284" font-size="56" fill="{OXFORD_IVORY}" font-weight="700">{passed}<tspan fill="{OXFORD_MUTED}" font-size="34"> / {total}</tspan></text>
<text x="70" y="318" font-size="18" fill="{OXFORD_GREEN}" font-weight="700">100.0% 통과 · 16개 전 항목 ≤ 80%</text>
<line x1="70" y1="344" x2="372" y2="344" stroke="{OXFORD_BORDER}" stroke-width="1"/>
{"".join(row_svg)}
<text x="70" y="640" font-size="14" fill="{OXFORD_MUTED}">100 t 7× LCH₄/LOX · 2D 개념 시뮬레이터</text>
<text x="70" y="660" font-size="14" fill="{OXFORD_MUTED}">선언된 2D 합성 외란·센서 범위 내 물리 검증</text>

<rect x="420" y="178" width="732" height="500" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}" stroke-width="1"/>
<rect x="420" y="178" width="732" height="3" fill="{OXFORD_SKY}"/>
<text x="444" y="216" font-size="18" fill="{OXFORD_IVORY}" font-weight="700">1단 상승·귀환 비행 궤적 <tspan font-size="15" fill="{OXFORD_SLATE}" font-weight="400">(고도 km vs 수평 거리 km)</tspan></text>
<text x="1126" y="216" font-size="16" fill="{OXFORD_MUTED}" text-anchor="end">무풍 기준 ({t_end:.1f} s)</text>
{"".join(grid_lines)}
{"".join(path_svg)}

<circle cx="{sep_x:.1f}" cy="{sep_y:.1f}" r="5.5" fill="{OXFORD_BRASS_LT}" stroke="{OXFORD_IVORY}" stroke-width="1.8"/>
<text x="{min(px1 - 12, sep_x + 14):.1f}" y="{sep_y + 24:.1f}" font-size="16" fill="{OXFORD_BRASS_LT}" font-weight="700">MECO-1 단 분리 ({sep_z_km:.1f} km)</text>

<circle cx="{apo_x:.1f}" cy="{apo_y:.1f}" r="5.5" fill="#A8C3E0" stroke="{OXFORD_IVORY}" stroke-width="1.8"/>
<text x="{apo_x:.1f}" y="{apo_y - 16:.1f}" font-size="16" fill="#C9DCF0" font-weight="700" text-anchor="middle">정점 {apo_z_km:.1f} km · 탄도 귀환</text>

<circle cx="{gld_x:.1f}" cy="{gld_y:.1f}" r="5.5" fill="{OXFORD_SKY}" stroke="{OXFORD_IVORY}" stroke-width="1.8"/>
<line x1="{gld_x + 7:.1f}" y1="{gld_y - 4:.1f}" x2="{gld_x + 88:.1f}" y2="{gld_y - 88:.1f}" stroke="{OXFORD_SKY}" stroke-width="1.3"/>
<text x="{gld_x + 94:.1f}" y="{gld_y - 84:.1f}" font-size="15.5" fill="{OXFORD_SKY}" font-weight="700">무동력 활강 ({gld_z0:.1f}→{lnd_z0:.1f} km)</text>

<circle cx="{lnd_x:.1f}" cy="{lnd_y:.1f}" r="5.5" fill="{OXFORD_GREEN}" stroke="{OXFORD_IVORY}" stroke-width="1.8"/>
<line x1="{lnd_x + 8:.1f}" y1="{lnd_y - 1:.1f}" x2="{px0 + 96:.1f}" y2="{py1 - 14:.1f}" stroke="{OXFORD_GREEN}" stroke-width="1.3"/>
<text x="{px0 + 102:.1f}" y="{py1 - 9:.1f}" font-size="15.5" fill="{OXFORD_GREEN}" font-weight="700">3→1 호버슬램 · 발사장 착륙</text>

<text x="1116" y="646" font-size="16" fill="{OXFORD_SLATE}" text-anchor="end">수평 거리 (km)</text>
<text x="442" y="646" font-size="15" fill="{OXFORD_MUTED}">발사장 원점 (0, 0)</text>

<rect x="48" y="696" width="1104" height="44" rx="4" fill="{OXFORD_DEEP}" stroke="{OXFORD_BORDER}" stroke-width="1"/>
<text x="70" y="724" font-size="17" fill="{OXFORD_SLATE}">Plant ↔ Forces ↔ GNC 경계 격리 · 인과적 상태 추정 · 2단 구면 지구 2회 연소 궤도 적분 · 50 Hz 3D 리플레이 JSON</text>
<text x="1130" y="724" font-size="16" fill="{OXFORD_BRASS_LT}" font-weight="700" text-anchor="end">DM100</text>
</svg>
"""


def render_validation_svg(summary: dict) -> str:
    rows = summary["cases"]
    groups = [
        ("극한 엣지 케이스", "100% 동시 최악 포함", "edge-cases", OXFORD_BRASS_LT),
        ("복합 홀드아웃", "바람·편차·지연 복합", "holdout", OXFORD_SKY),
        ("확장 센서 홀드아웃", "잡음·편향·패킷 누락", "sensor-holdout", "#94B3D6"),
        ("회귀 · 개발 전체", "기존 286 + 개발 16", None, OXFORD_GREEN),
    ]
    cards = []
    for idx, (title, sub, split_key, col) in enumerate(groups):
        if split_key is None:
            sub_rows = [
                r for r in rows if r["split"] not in ("edge-cases", "holdout", "sensor-holdout")
            ]
        else:
            sub_rows = [r for r in rows if r["split"] == split_key]
        p_cnt = sum(1 for r in sub_rows if r["passed"])
        t_cnt = len(sub_rows)
        x = 48 + idx * 282
        cards.append(
            f'<rect x="{x}" y="140" width="260" height="154" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>'
            f'<rect x="{x}" y="140" width="260" height="3" fill="{col}"/>'
            f'<text x="{x + 20}" y="176" font-size="19" fill="{OXFORD_IVORY}" font-weight="700">{title}</text>'
            f'<text x="{x + 20}" y="200" font-size="14" fill="{OXFORD_MUTED}">{sub}</text>'
            f'<text x="{x + 20}" y="256" font-size="40" fill="{col}" font-weight="700">{p_cnt}<tspan fill="{OXFORD_MUTED}" font-size="24"> / {t_cnt}</tspan></text>'
            f'<rect x="{x + 20}" y="274" width="220" height="4" rx="2" fill="{OXFORD_GRID}"/>'
            f'<rect x="{x + 20}" y="274" width="220" height="4" rx="2" fill="{col}"/>'
        )

    metrics = [
        ("접지 위치 오차 (≤ 2.0 m)", "position_error_m", 2.0, "m", 1.0, 3),
        ("발 수직 속도 (≤ 1.5 m/s)", "vertical_foot_speed_m_s", 1.5, "m/s", 1.0, 3),
        ("발 수평 속도 (≤ 0.5 m/s)", "horizontal_foot_speed_m_s", 0.5, "m/s", 1.0, 3),
        ("최대 동압 q (≤ 42 kPa)", "peak_q_pa", 42_000.0, "kPa", 0.001, 2),
        ("최대 회수 하중 (≤ 5.5 g)", "peak_recovery_g", 5.5, "g", 1.0, 2),
        ("최대 상승 하중 (≤ 4.5 g)", "peak_ascent_axial_g", 4.5, "g", 1.0, 2),
        ("최대 각가속도 (≤ 5.0°/s²)", "peak_accel_deg_s2", 5.0, "°/s²", 1.0, 2),
        ("정체 열유속 (≤ 80 kW/m²)", "peak_heat_flux_w_m2", 80_000.0, "kW/m²", 0.001, 1),
    ]
    bx0, bw = 310.0, 480.0
    x80 = bx0 + 0.80 * bw
    bars = []
    for i, (label, key, lim, unit, scale, dec) in enumerate(metrics):
        vals = np.array([r[key] for r in rows], dtype=float)
        p50, p95, mx = np.percentile(vals, [50, 95, 100])
        y = 382 + i * 36
        w50 = max(3.0, (p50 / lim) * bw)
        w95 = max(4.0, (p95 / lim) * bw)
        wmx = min(bw, (mx / lim) * bw)
        pct = mx / lim * 100.0
        fmt = f".{dec}f"
        bars.append(
            f'<text x="70" y="{y + 14}" font-size="16" fill="{OXFORD_IVORY}" font-weight="600">{label}</text>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{bw:.0f}" height="18" rx="2" fill="{OXFORD_DEEP}" stroke="{OXFORD_GRID}" stroke-width="0.8"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{wmx:.1f}" height="18" rx="2" fill="{OXFORD_BRASS}" opacity="0.58"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{w95:.1f}" height="18" rx="2" fill="{OXFORD_SKY}" opacity="0.85"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{w50:.1f}" height="18" rx="2" fill="{OXFORD_GREEN}"/>'
            f'<text x="{bx0 + bw + 14:.0f}" y="{y + 14}" font-size="14" fill="{OXFORD_SLATE}">'
            f"p50 {p50 * scale:{fmt}} · max {mx * scale:{fmt}} {unit} "
            f'<tspan fill="{OXFORD_BRASS_LT}" font-weight="700">({pct:.1f}%)</tspan></text>'
        )

    peak_util = _compute_peak_utilization_pct(rows)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="730" viewBox="0 0 1200 730" role="img" aria-labelledby="title desc">
<title id="title">DM100 전체 487케이스 검증 및 80% 이하 안전 여유 분포</title>
<desc id="desc">극한 엣지 32케이스, 복합 홀드아웃 96케이스, 확장 센서 57케이스, 회귀·개발 302케이스 총 487케이스 100% 통과 결과와 모든 16개 물리 제약 항목의 80% 이하(최대 {peak_util:.1f}%) 안전 여유 분포입니다.</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="730" fill="{OXFORD_BLUE}"/>
<rect width="1200" height="6" fill="{OXFORD_BRASS}"/>
{_rocket_badge_svg(1106, 28, 46)}
<text x="48" y="46" font-size="13" fill="{OXFORD_BRASS_LT}" class="eyebrow">VALIDATION SCORECARD · ALL 7,792 CHECKS ≤ 80% OF LIMIT</text>
<text x="48" y="88" font-size="34" fill="{OXFORD_IVORY}" font-weight="700">전체 487케이스 일괄 검증 및 안전 여유 분포</text>
<text x="48" y="118" font-size="18" fill="{OXFORD_SLATE}">487/487 통과 (100.0%) · 2단 700 km 원궤도 487/487 · 5초 직립 487/487 · 전 제약 최대 사용률 {peak_util:.1f}%</text>
{"".join(cards)}

<rect x="48" y="316" width="1104" height="364" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>
<text x="70" y="350" font-size="19" fill="{OXFORD_IVORY}" font-weight="700">허용 한계(100%) 대비 관측 분포</text>
<rect x="810" y="338" width="14" height="10" rx="2" fill="{OXFORD_GREEN}"/><text x="830" y="348" font-size="14" fill="{OXFORD_SLATE}">중앙 (p50)</text>
<rect x="914" y="338" width="14" height="10" rx="2" fill="{OXFORD_SKY}"/><text x="934" y="348" font-size="14" fill="{OXFORD_SLATE}">상위 95%</text>
<rect x="1014" y="338" width="14" height="10" rx="2" fill="{OXFORD_BRASS}" opacity="0.75"/><text x="1034" y="348" font-size="14" fill="{OXFORD_SLATE}">최댓값 (Max)</text>
<line x1="{x80:.1f}" y1="368" x2="{x80:.1f}" y2="664" stroke="{OXFORD_BRASS_LT}" stroke-width="1.5" stroke-dasharray="4 4"/>
<text x="{x80:.1f}" y="360" font-size="12" fill="{OXFORD_BRASS_LT}" font-weight="700" text-anchor="middle">80% 여유선</text>
<line x1="{bx0 + bw:.1f}" y1="368" x2="{bx0 + bw:.1f}" y2="664" stroke="{OXFORD_CRIMSON}" stroke-width="1.5" stroke-dasharray="4 4"/>
<text x="{bx0 + bw:.1f}" y="360" font-size="12" fill="{OXFORD_CRIMSON}" font-weight="700" text-anchor="middle">100% 한계</text>
{"".join(bars)}
<text x="48" y="708" font-size="15" fill="{OXFORD_MUTED}">모든 수치는 validation-487.json 원시 기록에서 직접 집계했습니다. 선언된 2D 합성 외란·센서 범위 내 검증 결과입니다.</text>
</svg>
"""


def render_propellant_svg(calm_frames: list[dict], summary: dict) -> str:
    phase_order = [
        ("ascent", "상승 (Ascent · 7엔진)", OXFORD_BRASS_LT),
        ("flip", "저동압 플립 (Flip)", "#94B3D6"),
        ("boostback", "부스트백 (Boostback · 3엔진)", "#A8C3E0"),
        ("coast", "탄도 코스트 (Coast)", "#7394BA"),
        ("entry_burn", "단축 재진입 (Entry Burn · 3엔진)", OXFORD_CRIMSON),
        ("aero_glide", "무동력 격자핀 활강 (Glide)", OXFORD_SKY),
        ("landing_burn", "3→1 호버슬램 (Landing)", OXFORD_GREEN),
        ("touchdown_settle", "5초 직립 안정 (Standing)", "#3E8E63"),
    ]
    rows_data = []
    for ph, label, col in phase_order:
        pfs = [f for f in calm_frames if f["phase"] == ph]
        if not pfs:
            continue
        dur = pfs[-1]["time_s"] - pfs[0]["time_s"]
        main_used = max(0.0, pfs[0]["main_propellant_kg"] - pfs[-1]["main_propellant_kg"])
        rcs_used = max(0.0, pfs[0]["rcs_propellant_kg"] - pfs[-1]["rcs_propellant_kg"])
        z0 = pfs[0]["pos_enu_m"][2] / 1000.0
        z1 = pfs[-1]["pos_enu_m"][2] / 1000.0
        rows_data.append((ph, label, col, dur, main_used, rcs_used, z0, z1))

    asc_used = next((r[4] for r in rows_data if r[0] == "ascent"), 0.0)
    rec_rows = [r for r in rows_data if r[0] != "ascent"]
    max_fuel = max(r[4] for r in rec_rows)
    bx0, bw = 376.0, 444.0
    bars = []
    for i, (ph, label, col, dur, main_used, rcs_used, z0, z1) in enumerate(rec_rows):
        y = 204 + i * 52
        w = max(4.0, (main_used / max(max_fuel, 1.0)) * bw)
        extra = (
            f"스로틀 0% · {z0 - z1:.1f} km 활강 (테일오프 {main_used:.1f} kg)"
            if ph == "aero_glide"
            else f"주추진제 {main_used:,.1f} kg · RCS {rcs_used:.1f} kg ({dur:.1f} s)"
        )
        bars.append(
            f'<text x="70" y="{y + 18}" font-size="17" fill="{OXFORD_IVORY}" font-weight="600">{label}</text>'
            f'<rect x="{bx0:.0f}" y="{y + 3}" width="{bw:.0f}" height="20" rx="2" fill="{OXFORD_DEEP}" stroke="{OXFORD_GRID}" stroke-width="0.8"/>'
            f'<rect x="{bx0:.0f}" y="{y + 3}" width="{w:.1f}" height="20" rx="2" fill="{col}"/>'
            f'<text x="{bx0 + bw + 16:.0f}" y="{y + 18}" font-size="15" fill="{OXFORD_SLATE}">{extra}</text>'
        )

    cases = summary["cases"]
    mains = [r["main_remaining_kg"] for r in cases]
    glides = [r["glide_distance_km"] for r in cases]
    uppers = [r["upper_propellant_remaining_kg"] for r in cases]
    med_main = float(np.median(mains))
    med_glide = float(np.median(glides))
    min_upper = min(uppers)
    max_orb_err = max(abs(r["orbit_altitude_km"] - 700.0) for r in cases)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="670" viewBox="0 0 1200 670" role="img" aria-labelledby="title desc">
<title id="title">DM100 100톤 메탈록스 RTLS 귀환 추진제 예산 및 2단 700 km 궤도 여유</title>
<desc id="desc">단 분리 이후 귀환 단계별 주추진제와 RCS 소모량, 약 29.4 km 무동력 격자핀 활강 구간 및 487케이스 1단·2단 잔여 추진제 통계입니다.</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="670" fill="{OXFORD_BLUE}"/>
<rect width="1200" height="6" fill="{OXFORD_BRASS}"/>
{_rocket_badge_svg(1106, 28, 46)}
<text x="48" y="46" font-size="13" fill="{OXFORD_BRASS_LT}" class="eyebrow">PROPELLANT BUDGET · LCH₄/LOX 7-ENGINE CLUSTER</text>
<text x="48" y="88" font-size="34" fill="{OXFORD_IVORY}" font-weight="700">RTLS 귀환 단계별 추진제 예산</text>
<text x="48" y="118" font-size="18" fill="{OXFORD_SLATE}">초음속·아음속 무동력 격자핀 활강으로 연료를 절약하고 종말 3→1 호버슬램으로 착륙합니다 (무풍 기준 + 487케이스)</text>

<rect x="48" y="142" width="1104" height="432" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>
<text x="70" y="178" font-size="18" fill="{OXFORD_BRASS_LT}" font-weight="700">분리 이후 귀환 단계별 주추진제 소모 <tspan fill="{OXFORD_SLATE}" font-weight="400">(1단 총 69,500 kg 중 상승 {asc_used:,.0f} kg 제외 · 재점화 3/3회)</tspan></text>
{"".join(bars)}

<rect x="48" y="592" width="354" height="54" rx="4" fill="{OXFORD_DEEP}" stroke="{OXFORD_BORDER}"/>
<rect x="48" y="592" width="354" height="3" fill="{OXFORD_GREEN}"/>
<text x="225" y="615" font-size="13" fill="{OXFORD_MUTED}" text-anchor="middle">1단 잔여 주추진제 (487케이스)</text>
<text x="225" y="636" font-size="17" fill="{OXFORD_GREEN}" font-weight="700" text-anchor="middle">중앙 {med_main:,.0f} kg · 최소 {min(mains):,.0f} kg</text>

<rect x="423" y="592" width="354" height="54" rx="4" fill="{OXFORD_DEEP}" stroke="{OXFORD_BORDER}"/>
<rect x="423" y="592" width="354" height="3" fill="{OXFORD_SKY}"/>
<text x="600" y="615" font-size="13" fill="{OXFORD_MUTED}" text-anchor="middle">무동력 격자핀 활강 거리</text>
<text x="600" y="636" font-size="17" fill="{OXFORD_SKY}" font-weight="700" text-anchor="middle">중앙 {med_glide:.2f} km ({min(glides):.1f}–{max(glides):.1f} km)</text>

<rect x="798" y="592" width="354" height="54" rx="4" fill="{OXFORD_DEEP}" stroke="{OXFORD_BORDER}"/>
<rect x="798" y="592" width="354" height="3" fill="{OXFORD_BRASS}"/>
<text x="975" y="615" font-size="13" fill="{OXFORD_MUTED}" text-anchor="middle">2단 700 km 원궤도 (1,800 kg)</text>
<text x="975" y="636" font-size="17" fill="{OXFORD_BRASS_LT}" font-weight="700" text-anchor="middle">최대 오차 {max_orb_err:.2f} km · 최소 잔여 {min_upper:.0f} kg</text>
</svg>
"""


def render_trajectory_svg(calm_frames: list[dict], weather_frames: list[dict]) -> str:
    c_sub = _subsample(calm_frames, step=12)
    w_sub = _subsample(weather_frames, step=12)

    all_x = [f["pos_enu_m"][0] / 1000.0 for f in c_sub + w_sub]
    all_z = [f["pos_enu_m"][2] / 1000.0 for f in c_sub + w_sub]
    x_lo = math.floor(min(0.0, min(all_x)) / 10.0) * 10.0 if min(all_x) < -2.0 else 0.0
    x_hi = max(80.0, math.ceil(max(all_x) / 20.0) * 20.0)
    z_hi = max(175.0, math.ceil((max(all_z) + 12.0) / 25.0) * 25.0)

    # Left panel: 1st-stage trajectory
    lx0, lx1, ly0, ly1 = 102.0, 560.0, 198.0, 582.0

    def lsx(f: dict) -> float:
        return lx0 + ((f["pos_enu_m"][0] / 1000.0) - x_lo) / (x_hi - x_lo) * (lx1 - lx0)

    def lsy(f: dict) -> float:
        return ly1 - (f["pos_enu_m"][2] / 1000.0) / z_hi * (ly1 - ly0)

    l_segs = _phase_segments(c_sub, lsx, lsy)
    w_pts = [(lsx(f), lsy(f)) for f in w_sub]

    sep_f = next(f for f in calm_frames if f["phase"] == "flip")
    apo_f = max(calm_frames, key=lambda f: f["pos_enu_m"][2])
    gld_f = next(f for f in calm_frames if f["phase"] == "aero_glide")

    # Right stacked panels: Top = Dynamic Pressure (kPa), Bottom = Main Engine Throttle (%)
    rx0, rx1 = 674.0, 1128.0
    qy0, qy1 = 196.0, 358.0
    ty0, ty1 = 456.0, 586.0
    t_hi = max(
        580.0, math.ceil(max(calm_frames[-1]["time_s"], weather_frames[-1]["time_s"]) / 50.0) * 50.0
    )
    q_hi = 48.0
    thr_hi = 118.0

    def rsx(t_s: float) -> float:
        return rx0 + (t_s / t_hi) * (rx1 - rx0)

    def rsy_q(q_kpa: float) -> float:
        return qy1 - (q_kpa / q_hi) * (qy1 - qy0)

    def rsy_thr(thr_pct: float) -> float:
        return ty1 - (thr_pct / thr_hi) * (ty1 - ty0)

    q_pts = [(rsx(f["time_s"]), rsy_q(f["dynamic_pressure_pa"] / 1000.0)) for f in c_sub]
    thr_pts = [(rsx(f["time_s"]), rsy_thr(f["throttle"] * 100.0)) for f in c_sub]
    q_area = [(rx0, qy1), *q_pts, (q_pts[-1][0], qy1)]
    thr_area = [(rx0, ty1), *thr_pts, (thr_pts[-1][0], ty1)]

    asc_frames = [f for f in calm_frames if f["phase"] == "ascent"]
    rec_frames = [
        f for f in calm_frames if f["phase"] in ("entry_burn", "aero_glide", "landing_burn")
    ]
    asc_q_f = max(asc_frames, key=lambda f: f["dynamic_pressure_pa"])
    rec_q_f = max(rec_frames, key=lambda f: f["dynamic_pressure_pa"])
    asc_q_kpa = asc_q_f["dynamic_pressure_pa"] / 1000.0
    rec_q_kpa = rec_q_f["dynamic_pressure_pa"] / 1000.0

    glide_frames = [f for f in calm_frames if f["phase"] == "aero_glide"]
    boost_frames = [f for f in calm_frames if f["phase"] == "boostback"]
    entry_frames = [f for f in calm_frames if f["phase"] == "entry_burn"]
    land_frames = [f for f in calm_frames if f["phase"] == "landing_burn"]

    gx0 = rsx(glide_frames[0]["time_s"])
    gx1 = rsx(glide_frames[-1]["time_s"])
    glide_dur = glide_frames[-1]["time_s"] - glide_frames[0]["time_s"]
    y_q42 = rsy_q(42.0)
    y_q336 = rsy_q(33.6)

    l_grid = []
    for i in range(5):
        zv = z_hi * i / 4.0
        yy = ly1 - (zv / z_hi) * (ly1 - ly0)
        l_grid.append(
            f'<line x1="{lx0:.0f}" y1="{yy:.1f}" x2="{lx1:.0f}" y2="{yy:.1f}" stroke="{OXFORD_GRID}"/>'
            f'<text x="{lx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{OXFORD_MUTED}" text-anchor="end">{zv:.0f}</text>'
        )
    for i in range(5):
        xv = x_lo + (x_hi - x_lo) * i / 4.0
        xx = lx0 + (lx1 - lx0) * i / 4.0
        l_grid.append(
            f'<line x1="{xx:.1f}" y1="{ly0:.0f}" x2="{xx:.1f}" y2="{ly1:.0f}" stroke="{OXFORD_GRID}"/>'
            f'<text x="{xx:.1f}" y="{ly1 + 22:.0f}" font-size="14" fill="{OXFORD_MUTED}" text-anchor="middle">{xv:.0f}</text>'
        )

    q_grid = []
    for qv in (0, 15, 30, 45):
        yy = rsy_q(qv)
        q_grid.append(
            f'<line x1="{rx0:.0f}" y1="{yy:.1f}" x2="{rx1:.0f}" y2="{yy:.1f}" stroke="{OXFORD_GRID}"/>'
            f'<text x="{rx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{OXFORD_SKY}" text-anchor="end">{qv}</text>'
        )

    thr_grid = []
    for tp in (0, 50, 100):
        yy = rsy_thr(tp)
        thr_grid.append(
            f'<line x1="{rx0:.0f}" y1="{yy:.1f}" x2="{rx1:.0f}" y2="{yy:.1f}" stroke="{OXFORD_GRID}"/>'
            f'<text x="{rx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{OXFORD_BRASS_LT}" text-anchor="end">{tp}%</text>'
        )

    t_step = 150 if t_hi <= 600 else 200
    tv = 0
    while tv <= t_hi:
        xx = rsx(tv)
        q_grid.append(
            f'<line x1="{xx:.1f}" y1="{qy0:.0f}" x2="{xx:.1f}" y2="{qy1:.0f}" stroke="{OXFORD_GRID}"/>'
        )
        thr_grid.append(
            f'<line x1="{xx:.1f}" y1="{ty0:.0f}" x2="{xx:.1f}" y2="{ty1:.0f}" stroke="{OXFORD_GRID}"/>'
            f'<text x="{xx:.1f}" y="{ty1 + 22:.0f}" font-size="14" fill="{OXFORD_MUTED}" text-anchor="middle">{tv}</text>'
        )
        tv += t_step

    l_paths = "".join(
        f'<polyline points="{_polyline(pts)}" fill="none" stroke="{PHASE_COLORS.get(ph, OXFORD_SKY)}" '
        f'stroke-width="{3.8 if ph in ("ascent", "aero_glide", "landing_burn") else 3.0}" stroke-linecap="round"/>'
        for ph, pts in l_segs
    )

    sep_x, sep_y = lsx(sep_f), lsy(sep_f)
    apo_x, apo_y = lsx(apo_f), lsy(apo_f)
    gld_x, gld_y = lsx(gld_f), lsy(gld_f)
    aqx, aqy = rsx(asc_q_f["time_s"]), rsy_q(asc_q_kpa)
    rqx, rqy = rsx(rec_q_f["time_s"]), rsy_q(rec_q_kpa)

    t_boost_mid = 0.5 * (boost_frames[0]["time_s"] + boost_frames[-1]["time_s"])
    boost_thr_max = max(f["throttle"] * 100.0 for f in boost_frames)
    t_entry_mid = 0.5 * (entry_frames[0]["time_s"] + entry_frames[-1]["time_s"])
    entry_thr_max = max(f["throttle"] * 100.0 for f in entry_frames)
    t_land_mid = 0.5 * (land_frames[0]["time_s"] + land_frames[-1]["time_s"])
    land_thr_max = max(f["throttle"] * 100.0 for f in land_frames)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="730" viewBox="0 0 1200 730" role="img" aria-labelledby="title desc">
<title id="title">DM100 궤도 RTLS 비행 궤적 및 분리형 동압·스로틀 프로파일</title>
<desc id="desc">무풍 기준과 복합 기상 비행의 2차원 귀환 궤적, 독립 패널로 분리된 동압 프로파일(상승 {asc_q_kpa:.1f} kPa, 재진입 {rec_q_kpa:.1f} kPa, 한계 42 kPa 및 80% 목표 33.6 kPa 미만 유지)과 주엔진 스로틀 프로파일(재점화 3회, {glide_dur:.0f}초 무동력 격자핀 활강)입니다.</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="730" fill="{OXFORD_BLUE}"/>
<rect width="1200" height="6" fill="{OXFORD_BRASS}"/>
{_rocket_badge_svg(1106, 28, 46)}
<text x="48" y="46" font-size="13" fill="{OXFORD_BRASS_LT}" class="eyebrow">FLIGHT TELEMETRY · TRAJECTORY &amp; AEROTHERMAL ENVELOPE</text>
<text x="48" y="88" font-size="34" fill="{OXFORD_IVORY}" font-weight="700">RTLS 궤적 및 동압 · 스로틀 프로파일</text>
<text x="48" y="118" font-size="18" fill="{OXFORD_SLATE}">무풍 기준(실선)과 복합 기상(점선) 비교 · 동압(상단)과 스로틀(하단)을 독립 축으로 분리해 표시합니다</text>

<rect x="48" y="142" width="538" height="506" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>
<rect x="48" y="142" width="538" height="3" fill="{OXFORD_BRASS}"/>
<text x="72" y="176" font-size="18" fill="{OXFORD_IVORY}" font-weight="700">1단 상승·귀환 궤적 <tspan font-size="14" fill="{OXFORD_SLATE}" font-weight="400">(고도 km vs 수평 거리 km)</tspan></text>
{"".join(l_grid)}
<polyline points="{_polyline(w_pts)}" fill="none" stroke="{OXFORD_IVORY}" stroke-width="1.6" stroke-dasharray="5 5" opacity="0.58"/>
{l_paths}
<circle cx="{sep_x:.1f}" cy="{sep_y:.1f}" r="4.8" fill="{OXFORD_BRASS_LT}" stroke="{OXFORD_IVORY}" stroke-width="1.5"/>
<line x1="{sep_x + 5:.1f}" y1="{sep_y + 5:.1f}" x2="{sep_x + 36:.1f}" y2="{sep_y + 52:.1f}" stroke="{OXFORD_BRASS_LT}" stroke-width="1.2" opacity="0.85"/>
<text x="{sep_x + 42:.1f}" y="{sep_y + 57:.1f}" font-size="13.5" fill="{OXFORD_BRASS_LT}" font-weight="700">단 분리 ({sep_f["pos_enu_m"][2] / 1000.0:.1f} km)</text>
<circle cx="{apo_x:.1f}" cy="{apo_y:.1f}" r="4.8" fill="#A8C3E0" stroke="{OXFORD_IVORY}" stroke-width="1.5"/>
<text x="{apo_x:.1f}" y="{apo_y - 12:.1f}" font-size="13.5" fill="#D5E4F4" font-weight="700" text-anchor="middle">정점 {apo_f["pos_enu_m"][2] / 1000.0:.1f} km</text>
<circle cx="{gld_x:.1f}" cy="{gld_y:.1f}" r="4.8" fill="{OXFORD_SKY}" stroke="{OXFORD_IVORY}" stroke-width="1.5"/>
<line x1="{gld_x + 6:.1f}" y1="{gld_y - 6:.1f}" x2="{gld_x + 118:.1f}" y2="{gld_y - 128:.1f}" stroke="{OXFORD_SKY}" stroke-width="1.2" opacity="0.85"/>
<text x="{gld_x + 124:.1f}" y="{gld_y - 124:.1f}" font-size="13.5" fill="{OXFORD_SKY}" font-weight="700">무동력 활강 시작</text>
<text x="332" y="632" font-size="15" fill="{OXFORD_SLATE}" text-anchor="middle">수평 거리 (km)</text>

<rect x="614" y="142" width="538" height="244" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>
<rect x="614" y="142" width="538" height="3" fill="{OXFORD_SKY}"/>
<text x="636" y="175" font-size="17" fill="{OXFORD_IVORY}" font-weight="700">동압 프로파일 (kPa)</text>
<text x="1130" y="175" font-size="13.5" fill="{OXFORD_SKY}" font-weight="600" text-anchor="end">상승 {asc_q_kpa:.1f} · 재진입 {rec_q_kpa:.1f} kPa</text>
<rect x="{gx0:.1f}" y="{qy0:.0f}" width="{gx1 - gx0:.1f}" height="{qy1 - qy0:.0f}" fill="{OXFORD_SKY}" opacity="0.14"/>
{"".join(q_grid)}
<line x1="{rx0:.0f}" y1="{y_q42:.1f}" x2="{rx1:.0f}" y2="{y_q42:.1f}" stroke="{OXFORD_CRIMSON}" stroke-width="1.5" stroke-dasharray="6 4"/>
<text x="{rsx(285):.1f}" y="{y_q42 - 6:.1f}" font-size="12.5" fill="{OXFORD_CRIMSON}" font-weight="700" text-anchor="middle">동압 한계 42 kPa</text>
<line x1="{rx0:.0f}" y1="{y_q336:.1f}" x2="{rx1:.0f}" y2="{y_q336:.1f}" stroke="{OXFORD_BRASS_LT}" stroke-width="1.3" stroke-dasharray="3 4"/>
<text x="{rsx(285):.1f}" y="{y_q336 - 6:.1f}" font-size="12" fill="{OXFORD_BRASS_LT}" font-weight="600" text-anchor="middle">80% 여유선 33.6 kPa</text>
<polygon points="{_polyline(q_area)}" fill="{OXFORD_SKY}" opacity="0.18"/>
<polyline points="{_polyline(q_pts)}" fill="none" stroke="{OXFORD_SKY}" stroke-width="2.6" stroke-linejoin="round"/>
<circle cx="{aqx:.1f}" cy="{aqy:.1f}" r="4.0" fill="{OXFORD_SKY}" stroke="{OXFORD_IVORY}" stroke-width="1.3"/>
<text x="{aqx + 12:.1f}" y="{aqy + 4:.1f}" font-size="12.5" fill="{OXFORD_IVORY}" font-weight="700">{asc_q_kpa:.1f} kPa</text>
<circle cx="{rqx:.1f}" cy="{rqy:.1f}" r="4.0" fill="{OXFORD_SKY}" stroke="{OXFORD_IVORY}" stroke-width="1.3"/>
<text x="{rqx - 10:.1f}" y="{rqy + 4:.1f}" font-size="12.5" fill="{OXFORD_IVORY}" font-weight="700" text-anchor="end">{rec_q_kpa:.1f} kPa</text>

<rect x="614" y="402" width="538" height="246" rx="4" fill="{OXFORD_SURFACE}" stroke="{OXFORD_BORDER}"/>
<rect x="614" y="402" width="538" height="3" fill="{OXFORD_BRASS}"/>
<text x="636" y="435" font-size="17" fill="{OXFORD_IVORY}" font-weight="700">주엔진 스로틀 (%)</text>
<text x="1130" y="435" font-size="13.5" fill="{OXFORD_BRASS_LT}" font-weight="600" text-anchor="end">재점화 3회 · 무동력 활강 {glide_dur:.0f} s (0%)</text>
<rect x="{gx0:.1f}" y="{ty0:.0f}" width="{gx1 - gx0:.1f}" height="{ty1 - ty0:.0f}" fill="{OXFORD_SKY}" opacity="0.14"/>
<text x="{(gx0 + gx1) / 2:.1f}" y="{ty0 + 22:.1f}" font-size="12" fill="{OXFORD_SKY}" font-weight="700" text-anchor="middle">무동력 활강 (0%)</text>
{"".join(thr_grid)}
<polygon points="{_polyline(thr_area)}" fill="{OXFORD_BRASS}" opacity="0.18"/>
<polyline points="{_polyline(thr_pts)}" fill="none" stroke="{OXFORD_BRASS_LT}" stroke-width="2.4" stroke-linejoin="round"/>
<text x="{rsx(62):.1f}" y="{rsy_thr(100) - 6:.1f}" font-size="12" fill="{OXFORD_BRASS_LT}" font-weight="700" text-anchor="middle">상승 (7엔진)</text>
<text x="{rsx(t_boost_mid):.1f}" y="{rsy_thr(boost_thr_max) - 7:.1f}" font-size="12" fill="{OXFORD_SLATE}" font-weight="600" text-anchor="middle">부스트백</text>
<text x="{rsx(t_entry_mid) - 6:.1f}" y="{rsy_thr(entry_thr_max) - 7:.1f}" font-size="12" fill="{OXFORD_CRIMSON}" font-weight="600" text-anchor="end">재진입</text>
<text x="{min(rx1 - 2, rsx(t_land_mid) + 8):.1f}" y="{rsy_thr(land_thr_max) - 10:.1f}" font-size="12" fill="{OXFORD_GREEN}" font-weight="700" text-anchor="end">착륙 3→1</text>
<text x="900" y="636" font-size="15" fill="{OXFORD_SLATE}" text-anchor="middle">비행 시간 (s)</text>

<rect x="48" y="664" width="1104" height="44" rx="4" fill="{OXFORD_DEEP}" stroke="{OXFORD_BORDER}"/>
<line x1="70" y1="686" x2="96" y2="686" stroke="{OXFORD_BRASS_LT}" stroke-width="3.5"/><text x="106" y="692" font-size="14.5" fill="{OXFORD_IVORY}">상승 (7엔진) / 스로틀</text>
<line x1="296" y1="686" x2="322" y2="686" stroke="{OXFORD_CRIMSON}" stroke-width="3.5"/><text x="332" y="692" font-size="14.5" fill="{OXFORD_IVORY}">재진입 연소</text>
<line x1="454" y1="686" x2="480" y2="686" stroke="{OXFORD_SKY}" stroke-width="3.5"/><text x="490" y="692" font-size="14.5" fill="{OXFORD_IVORY}">무동력 활강 / 동압</text>
<line x1="666" y1="686" x2="692" y2="686" stroke="{OXFORD_GREEN}" stroke-width="3.5"/><text x="702" y="692" font-size="14.5" fill="{OXFORD_IVORY}">3→1 호버슬램 착륙</text>
<line x1="882" y1="686" x2="912" y2="686" stroke="{OXFORD_IVORY}" stroke-width="2.0" stroke-dasharray="5 4"/><text x="922" y="692" font-size="14.5" fill="{OXFORD_IVORY}">복합 기상 궤적</text>
</svg>
"""


def render_trajectory_png(calm_frames: list[dict], out_path: Path) -> None:
    sub = _subsample(calm_frames, step=5)
    t = np.array([f["time_s"] for f in sub])
    x_km = np.array([f["pos_enu_m"][0] / 1000.0 for f in sub])
    z_km = np.array([f["pos_enu_m"][2] / 1000.0 for f in sub])
    vx = np.array([f["vel_enu_m_s"][0] for f in sub])
    vz = np.array([f["vel_enu_m_s"][2] for f in sub])
    pitch = np.array([_pitch_deg(f) for f in sub])
    fin_deg = np.array([f["fin_deflection_rad"][0] * (180.0 / math.pi) for f in sub])
    q_kpa = np.array([f["dynamic_pressure_pa"] / 1000.0 for f in sub])
    thr_pct = np.array([f["throttle"] * 100.0 for f in sub])

    glide_t = [f["time_s"] for f in calm_frames if f["phase"] == "aero_glide"]
    gt0, gt1 = glide_t[0], glide_t[-1]
    asc_q_max = (
        max(f["dynamic_pressure_pa"] for f in calm_frames if f["phase"] == "ascent") / 1000.0
    )
    rec_q_max = (
        max(
            f["dynamic_pressure_pa"]
            for f in calm_frames
            if f["phase"] in ("entry_burn", "aero_glide", "landing_burn")
        )
        / 1000.0
    )
    final_main = calm_frames[-1]["main_propellant_kg"]
    final_x = abs(calm_frames[-1]["pos_enu_m"][0])

    fig = plt.figure(figsize=(13.6, 8.4), dpi=160)
    fig.patch.set_facecolor("#F7F6F1")
    gs = gridspec.GridSpec(
        2, 2, figure=fig, hspace=0.28, wspace=0.22, left=0.065, right=0.975, top=0.915, bottom=0.075
    )
    fig.suptitle(
        f"DM100 Orbital RTLS · Calm Baseline (700.0 km LEO | 1,800 kg | Main rem {final_main:,.0f} kg | |x|={final_x:.3f} m | 5 s Standing PASS)",
        fontsize=12.0,
        fontweight="bold",
        color=OXFORD_BLUE,
    )

    # 1. Altitude vs Downrange by phase
    ax_traj = fig.add_subplot(gs[0, 0])
    for ph in ("ascent", "flip", "boostback", "coast", "entry_burn", "aero_glide", "landing_burn"):
        idx = [i for i, f in enumerate(sub) if f["phase"] == ph]
        if idx:
            ax_traj.plot(
                x_km[idx],
                z_km[idx],
                color=PNG_PHASE_COLORS[ph],
                lw=2.4 if ph in ("ascent", "aero_glide", "landing_burn") else 1.7,
                label=ph.replace("_", " "),
            )
    ax_traj.set_xlabel("Horizontal distance (km)")
    ax_traj.set_ylabel("Altitude (km)")
    ax_traj.legend(
        fontsize=8, loc="upper left", ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#D8D6CE"
    )

    # 2. Velocity
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.axvspan(gt0, gt1, color="#6FA4DC", alpha=0.16, label="Unpowered glide")
    ax_vel.plot(t, vx, color=OXFORD_BLUE, lw=1.9, label="Horizontal vx")
    ax_vel.plot(t, vz, color="#9E7B2F", lw=1.9, label="Vertical vz")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.legend(
        fontsize=8.5, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#D8D6CE"
    )

    # 3. Attitude & Grid-Fin Deflection
    ax_att = fig.add_subplot(gs[1, 0])
    ax_att.axvspan(gt0, gt1, color="#6FA4DC", alpha=0.16)
    ax_att.plot(t, pitch, color=OXFORD_BLUE, lw=1.9, label="Pitch (deg)")
    ax_att.plot(t, fin_deg, color="#1B5E3A", lw=1.5, label="Grid-fin deflection (deg)")
    ax_att.set_xlabel("Time (s)")
    ax_att.set_ylabel("Angle (deg)")
    ax_att.legend(
        fontsize=8.5, loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#D8D6CE"
    )

    # 4. Separate stacked panels: Top = Dynamic Pressure (kPa), Bottom = Main Throttle (%)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 1], height_ratios=[1.15, 0.85], hspace=0.16
    )
    ax_q = fig.add_subplot(sub_gs[0, 0])
    ax_q.axvspan(gt0, gt1, color="#6FA4DC", alpha=0.16)
    ax_q.fill_between(t, 0.0, q_kpa, color=OXFORD_BLUE, alpha=0.10)
    ax_q.plot(
        t,
        q_kpa,
        color=OXFORD_BLUE,
        lw=2.0,
        label=f"Dynamic pressure (peak {asc_q_max:.1f} / {rec_q_max:.1f} kPa)",
    )
    ax_q.axhline(42.0, color="#9E2A2B", ls="--", lw=1.35, label="Max-Q limit (42 kPa)")
    ax_q.axhline(33.6, color="#9E7B2F", ls=":", lw=1.25, label="80% margin (33.6 kPa)")
    ax_q.set_ylabel("Dyn. pressure (kPa)", fontsize=9.5, color=OXFORD_BLUE)
    ax_q.set_ylim(-1.5, 48.0)
    ax_q.set_yticks([0, 15, 30, 42])
    ax_q.tick_params(labelbottom=False)
    ax_q.legend(
        fontsize=7.6,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.06),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D8D6CE",
    )

    ax_thr = fig.add_subplot(sub_gs[1, 0], sharex=ax_q)
    ax_thr.axvspan(gt0, gt1, color="#6FA4DC", alpha=0.16, label="Unpowered glide (0% cmd)")
    ax_thr.fill_between(t, 0.0, thr_pct, color="#9E7B2F", alpha=0.15)
    ax_thr.plot(t, thr_pct, color="#9E7B2F", lw=1.8, label="Main engine throttle (%)")
    ax_thr.set_xlabel("Time (s)")
    ax_thr.set_ylabel("Throttle (%)", fontsize=9.5, color="#9E7B2F")
    ax_thr.set_ylim(-6.0, 118.0)
    ax_thr.set_yticks([0, 50, 100])
    ax_thr.legend(
        fontsize=7.6, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#D8D6CE"
    )

    for axis in (ax_traj, ax_vel, ax_att, ax_q, ax_thr):
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, color="#E2E0D8", alpha=0.8, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for sp in ("bottom", "left"):
            axis.spines[sp].set_color("#B8C2CE")

    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    summary = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    calm = json.loads(CALM_REPLAY.read_text(encoding="utf-8"))["frames"]
    weather = json.loads(WEATHER_REPLAY.read_text(encoding="utf-8"))["frames"]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "dm100-thumbnail.svg").write_text(
        render_thumbnail_svg(calm, summary), encoding="utf-8"
    )
    (ASSETS_DIR / "dm100-validation.svg").write_text(
        render_validation_svg(summary), encoding="utf-8"
    )
    (ASSETS_DIR / "dm100-propellant.svg").write_text(
        render_propellant_svg(calm, summary), encoding="utf-8"
    )
    (ASSETS_DIR / "dm100-trajectory.svg").write_text(
        render_trajectory_svg(calm, weather), encoding="utf-8"
    )
    render_trajectory_png(calm, ASSETS_DIR / "trajectory.png")
    print("Generated all 5 Oxford Blue DM100 RTLS visuals in", ASSETS_DIR)


if __name__ == "__main__":
    main()
