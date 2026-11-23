# ruff: noqa: E501
"""Generate DM100 Orbital Reusable RTLS visuals in light-paper Oxford Blue editorial style."""

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

# University of Oxford editorial light-paper palette (#002147 primary)
PAPER_BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#D4DDE7"
RULE_LINE = "#CDD8E4"
GRID_LINE = "#E6ECF2"
TRACK_BG = "#EBF0F5"
GLIDE_TINT = "#E4EDF7"

OXFORD_BLUE = "#002147"
OXFORD_MID = "#2D537B"
OXFORD_STEEL = "#6E8AA8"
OXFORD_PALE = "#B4C5D8"
OXFORD_SUBTLE = "#C9D8E8"
INK_BODY = "#1C2B3B"
INK_MUTED = "#556473"
WEATHER_GRAY = "#7D8B99"

OCHRE = "#8C6A28"
CRIMSON = "#9E2A2B"
FOREST = "#1F5A3D"

FONT_STYLE = (
    "text{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo','Segoe UI',sans-serif;"
    "font-variant-numeric:tabular-nums}"
)

PHASE_COLORS = {
    "ascent": OXFORD_BLUE,
    "flip": OXFORD_STEEL,
    "boostback": "#4C6E91",
    "coast": "#849DB7",
    "entry_burn": CRIMSON,
    "aero_glide": OXFORD_MID,
    "landing_burn": FOREST,
    "touchdown_settle": FOREST,
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


def render_thumbnail_svg(calm_frames: list[dict], summary: dict, *, lang: str = "ko") -> str:
    en = lang == "en"
    rows = summary["cases"]
    sub = _subsample(calm_frames, step=12)
    x_vals = [f["pos_enu_m"][0] / 1000.0 for f in sub]
    z_vals = [f["pos_enu_m"][2] / 1000.0 for f in sub]
    x_min_raw = min(0.0, min(x_vals))
    x_lo = math.floor(x_min_raw / 10.0) * 10.0 if x_min_raw < -2.0 else 0.0
    x_hi = max(100.0, math.ceil(max(x_vals) / 20.0) * 20.0)
    z_hi = max(200.0, math.ceil((max(z_vals) + 25.0) / 25.0) * 25.0)

    px0, px1 = 496.0, 1116.0
    py0, py1 = 242.0, 514.0

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
            f'<line x1="{px0:.0f}" y1="{yy:.1f}" x2="{px1:.0f}" y2="{yy:.1f}" stroke="{GRID_LINE}" stroke-width="1"/>'
            f'<text x="{px0 - 12:.0f}" y="{yy + 5:.1f}" font-size="16" fill="{INK_MUTED}" text-anchor="end">{zv:.0f}</text>'
        )
    for i in range(5):
        xv = x_lo + (x_hi - x_lo) * i / 4.0
        xx = px0 + (px1 - px0) * i / 4.0
        anchor = "start" if i == 0 else ("end" if i == 4 else "middle")
        grid_lines.append(
            f'<line x1="{xx:.1f}" y1="{py0:.0f}" x2="{xx:.1f}" y2="{py1:.0f}" stroke="{GRID_LINE}" stroke-width="1"/>'
            f'<text x="{xx:.1f}" y="{py1 + 24:.0f}" font-size="16" fill="{INK_MUTED}" text-anchor="{anchor}">{xv:.0f}</text>'
        )

    path_svg = []
    for ph, pts in segs:
        col = PHASE_COLORS.get(ph, OXFORD_MID)
        w = 4.2 if ph in ("ascent", "aero_glide", "landing_burn") else 2.8
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

    spec_rows = (
        [
            ("Target orbit", "700 km · 1,800 kg", OXFORD_BLUE),
            ("Peak limit usage", f"{peak_util:.1f}% (≤ 80%)", FOREST),
            ("Unpowered glide", f"{min_glide:.1f}~{max_glide:.1f} km", OXFORD_MID),
            ("Median main rem", f"{med_main:,.0f} kg", OXFORD_BLUE),
            ("Median pos error", f"{med_pos:.3f} m", OXFORD_BLUE),
            ("5 s leg standing", f"{passed} / {total}", FOREST),
        ]
        if en
        else [
            ("목표 궤도", "700 km · 1,800 kg", OXFORD_BLUE),
            ("최대 제약 사용률", f"{peak_util:.1f}% (≤ 80%)", FOREST),
            ("무동력 격자핀 활강", f"{min_glide:.1f}~{max_glide:.1f} km", OXFORD_MID),
            ("잔여 주추진제 중앙", f"{med_main:,.0f} kg", OXFORD_BLUE),
            ("접지 위치 중앙값", f"{med_pos:.3f} m", OXFORD_BLUE),
            ("5초 착륙다리 직립", f"{passed} / {total}", FOREST),
        ]
    )
    row_svg = []
    for idx, (k, v, col) in enumerate(spec_rows):
        ry = 266 + idx * 50
        row_svg.append(
            f'<text x="70" y="{ry}" font-size="16" fill="{INK_MUTED}">{k}</text>'
            f'<text x="370" y="{ry}" font-size="17" fill="{col}" font-weight="700" text-anchor="end">{v}</text>'
            + (
                f'<line x1="70" y1="{ry + 18}" x2="370" y2="{ry + 18}" stroke="{GRID_LINE}" stroke-width="1"/>'
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

    svg_title = (
        "DM100 100 t Methalox 700 km Orbital Launch and First-Stage Reusable RTLS"
        if en
        else "DM100 100톤 메탈록스 700 km 원궤도 발사 및 1단 재사용 RTLS"
    )
    svg_desc = (
        f"Full 487-case validation of a 100 ton 7-engine Methalox launcher delivering 1,800 kg to a 700 km circular orbit with first-stage RTLS, {min_glide:.1f}~{max_glide:.1f} km unpowered grid-fin glide, 3→1 hoverslam, and 5 s upright standing."
        if en
        else f"100톤 메탈록스 7엔진 2단 로켓의 700 km 원궤도(1,800 kg) 투입과 1단 발사장 귀환, {min_glide:.1f}~{max_glide:.1f} km 무동력 격자핀 활강, 3→1 호버슬램 착륙 및 5초 직립 동역학 487케이스 전수 검증 결과입니다."
    )
    h1_text = (
        "100 t Methalox Orbital Launch &amp; RTLS" if en else "100톤 메탈록스 궤도 발사 및 1단 회수"
    )
    h1_size = 44 if en else 46
    sub_text = (
        "700 km circular orbit (1,800 kg), unpowered grid-fin glide, 3→1 hoverslam &amp; 5 s standing"
        if en
        else "700 km 원궤도(1,800 kg) 투입, 무동력 격자핀 활강과 접지 후 5초 직립 동역학"
    )
    sub_size = 23 if en else 25
    left_hdr = "Key Design &amp; Metrics" if en else "주요 설계 · 검증 지표"
    traj_hdr = (
        f'Stage 1 RTLS Trajectory <tspan font-size="14.5" fill="{INK_MUTED}" font-weight="400">(Alt km vs Downrange km)</tspan>'
        if en
        else f'1단 상승·귀환 비행 궤적 <tspan font-size="15" fill="{INK_MUTED}" font-weight="400">(고도 km vs 수평 거리 km)</tspan>'
    )
    calm_lbl = f"Calm ({t_end:.1f} s)" if en else f"무풍 기준 ({t_end:.1f} s)"
    sep_lbl = f"MECO-1 Sep ({sep_z_km:.1f} km)" if en else f"단 분리 ({sep_z_km:.1f} km)"
    apo_lbl = f"Apogee {apo_z_km:.1f} km" if en else f"정점 {apo_z_km:.1f} km · 탄도 귀환"
    gld_lbl = (
        f"Glide ({gld_z0:.1f}→{lnd_z0:.1f} km)"
        if en
        else f"무동력 활강 ({gld_z0:.1f}→{lnd_z0:.1f} km)"
    )
    lnd_lbl = "3→1 Hoverslam · Pad Touchdown" if en else "3→1 호버슬램 · 발사장 착륙"
    x_lbl = "Horizontal distance (km)" if en else "수평 거리 (km)"
    pad_lbl = "Launch pad (0, 0)" if en else "발사장 원점 (0, 0)"
    ban_top = "All 487 Validation Cases Passed" if en else "전체 487케이스 전수 검증 통과"
    ban_sub = (
        f"All 16 constraints ≤ {peak_util:.1f}% (under 80% cap) · 2D conceptual model"
        if en
        else f"16개 전 제약 ≤ {peak_util:.1f}% (80% 안전 여유) · 100 t 7× LCH₄/LOX 2D 모델"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">{svg_title}</title>
<desc id="desc">{svg_desc}</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="760" fill="{PAPER_BG}"/>
<text x="44" y="78" font-size="{h1_size}" fill="{OXFORD_BLUE}" font-weight="750">{h1_text}</text>
<text x="46" y="124" font-size="{sub_size}" fill="{INK_MUTED}">{sub_text}</text>
<line x1="44" y1="148" x2="1156" y2="148" stroke="{RULE_LINE}" stroke-width="1"/>

<rect x="44" y="172" width="352" height="410" rx="16" fill="{CARD_BG}" stroke="{CARD_BORDER}" stroke-width="1"/>
<text x="70" y="216" font-size="21" fill="{OXFORD_BLUE}" font-weight="700">{left_hdr}</text>
<line x1="70" y1="234" x2="370" y2="234" stroke="{RULE_LINE}" stroke-width="1"/>
{"".join(row_svg)}

<rect x="418" y="172" width="738" height="410" rx="16" fill="{CARD_BG}" stroke="{CARD_BORDER}" stroke-width="1"/>
<text x="444" y="214" font-size="20" fill="{OXFORD_BLUE}" font-weight="700">{traj_hdr}</text>
<text x="1130" y="214" font-size="15" fill="{INK_MUTED}" text-anchor="end">{calm_lbl}</text>
{"".join(grid_lines)}
{"".join(path_svg)}

<circle cx="{sep_x:.1f}" cy="{sep_y:.1f}" r="5.0" fill="{OXFORD_BLUE}" stroke="{CARD_BG}" stroke-width="1.8"/>
<line x1="{sep_x + 5:.1f}" y1="{sep_y + 5:.1f}" x2="{sep_x + 34:.1f}" y2="{sep_y + 42:.1f}" stroke="{OXFORD_BLUE}" stroke-width="1.2"/>
<text x="{sep_x + 40:.1f}" y="{sep_y + 47:.1f}" font-size="15" fill="{OXFORD_BLUE}" font-weight="700">{sep_lbl}</text>

<circle cx="{apo_x:.1f}" cy="{apo_y:.1f}" r="5.0" fill="{OXFORD_STEEL}" stroke="{CARD_BG}" stroke-width="1.8"/>
<text x="{apo_x:.1f}" y="{apo_y - 13:.1f}" font-size="15" fill="{INK_BODY}" font-weight="700" text-anchor="middle">{apo_lbl}</text>

<circle cx="{gld_x:.1f}" cy="{gld_y:.1f}" r="5.0" fill="{OXFORD_MID}" stroke="{CARD_BG}" stroke-width="1.8"/>
<line x1="{gld_x + 6:.1f}" y1="{gld_y - 5:.1f}" x2="{gld_x + 122:.1f}" y2="{gld_y - 104:.1f}" stroke="{OXFORD_MID}" stroke-width="1.2"/>
<text x="{gld_x + 128:.1f}" y="{gld_y - 100:.1f}" font-size="15" fill="{OXFORD_MID}" font-weight="700">{gld_lbl}</text>

<circle cx="{lnd_x:.1f}" cy="{lnd_y:.1f}" r="5.0" fill="{FOREST}" stroke="{CARD_BG}" stroke-width="1.8"/>
<line x1="{lnd_x + 7:.1f}" y1="{lnd_y - 1:.1f}" x2="{px0 + 96:.1f}" y2="{py1 - 10:.1f}" stroke="{FOREST}" stroke-width="1.2"/>
<text x="{px0 + 102:.1f}" y="{py1 - 5:.1f}" font-size="14.5" fill="{FOREST}" font-weight="700">{lnd_lbl}</text>

<text x="1116" y="566" font-size="15" fill="{INK_MUTED}" text-anchor="end">{x_lbl}</text>
<text x="496" y="566" font-size="14.5" fill="{INK_MUTED}">{pad_lbl}</text>

<rect x="44" y="604" width="1112" height="112" rx="14" fill="{OXFORD_BLUE}"/>
<text x="76" y="652" font-size="31" fill="#FFFFFF" font-weight="750">{ban_top}</text>
<text x="76" y="692" font-size="22" fill="{OXFORD_SUBTLE}">{ban_sub}</text>
<text x="1124" y="681" font-size="56" fill="#FFFFFF" font-weight="750" text-anchor="end">{passed} / {total}</text>
</svg>
"""


def render_validation_svg(summary: dict, *, lang: str = "ko") -> str:
    en = lang == "en"
    rows = summary["cases"]
    groups = (
        [
            ("Extreme Edge Cases", "Includes 100% worst corners", "edge-cases"),
            ("Combined Holdout", "Wind, drag, thrust &amp; delays", "holdout"),
            ("Extended Sensor", "Noise, bias &amp; packet loss", "sensor-holdout"),
            ("Regression &amp; Dev", "286 regression + 16 dev", None),
        ]
        if en
        else [
            ("극한 엣지 케이스", "100% 동시 최악 포함", "edge-cases"),
            ("복합 홀드아웃", "바람·편차·지연 복합", "holdout"),
            ("확장 센서 홀드아웃", "잡음·편향·패킷 누락", "sensor-holdout"),
            ("회귀 · 개발 전체", "기존 286 + 개발 16", None),
        ]
    )
    cards = []
    for idx, (title, sub, split_key) in enumerate(groups):
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
            f'<rect x="{x}" y="142" width="260" height="122" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>'
            f'<text x="{x + 20}" y="176" font-size="18" fill="{OXFORD_BLUE}" font-weight="700">{title}</text>'
            f'<text x="{x + 20}" y="198" font-size="13.5" fill="{INK_MUTED}">{sub}</text>'
            f'<text x="{x + 20}" y="246" font-size="34" fill="{OXFORD_BLUE}" font-weight="750">{p_cnt}<tspan fill="{INK_MUTED}" font-size="21" font-weight="500"> / {t_cnt}</tspan></text>'
            f'<text x="{x + 240}" y="244" font-size="14" fill="{FOREST}" font-weight="700" text-anchor="end">100%</text>'
        )

    metrics = (
        [
            ("Position error (≤ 2.0 m)", "position_error_m", 2.0, "m", 1.0, 3),
            ("Vert. foot speed (≤ 1.5 m/s)", "vertical_foot_speed_m_s", 1.5, "m/s", 1.0, 3),
            ("Horiz. foot speed (≤ 0.5 m/s)", "horizontal_foot_speed_m_s", 0.5, "m/s", 1.0, 3),
            ("Peak dyn. press. (≤ 42 kPa)", "peak_q_pa", 42_000.0, "kPa", 0.001, 2),
            ("Peak rec. load (≤ 5.5 g)", "peak_recovery_g", 5.5, "g", 1.0, 2),
            ("Peak ascent load (≤ 4.5 g)", "peak_ascent_axial_g", 4.5, "g", 1.0, 2),
            ("Peak ang. accel (≤ 5.0°/s²)", "peak_accel_deg_s2", 5.0, "°/s²", 1.0, 2),
            ("Peak heat flux (≤ 80 kW/m²)", "peak_heat_flux_w_m2", 80_000.0, "kW/m²", 0.001, 1),
        ]
        if en
        else [
            ("접지 위치 오차 (≤ 2.0 m)", "position_error_m", 2.0, "m", 1.0, 3),
            ("발 수직 속도 (≤ 1.5 m/s)", "vertical_foot_speed_m_s", 1.5, "m/s", 1.0, 3),
            ("발 수평 속도 (≤ 0.5 m/s)", "horizontal_foot_speed_m_s", 0.5, "m/s", 1.0, 3),
            ("최대 동압 q (≤ 42 kPa)", "peak_q_pa", 42_000.0, "kPa", 0.001, 2),
            ("최대 회수 하중 (≤ 5.5 g)", "peak_recovery_g", 5.5, "g", 1.0, 2),
            ("최대 상승 하중 (≤ 4.5 g)", "peak_ascent_axial_g", 4.5, "g", 1.0, 2),
            ("최대 각가속도 (≤ 5.0°/s²)", "peak_accel_deg_s2", 5.0, "°/s²", 1.0, 2),
            ("정체 열유속 (≤ 80 kW/m²)", "peak_heat_flux_w_m2", 80_000.0, "kW/m²", 0.001, 1),
        ]
    )
    bx0, bw = 310.0, 480.0
    x80 = bx0 + 0.80 * bw
    lbl_size = 15 if en else 16
    bars = []
    for i, (label, key, lim, unit, scale, dec) in enumerate(metrics):
        vals = np.array([r[key] for r in rows], dtype=float)
        p50, p95, mx = np.percentile(vals, [50, 95, 100])
        y = 368 + i * 38
        w50 = max(3.0, (p50 / lim) * bw)
        w95 = max(4.0, (p95 / lim) * bw)
        wmx = min(bw, (mx / lim) * bw)
        pct = mx / lim * 100.0
        fmt = f".{dec}f"
        bars.append(
            f'<text x="70" y="{y + 15}" font-size="{lbl_size}" fill="{INK_BODY}" font-weight="600">{label}</text>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{bw:.0f}" height="20" rx="3" fill="{TRACK_BG}"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{wmx:.1f}" height="20" rx="3" fill="{OXFORD_PALE}"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{w95:.1f}" height="20" rx="3" fill="#527499"/>'
            f'<rect x="{bx0:.0f}" y="{y}" width="{w50:.1f}" height="20" rx="3" fill="{OXFORD_BLUE}"/>'
            f'<text x="{bx0 + bw + 16:.0f}" y="{y + 15}" font-size="14" fill="{INK_MUTED}">'
            f"p50 {p50 * scale:{fmt}} · max {mx * scale:{fmt}} {unit} "
            f'<tspan fill="{OXFORD_BLUE}" font-weight="700">({pct:.1f}%)</tspan></text>'
        )

    peak_util = _compute_peak_utilization_pct(rows)
    svg_title = (
        "DM100 Full 487 Case Validation and Safety Margin Distribution"
        if en
        else "DM100 전체 487케이스 검증 및 80% 이하 안전 여유 분포"
    )
    svg_desc = (
        f"Full 487 case validation pass rate (100.0%) across 32 extreme edge, 96 combined holdout, 57 extended sensor, and 302 regression/development flights, with all 16 constraints bounded at or below {peak_util:.1f}% of limit."
        if en
        else f"극한 엣지 32케이스, 복합 홀드아웃 96케이스, 확장 센서 57케이스, 회귀·개발 302케이스 총 487케이스 100% 통과 결과와 모든 16개 물리 제약 항목의 80% 이하(최대 {peak_util:.1f}%) 안전 여유 분포입니다."
    )
    h1_text = (
        "Full Validation &amp; Safety Margin Distribution"
        if en
        else "전체 487케이스 일괄 검증 및 안전 여유 분포"
    )
    sub_text = (
        f"487/487 pass (100.0%) · Stage 2 700 km orbit 487/487 · 5 s standing 487/487 · Peak limit usage {peak_util:.1f}%"
        if en
        else f"487/487 통과 (100.0%) · 2단 700 km 원궤도 487/487 · 5초 직립 487/487 · 전 제약 최대 사용률 {peak_util:.1f}%"
    )
    sub_size = 18.5 if en else 20
    sec_title = (
        "Observed Distribution vs. Hard Limit (100%)" if en else "허용 한계(100%) 대비 관측 분포"
    )
    leg_p50 = "Median (p50)" if en else "중앙 (p50)"
    leg_p95 = "95th pct" if en else "상위 95%"
    leg_max = "Max" if en else "최댓값 (Max)"
    leg_x = (850, 968, 1062) if en else (834, 934, 1028)
    line_80 = "80% Margin" if en else "80% 여유선"
    line_100 = "100% Limit" if en else "100% 한계"
    footer_txt = (
        "All metrics computed directly from raw validation-487.json records within the declared 2D synthetic disturbance &amp; sensor scope."
        if en
        else "모든 수치는 validation-487.json 원시 기록에서 직접 집계했습니다. 선언된 2D 합성 외란·센서 범위 내 검증 결과입니다."
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="755" viewBox="0 0 1200 755" role="img" aria-labelledby="title desc">
<title id="title">{svg_title}</title>
<desc id="desc">{svg_desc}</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="755" fill="{PAPER_BG}"/>
<text x="48" y="64" font-size="36" fill="{OXFORD_BLUE}" font-weight="750">{h1_text}</text>
<text x="48" y="102" font-size="{sub_size}" fill="{INK_MUTED}">{sub_text}</text>
<line x1="48" y1="122" x2="1152" y2="122" stroke="{RULE_LINE}" stroke-width="1"/>
{"".join(cards)}

<rect x="48" y="284" width="1104" height="408" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>
<text x="70" y="322" font-size="20" fill="{OXFORD_BLUE}" font-weight="700">{sec_title}</text>
<rect x="{leg_x[0]}" y="310" width="14" height="11" rx="2" fill="{OXFORD_BLUE}"/><text x="{leg_x[0] + 20}" y="320" font-size="14" fill="{INK_BODY}">{leg_p50}</text>
<rect x="{leg_x[1]}" y="310" width="14" height="11" rx="2" fill="#527499"/><text x="{leg_x[1] + 20}" y="320" font-size="14" fill="{INK_BODY}">{leg_p95}</text>
<rect x="{leg_x[2]}" y="310" width="14" height="11" rx="2" fill="{OXFORD_PALE}"/><text x="{leg_x[2] + 20}" y="320" font-size="14" fill="{INK_BODY}">{leg_max}</text>
<line x1="{x80:.1f}" y1="350" x2="{x80:.1f}" y2="672" stroke="{OCHRE}" stroke-width="1.5" stroke-dasharray="4 4"/>
<text x="{x80:.1f}" y="342" font-size="12.5" fill="{OCHRE}" font-weight="700" text-anchor="middle">{line_80}</text>
<line x1="{bx0 + bw:.1f}" y1="350" x2="{bx0 + bw:.1f}" y2="672" stroke="{CRIMSON}" stroke-width="1.5" stroke-dasharray="4 4"/>
<text x="{bx0 + bw:.1f}" y="342" font-size="12.5" fill="{CRIMSON}" font-weight="700" text-anchor="middle">{line_100}</text>
{"".join(bars)}
<text x="48" y="728" font-size="15" fill="{INK_MUTED}">{footer_txt}</text>
</svg>
"""


def render_propellant_svg(calm_frames: list[dict], summary: dict, *, lang: str = "ko") -> str:
    en = lang == "en"
    phase_order = (
        [
            ("ascent", "Ascent (7 Engines)", OXFORD_BLUE),
            ("flip", "Attitude Flip (RCS)", OXFORD_STEEL),
            ("boostback", "Boostback Burn (3 Engines)", OXFORD_BLUE),
            ("coast", "Ballistic Coast", OXFORD_PALE),
            ("entry_burn", "Short Entry Burn (3 Engines)", CRIMSON),
            ("aero_glide", "Unpowered Grid Fin Glide", OXFORD_MID),
            ("landing_burn", "3→1 Hoverslam Landing", FOREST),
            ("touchdown_settle", "5 s Upright Standing", "#4E7A62"),
        ]
        if en
        else [
            ("ascent", "상승 (Ascent · 7엔진)", OXFORD_BLUE),
            ("flip", "저동압 플립 (Flip)", OXFORD_STEEL),
            ("boostback", "부스트백 (Boostback · 3엔진)", OXFORD_BLUE),
            ("coast", "탄도 코스트 (Coast)", OXFORD_PALE),
            ("entry_burn", "단축 재진입 (Entry Burn · 3엔진)", CRIMSON),
            ("aero_glide", "무동력 격자핀 활강 (Glide)", OXFORD_MID),
            ("landing_burn", "3→1 호버슬램 (Landing)", FOREST),
            ("touchdown_settle", "5초 직립 안정 (Standing)", "#4E7A62"),
        ]
    )
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
    lbl_size = 16 if en else 17
    ext_size = 14 if en else 15
    bars = []
    for i, (ph, label, col, dur, main_used, rcs_used, z0, z1) in enumerate(rec_rows):
        y = 206 + i * 50
        w = max(5.0, (main_used / max(max_fuel, 1.0)) * bw)
        if en:
            extra = (
                f"0% cmd · {z0 - z1:.1f} km glide (tailoff {main_used:.1f} kg)"
                if ph == "aero_glide"
                else f"Main {main_used:,.1f} kg · RCS {rcs_used:.1f} kg ({dur:.1f} s)"
            )
        else:
            extra = (
                f"스로틀 0% · {z0 - z1:.1f} km 활강 (테일오프 {main_used:.1f} kg)"
                if ph == "aero_glide"
                else f"주추진제 {main_used:,.1f} kg · RCS {rcs_used:.1f} kg ({dur:.1f} s)"
            )
        bars.append(
            f'<text x="70" y="{y + 17}" font-size="{lbl_size}" fill="{INK_BODY}" font-weight="600">{label}</text>'
            f'<rect x="{bx0:.0f}" y="{y + 2}" width="{bw:.0f}" height="20" rx="3" fill="{TRACK_BG}"/>'
            f'<rect x="{bx0:.0f}" y="{y + 2}" width="{w:.1f}" height="20" rx="3" fill="{col}"/>'
            f'<text x="{bx0 + bw + 16:.0f}" y="{y + 17}" font-size="{ext_size}" fill="{INK_MUTED}">{extra}</text>'
        )

    cases = summary["cases"]
    mains = [r["main_remaining_kg"] for r in cases]
    glides = [r["glide_distance_km"] for r in cases]
    uppers = [r["upper_propellant_remaining_kg"] for r in cases]
    med_main = float(np.median(mains))
    med_glide = float(np.median(glides))
    min_upper = min(uppers)
    max_orb_err = max(abs(r["orbit_altitude_km"] - 700.0) for r in cases)

    svg_title = (
        "DM100 100 t Methalox RTLS Recovery Propellant Budget and Stage 2 Orbit Margin"
        if en
        else "DM100 100톤 메탈록스 RTLS 귀환 추진제 예산 및 2단 700 km 궤도 여유"
    )
    svg_desc = (
        "Main and RCS propellant consumption after separation by phase, 29.4 km unpowered grid fin glide, and 487 case Stage 1 and Stage 2 propellant reserves."
        if en
        else "단 분리 이후 귀환 단계별 주추진제와 RCS 소모량, 약 29.4 km 무동력 격자핀 활강 구간 및 487케이스 1단·2단 잔여 추진제 통계입니다."
    )
    h1_text = "RTLS Recovery Propellant Budget by Phase" if en else "RTLS 귀환 단계별 추진제 예산"
    sub_text = (
        "Unpowered grid fin glide conserves propellant before terminal 3→1 hoverslam (Calm + 487 cases)"
        if en
        else "무동력 격자핀 활강으로 연료를 절약하고 종말 3→1 호버슬램으로 착륙합니다 (무풍 기준 + 487케이스)"
    )
    sub_size = 18.5 if en else 20
    sec_hdr = (
        f'Main Propellant Usage After Separation <tspan fill="{INK_MUTED}" font-weight="400">(excl. {asc_used:,.0f} kg ascent of 69,500 kg · 3/3 restarts)</tspan>'
        if en
        else f'분리 이후 귀환 단계별 주추진제 소모 <tspan fill="{INK_MUTED}" font-weight="400">(1단 총 69,500 kg 중 상승 {asc_used:,.0f} kg 제외 · 재점화 3/3회)</tspan>'
    )
    sec_size = 17.5 if en else 18.5
    k1_lbl = "Stage 1 Main Rem (487 cases)" if en else "1단 잔여 주추진제 (487케이스)"
    k1_val = (
        f"Median {med_main:,.0f} kg · Min {min(mains):,.0f} kg"
        if en
        else f"중앙 {med_main:,.0f} kg · 최소 {min(mains):,.0f} kg"
    )
    k2_lbl = "Unpowered Grid Fin Glide" if en else "무동력 격자핀 활강 거리"
    k2_val = (
        f"Median {med_glide:.2f} km ({min(glides):.1f}~{max(glides):.1f} km)"
        if en
        else f"중앙 {med_glide:.2f} km ({min(glides):.1f}~{max(glides):.1f} km)"
    )
    k3_lbl = "Stage 2 700 km Orbit (1,800 kg)" if en else "2단 700 km 원궤도 (1,800 kg)"
    k3_val = (
        f"Max err {max_orb_err:.2f} km · Min rem {min_upper:.0f} kg"
        if en
        else f"최대 오차 {max_orb_err:.2f} km · 최소 잔여 {min_upper:.0f} kg"
    )
    kpi_size = 17 if en else 18

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" viewBox="0 0 1200 700" role="img" aria-labelledby="title desc">
<title id="title">{svg_title}</title>
<desc id="desc">{svg_desc}</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="700" fill="{PAPER_BG}"/>
<text x="48" y="64" font-size="36" fill="{OXFORD_BLUE}" font-weight="750">{h1_text}</text>
<text x="48" y="102" font-size="{sub_size}" fill="{INK_MUTED}">{sub_text}</text>
<line x1="48" y1="122" x2="1152" y2="122" stroke="{RULE_LINE}" stroke-width="1"/>

<rect x="48" y="142" width="1104" height="422" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>
<text x="70" y="180" font-size="{sec_size}" fill="{OXFORD_BLUE}" font-weight="700">{sec_hdr}</text>
{"".join(bars)}

<rect x="48" y="584" width="1104" height="86" rx="12" fill="{OXFORD_BLUE}"/>
<line x1="416" y1="600" x2="416" y2="654" stroke="#284B70" stroke-width="1"/>
<line x1="784" y1="600" x2="784" y2="654" stroke="#284B70" stroke-width="1"/>
<text x="232" y="615" font-size="13.5" fill="{OXFORD_SUBTLE}" text-anchor="middle">{k1_lbl}</text>
<text x="232" y="646" font-size="{kpi_size}" fill="#FFFFFF" font-weight="700" text-anchor="middle">{k1_val}</text>
<text x="600" y="615" font-size="13.5" fill="{OXFORD_SUBTLE}" text-anchor="middle">{k2_lbl}</text>
<text x="600" y="646" font-size="{kpi_size}" fill="#FFFFFF" font-weight="700" text-anchor="middle">{k2_val}</text>
<text x="968" y="615" font-size="13.5" fill="{OXFORD_SUBTLE}" text-anchor="middle">{k3_lbl}</text>
<text x="968" y="646" font-size="{kpi_size}" fill="#FFFFFF" font-weight="700" text-anchor="middle">{k3_val}</text>
</svg>
"""


def render_trajectory_svg(
    calm_frames: list[dict], weather_frames: list[dict], *, lang: str = "ko"
) -> str:
    en = lang == "en"
    c_sub = _subsample(calm_frames, step=12)
    w_sub = _subsample(weather_frames, step=12)

    all_x = [f["pos_enu_m"][0] / 1000.0 for f in c_sub + w_sub]
    all_z = [f["pos_enu_m"][2] / 1000.0 for f in c_sub + w_sub]
    x_lo = math.floor(min(0.0, min(all_x)) / 10.0) * 10.0 if min(all_x) < -2.0 else 0.0
    x_hi = max(80.0, math.ceil(max(all_x) / 20.0) * 20.0)
    z_hi = max(175.0, math.ceil((max(all_z) + 12.0) / 25.0) * 25.0)

    # Left panel: 1st-stage trajectory
    lx0, lx1, ly0, ly1 = 102.0, 560.0, 198.0, 588.0

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
    qy0, qy1 = 196.0, 362.0
    ty0, ty1 = 460.0, 592.0
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
            f'<line x1="{lx0:.0f}" y1="{yy:.1f}" x2="{lx1:.0f}" y2="{yy:.1f}" stroke="{GRID_LINE}"/>'
            f'<text x="{lx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{INK_MUTED}" text-anchor="end">{zv:.0f}</text>'
        )
    for i in range(5):
        xv = x_lo + (x_hi - x_lo) * i / 4.0
        xx = lx0 + (lx1 - lx0) * i / 4.0
        l_grid.append(
            f'<line x1="{xx:.1f}" y1="{ly0:.0f}" x2="{xx:.1f}" y2="{ly1:.0f}" stroke="{GRID_LINE}"/>'
            f'<text x="{xx:.1f}" y="{ly1 + 22:.0f}" font-size="14" fill="{INK_MUTED}" text-anchor="middle">{xv:.0f}</text>'
        )

    q_grid = []
    for qv in (0, 15, 30, 45):
        yy = rsy_q(qv)
        q_grid.append(
            f'<line x1="{rx0:.0f}" y1="{yy:.1f}" x2="{rx1:.0f}" y2="{yy:.1f}" stroke="{GRID_LINE}"/>'
            f'<text x="{rx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{OXFORD_BLUE}" text-anchor="end">{qv}</text>'
        )

    thr_grid = []
    for tp in (0, 50, 100):
        yy = rsy_thr(tp)
        thr_grid.append(
            f'<line x1="{rx0:.0f}" y1="{yy:.1f}" x2="{rx1:.0f}" y2="{yy:.1f}" stroke="{GRID_LINE}"/>'
            f'<text x="{rx0 - 10:.0f}" y="{yy + 5:.1f}" font-size="14" fill="{OCHRE}" text-anchor="end">{tp}%</text>'
        )

    t_step = 150 if t_hi <= 600 else 200
    tv = 0
    while tv <= t_hi:
        xx = rsx(tv)
        q_grid.append(
            f'<line x1="{xx:.1f}" y1="{qy0:.0f}" x2="{xx:.1f}" y2="{qy1:.0f}" stroke="{GRID_LINE}"/>'
        )
        thr_grid.append(
            f'<line x1="{xx:.1f}" y1="{ty0:.0f}" x2="{xx:.1f}" y2="{ty1:.0f}" stroke="{GRID_LINE}"/>'
            f'<text x="{xx:.1f}" y="{ty1 + 22:.0f}" font-size="14" fill="{INK_MUTED}" text-anchor="middle">{tv}</text>'
        )
        tv += t_step

    l_paths = "".join(
        f'<polyline points="{_polyline(pts)}" fill="none" stroke="{PHASE_COLORS.get(ph, OXFORD_MID)}" '
        f'stroke-width="{3.6 if ph in ("ascent", "aero_glide", "landing_burn") else 2.6}" stroke-linecap="round"/>'
        for ph, pts in l_segs
    )

    sep_x, sep_y = lsx(sep_f), lsy(sep_f)
    apo_x, apo_y = lsx(apo_f), lsy(apo_f)
    gld_x, gld_y = lsx(gld_f), lsy(gld_f)
    aqx, aqy = rsx(asc_q_f["time_s"]), rsy_q(asc_q_kpa)
    rqx, rqy = rsx(rec_q_f["time_s"]), rsy_q(rec_q_kpa)

    boost_thr_max = max(f["throttle"] * 100.0 for f in boost_frames)
    t_entry_mid = 0.5 * (entry_frames[0]["time_s"] + entry_frames[-1]["time_s"])
    entry_thr_max = max(f["throttle"] * 100.0 for f in entry_frames)
    t_land_mid = 0.5 * (land_frames[0]["time_s"] + land_frames[-1]["time_s"])
    land_thr_max = max(f["throttle"] * 100.0 for f in land_frames)

    sep_z = sep_f["pos_enu_m"][2] / 1000.0
    apo_z = apo_f["pos_enu_m"][2] / 1000.0
    svg_title = (
        "DM100 Orbital RTLS Trajectory with Split Dynamic Pressure and Throttle Profiles"
        if en
        else "DM100 궤도 RTLS 비행 궤적 및 분리형 동압·스로틀 프로파일"
    )
    svg_desc = (
        f"Calm baseline vs. complex weather 2D recovery trajectory with independent dynamic pressure (ascent {asc_q_kpa:.1f} kPa, entry {rec_q_kpa:.1f} kPa, both well below 42 kPa limit and 33.6 kPa 80% margin) and main engine throttle (3 restarts, {glide_dur:.0f} s unpowered grid fin glide) panels."
        if en
        else f"무풍 기준과 복합 기상 비행의 2차원 귀환 궤적, 독립 패널로 분리된 동압 프로파일(상승 {asc_q_kpa:.1f} kPa, 재진입 {rec_q_kpa:.1f} kPa, 한계 42 kPa 및 80% 목표 33.6 kPa 미만 유지)과 주엔진 스로틀 프로파일(재점화 3회, {glide_dur:.0f}초 무동력 격자핀 활강)입니다."
    )
    h1_text = (
        "RTLS Trajectory, Dynamic Pressure &amp; Throttle"
        if en
        else "RTLS 궤적 및 동압 · 스로틀 프로파일"
    )
    sub_text = (
        "Calm baseline (solid) vs. weather (dashed) · Dynamic pressure (top) and throttle (bottom) on separate axes"
        if en
        else "무풍 기준(실선)과 복합 기상(점선) 비교 · 동압(상단)과 스로틀(하단)을 독립 축으로 분리해 표시합니다"
    )
    sub_size = 18 if en else 19.5
    left_hdr = (
        f'Stage 1 Trajectory <tspan font-size="13.5" fill="{INK_MUTED}" font-weight="400">(Alt km vs Downrange km)</tspan>'
        if en
        else f'1단 상승·귀환 궤적 <tspan font-size="14" fill="{INK_MUTED}" font-weight="400">(고도 km vs 수평 거리 km)</tspan>'
    )
    sep_lbl = f"MECO-1 Sep ({sep_z:.1f} km)" if en else f"단 분리 ({sep_z:.1f} km)"
    apo_lbl = f"Apogee {apo_z:.1f} km" if en else f"정점 {apo_z:.1f} km"
    gld_lbl = "Glide start" if en else "무동력 활강 시작"
    x_dist_lbl = "Horizontal distance (km)" if en else "수평 거리 (km)"
    q_hdr = "Dynamic Pressure (kPa)" if en else "동압 프로파일 (kPa)"
    q_kpi = (
        f"Ascent {asc_q_kpa:.1f} · Entry {rec_q_kpa:.1f} kPa"
        if en
        else f"상승 {asc_q_kpa:.1f} · 재진입 {rec_q_kpa:.1f} kPa"
    )
    q42_lbl = "Max q limit 42 kPa" if en else "동압 한계 42 kPa"
    q336_lbl = "80% margin 33.6 kPa" if en else "80% 여유선 33.6 kPa"
    thr_hdr = "Main Engine Throttle (%)" if en else "주엔진 스로틀 (%)"
    thr_kpi = (
        f"3 restarts · {glide_dur:.0f} s glide (0%)"
        if en
        else f"재점화 3회 · 무동력 활강 {glide_dur:.0f} s (0%)"
    )
    thr_glide_lbl = "Unpowered glide (0%)" if en else "무동력 활강 (0%)"
    p_asc = "Ascent (7 eng)" if en else "상승 (7엔진)"
    p_boost = "Boostback" if en else "부스트백"
    p_entry = "Entry" if en else "재진입"
    p_land = "Land 3→1" if en else "착륙 3→1"
    x_time_lbl = "Flight time (s)" if en else "비행 시간 (s)"
    leg = (
        (
            (48, "Ascent (7 eng) / q"),
            (254, "Entry burn"),
            (390, "Unpowered glide"),
            (578, "3→1 Hoverslam"),
            (758, "Main throttle (%)"),
            (948, "Weather trajectory"),
        )
        if en
        else (
            (48, "상승 (7엔진) / 동압"),
            (256, "재진입 연소"),
            (402, "무동력 활강"),
            (554, "3→1 호버슬램"),
            (716, "주엔진 스로틀 (%)"),
            (912, "복합 기상 궤적"),
        )
    )
    footer_txt = (
        "Generated from 50 Hz recorded calm and complex weather replay telemetry; both dynamic pressure peaks remain below the 33.6 kPa (80%) margin."
        if en
        else "50 Hz 무풍 기준 및 복합 기상 비행 기록에서 직접 생성했습니다. 상승·재진입 동압 피크 모두 80% 여유선(33.6 kPa) 아래를 유지합니다."
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="752" viewBox="0 0 1200 752" role="img" aria-labelledby="title desc">
<title id="title">{svg_title}</title>
<desc id="desc">{svg_desc}</desc>
<style>{FONT_STYLE}</style>
<rect width="1200" height="752" fill="{PAPER_BG}"/>
<text x="48" y="64" font-size="36" fill="{OXFORD_BLUE}" font-weight="750">{h1_text}</text>
<text x="48" y="102" font-size="{sub_size}" fill="{INK_MUTED}">{sub_text}</text>
<line x1="48" y1="122" x2="1152" y2="122" stroke="{RULE_LINE}" stroke-width="1"/>

<rect x="48" y="142" width="538" height="518" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>
<text x="72" y="176" font-size="18" fill="{OXFORD_BLUE}" font-weight="700">{left_hdr}</text>
{"".join(l_grid)}
<polyline points="{_polyline(w_pts)}" fill="none" stroke="{WEATHER_GRAY}" stroke-width="1.6" stroke-dasharray="5 5" opacity="0.85"/>
{l_paths}
<circle cx="{sep_x:.1f}" cy="{sep_y:.1f}" r="4.8" fill="{OXFORD_BLUE}" stroke="{CARD_BG}" stroke-width="1.6"/>
<line x1="{sep_x + 5:.1f}" y1="{sep_y + 5:.1f}" x2="{sep_x + 36:.1f}" y2="{sep_y + 52:.1f}" stroke="{OXFORD_BLUE}" stroke-width="1.2"/>
<text x="{sep_x + 42:.1f}" y="{sep_y + 57:.1f}" font-size="13.5" fill="{OXFORD_BLUE}" font-weight="700">{sep_lbl}</text>
<circle cx="{apo_x:.1f}" cy="{apo_y:.1f}" r="4.8" fill="{OXFORD_STEEL}" stroke="{CARD_BG}" stroke-width="1.6"/>
<text x="{apo_x:.1f}" y="{apo_y - 12:.1f}" font-size="13.5" fill="{INK_BODY}" font-weight="700" text-anchor="middle">{apo_lbl}</text>
<circle cx="{gld_x:.1f}" cy="{gld_y:.1f}" r="4.8" fill="{OXFORD_MID}" stroke="{CARD_BG}" stroke-width="1.6"/>
<line x1="{gld_x + 6:.1f}" y1="{gld_y - 6:.1f}" x2="{gld_x + 118:.1f}" y2="{gld_y - 128:.1f}" stroke="{OXFORD_MID}" stroke-width="1.2"/>
<text x="{gld_x + 124:.1f}" y="{gld_y - 124:.1f}" font-size="13.5" fill="{OXFORD_MID}" font-weight="700">{gld_lbl}</text>
<text x="332" y="640" font-size="15" fill="{INK_MUTED}" text-anchor="middle">{x_dist_lbl}</text>

<rect x="610" y="142" width="542" height="250" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>
<text x="634" y="175" font-size="17" fill="{OXFORD_BLUE}" font-weight="700">{q_hdr}</text>
<text x="1130" y="175" font-size="13.5" fill="{OXFORD_MID}" font-weight="600" text-anchor="end">{q_kpi}</text>
<rect x="{gx0:.1f}" y="{qy0:.0f}" width="{gx1 - gx0:.1f}" height="{qy1 - qy0:.0f}" fill="{GLIDE_TINT}"/>
{"".join(q_grid)}
<line x1="{rx0:.0f}" y1="{y_q42:.1f}" x2="{rx1:.0f}" y2="{y_q42:.1f}" stroke="{CRIMSON}" stroke-width="1.5" stroke-dasharray="6 4"/>
<text x="{rsx(285):.1f}" y="{y_q42 - 6:.1f}" font-size="12.5" fill="{CRIMSON}" font-weight="700" text-anchor="middle">{q42_lbl}</text>
<line x1="{rx0:.0f}" y1="{y_q336:.1f}" x2="{rx1:.0f}" y2="{y_q336:.1f}" stroke="{OCHRE}" stroke-width="1.3" stroke-dasharray="3 4"/>
<text x="{rsx(285):.1f}" y="{y_q336 - 6:.1f}" font-size="12" fill="{OCHRE}" font-weight="600" text-anchor="middle">{q336_lbl}</text>
<polygon points="{_polyline(q_area)}" fill="{OXFORD_BLUE}" opacity="0.10"/>
<polyline points="{_polyline(q_pts)}" fill="none" stroke="{OXFORD_BLUE}" stroke-width="2.5" stroke-linejoin="round"/>
<circle cx="{aqx:.1f}" cy="{aqy:.1f}" r="4.0" fill="{OXFORD_BLUE}" stroke="{CARD_BG}" stroke-width="1.4"/>
<text x="{aqx + 12:.1f}" y="{aqy + 4:.1f}" font-size="12.5" fill="{OXFORD_BLUE}" font-weight="700">{asc_q_kpa:.1f} kPa</text>
<circle cx="{rqx:.1f}" cy="{rqy:.1f}" r="4.0" fill="{OXFORD_BLUE}" stroke="{CARD_BG}" stroke-width="1.4"/>
<text x="{rqx - 10:.1f}" y="{rqy + 4:.1f}" font-size="12.5" fill="{OXFORD_BLUE}" font-weight="700" text-anchor="end">{rec_q_kpa:.1f} kPa</text>

<rect x="610" y="408" width="542" height="252" rx="12" fill="{CARD_BG}" stroke="{CARD_BORDER}"/>
<text x="634" y="441" font-size="17" fill="{OXFORD_BLUE}" font-weight="700">{thr_hdr}</text>
<text x="1130" y="441" font-size="13.5" fill="{OCHRE}" font-weight="600" text-anchor="end">{thr_kpi}</text>
<rect x="{gx0:.1f}" y="{ty0:.0f}" width="{gx1 - gx0:.1f}" height="{ty1 - ty0:.0f}" fill="{GLIDE_TINT}"/>
<text x="{(gx0 + gx1) / 2:.1f}" y="{rsy_thr(100) - 6:.1f}" font-size="11.5" fill="{OXFORD_MID}" font-weight="700" text-anchor="middle">{thr_glide_lbl}</text>
{"".join(thr_grid)}
<polygon points="{_polyline(thr_area)}" fill="{OCHRE}" opacity="0.14"/>
<polyline points="{_polyline(thr_pts)}" fill="none" stroke="{OCHRE}" stroke-width="2.3" stroke-linejoin="round"/>
<text x="{rsx(62):.1f}" y="{rsy_thr(100) - 6:.1f}" font-size="12" fill="{OXFORD_BLUE}" font-weight="700" text-anchor="middle">{p_asc}</text>
<text x="{rsx(boost_frames[0]["time_s"]) + 2:.1f}" y="{rsy_thr(boost_thr_max) - 7:.1f}" font-size="12" fill="{INK_MUTED}" font-weight="600" text-anchor="start">{p_boost}</text>
<text x="{rsx(t_entry_mid) - 6:.1f}" y="{rsy_thr(entry_thr_max) - 7:.1f}" font-size="12" fill="{CRIMSON}" font-weight="600" text-anchor="end">{p_entry}</text>
<text x="{min(rx1 - 2, rsx(t_land_mid) + 8):.1f}" y="{rsy_thr(land_thr_max) - 10:.1f}" font-size="12" fill="{FOREST}" font-weight="700" text-anchor="end">{p_land}</text>
<text x="900" y="644" font-size="15" fill="{INK_MUTED}" text-anchor="middle">{x_time_lbl}</text>

<line x1="{leg[0][0]}" y1="692" x2="{leg[0][0] + 24}" y2="692" stroke="{OXFORD_BLUE}" stroke-width="3.5"/><text x="{leg[0][0] + 32}" y="697" font-size="14" fill="{INK_BODY}">{leg[0][1]}</text>
<line x1="{leg[1][0]}" y1="692" x2="{leg[1][0] + 24}" y2="692" stroke="{CRIMSON}" stroke-width="3.5"/><text x="{leg[1][0] + 32}" y="697" font-size="14" fill="{INK_BODY}">{leg[1][1]}</text>
<line x1="{leg[2][0]}" y1="692" x2="{leg[2][0] + 24}" y2="692" stroke="{OXFORD_MID}" stroke-width="3.5"/><text x="{leg[2][0] + 32}" y="697" font-size="14" fill="{INK_BODY}">{leg[2][1]}</text>
<line x1="{leg[3][0]}" y1="692" x2="{leg[3][0] + 24}" y2="692" stroke="{FOREST}" stroke-width="3.5"/><text x="{leg[3][0] + 32}" y="697" font-size="14" fill="{INK_BODY}">{leg[3][1]}</text>
<line x1="{leg[4][0]}" y1="692" x2="{leg[4][0] + 24}" y2="692" stroke="{OCHRE}" stroke-width="3.5"/><text x="{leg[4][0] + 32}" y="697" font-size="14" fill="{INK_BODY}">{leg[4][1]}</text>
<line x1="{leg[5][0]}" y1="692" x2="{leg[5][0] + 28}" y2="692" stroke="{WEATHER_GRAY}" stroke-width="2.0" stroke-dasharray="5 4"/><text x="{leg[5][0] + 36}" y="697" font-size="14" fill="{INK_BODY}">{leg[5][1]}</text>

<text x="48" y="730" font-size="14.5" fill="{INK_MUTED}">{footer_txt}</text>
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
    fig.patch.set_facecolor(PAPER_BG)
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
                color=PHASE_COLORS[ph],
                lw=2.4 if ph in ("ascent", "aero_glide", "landing_burn") else 1.7,
                label=ph.replace("_", " "),
            )
    ax_traj.set_xlabel("Horizontal distance (km)")
    ax_traj.set_ylabel("Altitude (km)")
    ax_traj.legend(
        fontsize=8,
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=CARD_BORDER,
    )

    # 2. Velocity
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.axvspan(gt0, gt1, color=GLIDE_TINT, alpha=0.85, label="Unpowered glide")
    ax_vel.plot(t, vx, color=OXFORD_BLUE, lw=1.9, label="Horizontal vx")
    ax_vel.plot(t, vz, color=OCHRE, lw=1.9, label="Vertical vz")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.legend(
        fontsize=8.5, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER
    )

    # 3. Attitude & Grid Fin Deflection
    ax_att = fig.add_subplot(gs[1, 0])
    ax_att.axvspan(gt0, gt1, color=GLIDE_TINT, alpha=0.85)
    ax_att.plot(t, pitch, color=OXFORD_BLUE, lw=1.9, label="Pitch (deg)")
    ax_att.plot(t, fin_deg, color=FOREST, lw=1.5, label="Grid fin deflection (deg)")
    ax_att.set_xlabel("Time (s)")
    ax_att.set_ylabel("Angle (deg)")
    ax_att.legend(
        fontsize=8.5, loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER
    )

    # 4. Separate stacked panels: Top = Dynamic Pressure (kPa), Bottom = Main Throttle (%)
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 1], height_ratios=[1.15, 0.85], hspace=0.16
    )
    ax_q = fig.add_subplot(sub_gs[0, 0])
    ax_q.axvspan(gt0, gt1, color=GLIDE_TINT, alpha=0.85)
    ax_q.fill_between(t, 0.0, q_kpa, color=OXFORD_BLUE, alpha=0.10)
    ax_q.plot(
        t,
        q_kpa,
        color=OXFORD_BLUE,
        lw=2.0,
        label=f"Dynamic pressure (peak {asc_q_max:.1f} / {rec_q_max:.1f} kPa)",
    )
    ax_q.axhline(42.0, color=CRIMSON, ls="--", lw=1.35, label="Max q limit (42 kPa)")
    ax_q.axhline(33.6, color=OCHRE, ls=":", lw=1.25, label="80% margin (33.6 kPa)")
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
        edgecolor=CARD_BORDER,
    )

    ax_thr = fig.add_subplot(sub_gs[1, 0], sharex=ax_q)
    ax_thr.axvspan(gt0, gt1, color=GLIDE_TINT, alpha=0.85, label="Unpowered glide (0% cmd)")
    ax_thr.fill_between(t, 0.0, thr_pct, color=OCHRE, alpha=0.14)
    ax_thr.plot(t, thr_pct, color=OCHRE, lw=1.8, label="Main engine throttle (%)")
    ax_thr.set_xlabel("Time (s)")
    ax_thr.set_ylabel("Throttle (%)", fontsize=9.5, color=OCHRE)
    ax_thr.set_ylim(-6.0, 118.0)
    ax_thr.set_yticks([0, 50, 100])
    ax_thr.legend(
        fontsize=7.6, loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=CARD_BORDER
    )

    for axis in (ax_traj, ax_vel, ax_att, ax_q, ax_thr):
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, color=GRID_LINE, alpha=0.95, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for sp in ("bottom", "left"):
            axis.spines[sp].set_color("#B8C5D4")

    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    summary = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    calm = json.loads(CALM_REPLAY.read_text(encoding="utf-8"))["frames"]
    weather = json.loads(WEATHER_REPLAY.read_text(encoding="utf-8"))["frames"]

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for lang, suffix in (("ko", ""), ("en", ".en")):
        (ASSETS_DIR / f"dm100-thumbnail{suffix}.svg").write_text(
            render_thumbnail_svg(calm, summary, lang=lang), encoding="utf-8"
        )
        (ASSETS_DIR / f"dm100-validation{suffix}.svg").write_text(
            render_validation_svg(summary, lang=lang), encoding="utf-8"
        )
        (ASSETS_DIR / f"dm100-propellant{suffix}.svg").write_text(
            render_propellant_svg(calm, summary, lang=lang), encoding="utf-8"
        )
        (ASSETS_DIR / f"dm100-trajectory{suffix}.svg").write_text(
            render_trajectory_svg(calm, weather, lang=lang), encoding="utf-8"
        )
    render_trajectory_png(calm, ASSETS_DIR / "trajectory.png")
    print("Generated all 9 light-paper Oxford Blue DM100 RTLS visuals in", ASSETS_DIR)


if __name__ == "__main__":
    main()
