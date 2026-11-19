"""2D Convex Optimal Hoverslam trajectory planner (3D SOCP-ready architecture).

Formulates minimum-fuel terminal powered descent with:
  - Minimum and maximum vertical thrust acceleration bounds (non-hovering Hoverslam)
  - Tilt-cone constraint |a_x| <= tan(theta_max) * a_z
  - Terminal verticality constraint |a_x| <= tan(theta_final) * a_z near touchdown
  - Glide-slope and non-negative altitude constraints
  - Exact boundary conditions (x, z, vx, vz) -> (0, 0, 0, -v_td)

In 2D, these cone constraints are exact linear inequalities solved in ~2 ms
by SciPy's HiGHS LP solver with zero extra dependencies. In Stage 2 (3D MuJoCo),
replacing |a_x| <= tan(theta) a_z with ||(a_x, a_y)||_2 <= tan(theta) a_z yields
the exact 3D GFOLD second-order cone program (SOCP).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, tan

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True, slots=True)
class ConvexLandingProblem:
    x0_m: float
    z0_m: float
    vx0_m_s: float
    vz0_m_s: float
    a_min_m_s2: float
    a_max_m_s2: float
    gravity_m_s2: float = 9.80665
    v_touchdown_m_s: float = 0.85
    max_tilt_deg: float = 14.0
    final_tilt_deg: float = 1.5
    final_vertical_fraction: float = 0.25
    glide_slope_max_deg: float = 60.0
    pad_margin_m: float = 2.0
    n_steps: int = 20


@dataclass(frozen=True, slots=True)
class ConvexLandingSolution:
    feasible: bool
    status: str
    flight_time_s: float
    dt_s: float
    times_s: tuple[float, ...]
    x_m: tuple[float, ...]
    z_m: tuple[float, ...]
    vx_m_s: tuple[float, ...]
    vz_m_s: tuple[float, ...]
    ax_m_s2: tuple[float, ...]
    az_m_s2: tuple[float, ...]
    fuel_proxy_dv_m_s: float


def _solve_fixed_tf(prob: ConvexLandingProblem, tf: float) -> ConvexLandingSolution | None:
    """Solve the discretized convex Hoverslam LP for a fixed duration `tf`."""
    n = prob.n_steps
    dt = tf / n
    g = prob.gravity_m_s2
    half_dt2 = 0.5 * dt * dt

    n_state = 4 * (n + 1)
    n_ctrl = 3 * n
    n_vars = n_state + n_ctrl

    def sx(k: int) -> int:
        return 4 * k

    def sz(k: int) -> int:
        return 4 * k + 1

    def svx(k: int) -> int:
        return 4 * k + 2

    def svz(k: int) -> int:
        return 4 * k + 3

    def uax(k: int) -> int:
        return n_state + 3 * k

    def uaz(k: int) -> int:
        return n_state + 3 * k + 1

    def us(k: int) -> int:
        return n_state + 3 * k + 2

    c = np.zeros(n_vars, dtype=np.float64)
    for k in range(n):
        c[uaz(k)] = dt
        c[us(k)] = 0.05 * dt

    n_eq = 4 + 4 * n + 4
    a_eq = np.zeros((n_eq, n_vars), dtype=np.float64)
    b_eq = np.zeros(n_eq, dtype=np.float64)
    row = 0

    for idx, val in (
        (sx(0), prob.x0_m),
        (sz(0), prob.z0_m),
        (svx(0), prob.vx0_m_s),
        (svz(0), prob.vz0_m_s),
    ):
        a_eq[row, idx] = 1.0
        b_eq[row] = val
        row += 1

    for k in range(n):
        a_eq[row, sx(k + 1)] = 1.0
        a_eq[row, sx(k)] = -1.0
        a_eq[row, svx(k)] = -dt
        a_eq[row, uax(k)] = -half_dt2
        b_eq[row] = 0.0
        row += 1

        a_eq[row, sz(k + 1)] = 1.0
        a_eq[row, sz(k)] = -1.0
        a_eq[row, svz(k)] = -dt
        a_eq[row, uaz(k)] = -half_dt2
        b_eq[row] = -half_dt2 * g
        row += 1

        a_eq[row, svx(k + 1)] = 1.0
        a_eq[row, svx(k)] = -1.0
        a_eq[row, uax(k)] = -dt
        b_eq[row] = 0.0
        row += 1

        a_eq[row, svz(k + 1)] = 1.0
        a_eq[row, svz(k)] = -1.0
        a_eq[row, uaz(k)] = -dt
        b_eq[row] = -dt * g
        row += 1

    for idx, val in (
        (sx(n), 0.0),
        (sz(n), 0.0),
        (svx(n), 0.0),
        (svz(n), -abs(prob.v_touchdown_m_s)),
    ):
        a_eq[row, idx] = 1.0
        b_eq[row] = val
        row += 1

    n_ub = 4 * n + 2 * (n - 1)
    a_ub = np.zeros((n_ub, n_vars), dtype=np.float64)
    b_ub = np.zeros(n_ub, dtype=np.float64)
    urow = 0
    tan_max = tan(prob.max_tilt_deg * pi / 180.0)
    tan_fin = tan(prob.final_tilt_deg * pi / 180.0)
    tan_gs = tan(prob.glide_slope_max_deg * pi / 180.0)
    fin_start = int(round((1.0 - prob.final_vertical_fraction) * n))

    for k in range(n):
        t_lim = tan_fin if k >= fin_start else tan_max
        a_ub[urow, uax(k)] = 1.0
        a_ub[urow, uaz(k)] = -t_lim
        urow += 1

        a_ub[urow, uax(k)] = -1.0
        a_ub[urow, uaz(k)] = -t_lim
        urow += 1

        a_ub[urow, uax(k)] = 1.0
        a_ub[urow, us(k)] = -1.0
        urow += 1

        a_ub[urow, uax(k)] = -1.0
        a_ub[urow, us(k)] = -1.0
        urow += 1

    for k in range(1, n):
        a_ub[urow, sx(k)] = 1.0
        a_ub[urow, sz(k)] = -tan_gs
        b_ub[urow] = prob.pad_margin_m
        urow += 1

        a_ub[urow, sx(k)] = -1.0
        a_ub[urow, sz(k)] = -tan_gs
        b_ub[urow] = prob.pad_margin_m
        urow += 1

    bounds: list[tuple[float | None, float | None]] = [(None, None)] * n_vars
    for k in range(n + 1):
        bounds[sz(k)] = (0.0, None)
        bounds[svz(k)] = (None, 0.0)
    for k in range(n):
        bounds[uaz(k)] = (prob.a_min_m_s2, prob.a_max_m_s2)
        bounds[us(k)] = (0.0, None)

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return None

    sol = res.x
    times = tuple(k * dt for k in range(n + 1))
    xs = tuple(float(sol[sx(k)]) for k in range(n + 1))
    zs = tuple(float(sol[sz(k)]) for k in range(n + 1))
    vxs = tuple(float(sol[svx(k)]) for k in range(n + 1))
    vzs = tuple(float(sol[svz(k)]) for k in range(n + 1))
    axs = tuple(float(sol[uax(k)]) for k in range(n))
    azs = tuple(float(sol[uaz(k)]) for k in range(n))
    fuel_dv = float(sum(azs) * dt)
    return ConvexLandingSolution(
        feasible=True,
        status="optimal",
        flight_time_s=tf,
        dt_s=dt,
        times_s=times,
        x_m=xs,
        z_m=zs,
        vx_m_s=vxs,
        vz_m_s=vzs,
        ax_m_s2=axs,
        az_m_s2=azs,
        fuel_proxy_dv_m_s=fuel_dv,
    )


def solve_convex_hoverslam(prob: ConvexLandingProblem) -> ConvexLandingSolution:
    """Search flight duration tf for the minimum-fuel feasible Hoverslam trajectory."""
    v0 = max(-prob.vz0_m_s, 1.0)
    z0 = max(prob.z0_m, 1.0)
    a_net_max = max(prob.a_max_m_s2 - prob.gravity_m_s2, 1.0)
    a_net_min = max(prob.a_min_m_s2 - prob.gravity_m_s2, 0.3)

    tf_min = max(1.0, min(v0 / a_net_max, 2.0 * z0 / (v0 + prob.v_touchdown_m_s)) * 0.75)
    tf_max = max(tf_min + 2.0, min(60.0, max(v0 / a_net_min, 2.0 * z0 / (0.5 * v0 + 1.0)) * 1.4))

    best: ConvexLandingSolution | None = None
    candidates = np.linspace(tf_min, tf_max, 10)
    for tf in candidates:
        sol = _solve_fixed_tf(prob, float(tf))
        if sol is not None and (best is None or sol.fuel_proxy_dv_m_s < best.fuel_proxy_dv_m_s):
            best = sol

    if best is None:
        return ConvexLandingSolution(
            feasible=False,
            status="infeasible",
            flight_time_s=0.0,
            dt_s=0.0,
            times_s=(),
            x_m=(),
            z_m=(),
            vx_m_s=(),
            vz_m_s=(),
            ax_m_s2=(),
            az_m_s2=(),
            fuel_proxy_dv_m_s=float("inf"),
        )

    step = (tf_max - tf_min) / 9.0
    lo = max(tf_min, best.flight_time_s - step)
    hi = min(tf_max, best.flight_time_s + step)
    for tf in np.linspace(lo, hi, 7):
        sol = _solve_fixed_tf(prob, float(tf))
        if sol is not None and sol.fuel_proxy_dv_m_s < best.fuel_proxy_dv_m_s:
            best = sol
    return best
