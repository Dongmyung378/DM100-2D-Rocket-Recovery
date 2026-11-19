"""Core physics, atmosphere, integrator, estimator, and input-validation tests."""

from __future__ import annotations

import numpy as np
import pytest

from powered_landing_guidance.apr_recovery import (
    AprConditions,
    Design,
    RecoveryDisturbances,
)
from powered_landing_guidance.integrators import rk4_step
from powered_landing_guidance.recovery_physics import (
    G0,
    AtmosphereSettings,
    PlanarNavigationEstimator,
    atmosphere,
    gravity,
)


def test_rk4_step_matches_harmonic_oscillator_solution() -> None:
    y = np.array([1.0, 0.0], dtype=np.float64)
    dt = 0.01
    for step in range(100):
        y = rk4_step(
            lambda _t, s: np.array([s[1], -s[0]], dtype=np.float64),
            step * dt,
            y,
            dt,
        )
    assert y[0] == pytest.approx(np.cos(1.0), abs=1e-9)
    assert y[1] == pytest.approx(-np.sin(1.0), abs=1e-9)


def test_standard_atmosphere_and_gravity_monotone() -> None:
    t0, p0, rho0 = atmosphere(0.0, AtmosphereSettings())
    t20, p20, rho20 = atmosphere(20_000.0, AtmosphereSettings())
    assert t0 == pytest.approx(288.15)
    assert p0 == pytest.approx(101_325.0)
    assert rho0 == pytest.approx(1.225, abs=0.002)
    assert p20 < p0 and rho20 < rho0 and t20 < t0
    assert gravity(0.0) == pytest.approx(G0, rel=1e-5)
    assert gravity(50_000.0) < gravity(0.0)


def test_design_mass_budget_consistency() -> None:
    d = Design()
    assert d.launch_mass_kg == pytest.approx(100_000.0)
    assert d.dry_mass_kg + d.main_propellant_kg + d.rcs_propellant_kg == pytest.approx(82_500.0)


def test_operating_envelope_validation_rejects_out_of_range() -> None:
    assert AprConditions().density_scale == 1.0
    assert RecoveryDisturbances().thrust_scale == 1.0
    with pytest.raises(ValueError):
        AprConditions(density_scale=1.25)
    with pytest.raises(ValueError):
        RecoveryDisturbances(thrust_scale=1.20)
    with pytest.raises(ValueError):
        RecoveryDisturbances(observation_delay_steps=5)


def test_navigation_estimator_tracks_constant_acceleration_and_bias() -> None:
    nav = PlanarNavigationEstimator(position_error_m=0.5, velocity_error_m_s=0.2)
    rng = np.random.default_rng(42)
    true_bx, true_bz = 0.4, -0.6
    x, z, vx, vz = 0.0, 100.0, 2.0, -10.0
    dt = 0.02
    for step in range(250):
        t = step * dt
        raw = np.zeros(6, dtype=np.float64)
        raw[0] = x + rng.uniform(-0.5, 0.5)
        raw[1] = z + rng.uniform(-0.5, 0.5)
        raw[2] = vx + rng.uniform(-0.2, 0.2)
        raw[3] = vz + rng.uniform(-0.2, 0.2)
        est = nav.observe(raw, t, (0.0, -G0), delay_s=0.0)
        x += vx * dt + 0.5 * true_bx * dt * dt
        z += vz * dt + 0.5 * (-G0 + true_bz) * dt * dt
        vx += true_bx * dt
        vz += (-G0 + true_bz) * dt
    assert nav.estimate is not None
    assert nav.estimate[0, 2] == pytest.approx(true_bx, abs=0.25)
    assert nav.estimate[1, 2] == pytest.approx(true_bz, abs=0.25)
    assert abs(est[0] - x) < 1.0
