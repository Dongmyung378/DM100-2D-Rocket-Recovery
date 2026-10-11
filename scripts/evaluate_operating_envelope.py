"""Reproducible development and predeclared holdout sweeps for the 100 t APR model.

The cases are synthetic inputs inside the *project* operating envelope. A
passing input is never counted as a successful flight unless the simulated
mission also passes every applicable criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from powered_landing_guidance.apr_recovery import (
    OPERATING_SCOPE,
    PROFILE_TOP_M,
    AprConditions,
    AprFlight,
    Design,
    RecoveryDisturbances,
    _density_ratio_extrema,
    load_apr_operating,
)
from powered_landing_guidance.recovery_physics import AtmosphereSettings, atmosphere

ROOT = Path(__file__).resolve().parents[1]
ROOT_SEED = 378
MANIFEST_SCHEMA_VERSION = 13
VEHICLE_ID = "APR 100 t fixed design"
GROUPS = ("surface", "upper", "shear", "gust", "vertical", "density", "vehicle", "combined")
SOURCE_PATHS = (
    "src/powered_landing_guidance/apr_recovery.py",
    "src/powered_landing_guidance/recovery_physics.py",
    "src/powered_landing_guidance/vehicle_systems.py",
    "src/powered_landing_guidance/integrators.py",
    "src/powered_landing_guidance/observation_models.py",
    "scripts/evaluate_operating_envelope.py",
    "configs/apr-operating-v1.yaml",
)


def _encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.item(),
        allow_nan=False,
    )


def _sha(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def observation_seed_sequence(case):
    """Give each case its own deterministic sensor stream under root seed 378."""
    identity = f"{case['group']}\0{case['id']}".encode()
    digest = hashlib.sha256(identity).digest()
    identity_words = [
        int.from_bytes(digest[offset : offset + 4], "little") for offset in range(0, len(digest), 4)
    ]
    return np.random.SeedSequence([ROOT_SEED, case["stream"], case["index"], *identity_words])


def _profile(pad, middle, upper):
    return (
        (0.0, pad),
        (50.0, pad),
        (1000.0, pad + (1000.0 - 50.0) / (5000.0 - 50.0) * (middle - pad)),
        (5000.0, middle),
        (10000.0, upper),
        (PROFILE_TOP_M, upper),
    )


def _case(name, group, conditions, disturbances, stream, index):
    # Validate each generated case through the same public input contract as
    # a manually supplied YAML file; do not silently discard invalid draws.
    checked_conditions = AprConditions(**asdict(conditions))
    checked_disturbances = RecoveryDisturbances(**asdict(disturbances))
    return {
        "id": name,
        "group": group,
        "stream": stream,
        "index": index,
        "conditions": asdict(checked_conditions),
        "disturbances": asdict(checked_disturbances),
    }


def _random_case(group, stream, index):
    rng = np.random.default_rng(
        np.random.SeedSequence([ROOT_SEED, stream, GROUPS.index(group), index])
    )
    sign = float(rng.choice((-1, 1)))
    disturbances = RecoveryDisturbances()
    pad = middle = upper = 0.0
    gust = 0.0
    period = 12.0
    phase = 0.0
    gust_start = 0.0
    gust_duration = 60.0
    gust_center = 10000.0
    gust_half_width = 2500.0
    forecast_gust = 0.0
    vertical = 0.0
    density_scale = 1.0
    sea_level_temperature = 288.15
    sea_level_pressure = 101325.0
    temperature_profile = ()
    pressure_profile = ()
    latitude = 37.0
    forecast_offset = 0.0
    if group == "surface":
        pad = sign * rng.uniform(3.0, 8.0)
        middle = upper = pad
    elif group == "upper":
        pad = 0.0
        upper = sign * rng.uniform(8.0, 20.0)
        middle = upper / 2.0
    elif group == "shear":
        pad = sign * rng.uniform(4.0, 8.0)
        middle = -pad
        upper = -sign * rng.uniform(8.0, 20.0)
    elif group == "gust":
        pad = sign * rng.uniform(1.0, 8.0)
        middle = upper = pad
        gust = rng.uniform(0.5, 5.0)
        period = rng.uniform(8.0, 60.0)
        phase = rng.uniform(-np.pi, np.pi)
        gust_start = rng.uniform(30.0, 60.0)
        gust_duration = rng.uniform(60.0, 120.0)
        gust_center = rng.uniform(2000.0, 12000.0)
        gust_half_width = rng.uniform(2000.0, 5000.0)
        forecast_gust = rng.uniform(0.0, 5.0)
    elif group == "vertical":
        vertical = sign * rng.uniform(2.0, 3.0)
    elif group == "density":
        # These are separate, predeclared temperature-only and pressure-only
        # perturbations. The outer ±10% density-scale limits are explicit below.
        if index == 0:
            sea_level_temperature += sign * rng.uniform(0.5, 1.0)
        else:
            sea_level_pressure += sign * rng.uniform(400.0, 800.0)
    elif group == "vehicle":
        disturbances = RecoveryDisturbances(
            thrust_scale=float(rng.uniform(0.97, 1.03)),
            drag_coefficient_scale=float(rng.uniform(0.85, 1.15)),
            initial_mass_scale=float(rng.uniform(0.99, 1.01)),
            inertia_scale=float(rng.uniform(0.97, 1.03)),
            sensor_position_error_m=float(rng.uniform(1.0, 2.0)),
            sensor_velocity_error_m_s=float(rng.uniform(0.5, 1.0)),
            observation_delay_steps=int(rng.integers(0, 3)),
            action_delay_steps=int(rng.integers(0, 2)),
            engine_lag_s=float(rng.uniform(0.0, 0.05)),
        )
    elif group == "combined":
        pad = float(rng.uniform(-6.0, 6.0))
        upper = float(rng.uniform(-18.0, 18.0))
        middle = (pad + upper) / 2.0
        gust = float(rng.uniform(0.2, 5.0))
        period = float(rng.uniform(8.0, 60.0))
        phase = float(rng.uniform(-np.pi, np.pi))
        gust_start = float(rng.uniform(30.0, 60.0))
        gust_duration = float(rng.uniform(60.0, 120.0))
        gust_center = float(rng.uniform(2000.0, 12000.0))
        gust_half_width = float(rng.uniform(2000.0, 5000.0))
        forecast_gust = float(rng.uniform(0.0, 5.0))
        vertical = float(rng.uniform(-3.0, 3.0))
        # Small independent profile and scale perturbations fit the fixed-ISA
        # total-density band without clipping or discarding random draws.
        sea_level_temperature += float(rng.uniform(-0.75, 0.75))
        sea_level_pressure += float(rng.uniform(-400.0, 400.0))
        temperature_offset = sea_level_temperature - 288.15
        temperature_profile = tuple(
            (height, atmosphere(height, AtmosphereSettings())[0] + temperature_offset)
            for height in (
                0.0,
                11000.0,
                20000.0,
                32000.0,
                47000.0,
                51000.0,
                71000.0,
                84852.0,
                PROFILE_TOP_M,
            )
        )
        hydrostatic = AtmosphereSettings(
            sea_level_temperature_k=sea_level_temperature,
            sea_level_pressure_pa=sea_level_pressure,
            temperature_profile=temperature_profile,
        )
        pressure_profile = tuple(
            (height, atmosphere(height, hydrostatic)[1] * (1.0 + rng.uniform(-0.003, 0.003)))
            if height
            else (0.0, sea_level_pressure)
            for height, _ in temperature_profile
        )
        density_scale = float(rng.uniform(0.98, 1.02))
        forecast_offset = float(rng.uniform(-1.5, 1.5))
        latitude = float(rng.uniform(-60.0, 60.0))
        disturbances = RecoveryDisturbances(
            thrust_scale=float(rng.uniform(0.97, 1.03)),
            drag_coefficient_scale=float(rng.uniform(0.85, 1.15)),
            initial_mass_scale=float(rng.uniform(0.99, 1.01)),
            inertia_scale=float(rng.uniform(0.97, 1.03)),
            sensor_position_error_m=float(rng.uniform(1.0, 2.0)),
            sensor_velocity_error_m_s=float(rng.uniform(0.5, 1.0)),
            observation_delay_steps=int(rng.integers(0, 3)),
            action_delay_steps=int(rng.integers(0, 2)),
            engine_lag_s=float(rng.uniform(0.0, 0.05)),
        )
    else:
        raise ValueError(f"unknown operating group: {group}")
    # Predeclare late events rather than moving the event after observing a
    # trajectory. Exposure is measured from each actual flight below: a gust
    # input that the vehicle never meets is not evidence of gust rejection.
    if group in ("gust", "combined") and index % 4 >= 2:
        gust_start = float(rng.uniform(250.0, 280.0) if index % 4 == 2 else 310.0)
        gust_duration = 120.0
        gust_center = float(rng.uniform(18000.0, 22000.0) if index % 4 == 2 else 0.0)
        gust_half_width = 5000.0
    actual = _profile(float(pad), float(middle), float(upper))
    forecast = tuple((z, w + forecast_offset) for z, w in actual)
    conditions = AprConditions(
        wind_profile=actual,
        forecast_wind_profile=forecast,
        gust_amplitude_m_s=float(gust),
        gust_period_s=float(period),
        gust_phase_rad=float(phase),
        gust_start_s=float(gust_start),
        gust_duration_s=float(gust_duration),
        gust_altitude_center_m=float(gust_center),
        gust_altitude_half_width_m=float(gust_half_width),
        forecast_gust_amplitude_m_s=float(forecast_gust),
        vertical_wind_profile=((0.0, float(vertical)), (PROFILE_TOP_M, float(vertical))),
        density_scale=float(density_scale),
        sea_level_temperature_k=float(sea_level_temperature),
        sea_level_pressure_pa=float(sea_level_pressure),
        temperature_profile=temperature_profile,
        pressure_profile=pressure_profile,
        latitude_deg=latitude,
    )
    return _case(f"{group}-{stream}-{index:02d}", group, conditions, disturbances, stream, index)


def _boundary_conditions(
    *,
    pad=0.0,
    middle=0.0,
    upper=0.0,
    gust=0.0,
    phase=0.0,
    gust_center=10000.0,
    gust_start=0.0,
    gust_period=12.0,
    gust_duration=120.0,
    gust_half_width=2500.0,
    forecast_gust=0.0,
    vertical=0.0,
    vertical_profile=None,
    density_scale=1.0,
    sea_level_temperature=288.15,
    sea_level_pressure=101325.0,
    temperature_profile=(),
    pressure_profile=(),
):
    profile = _profile(pad, middle, upper)
    return AprConditions(
        wind_profile=profile,
        forecast_wind_profile=profile,
        gust_amplitude_m_s=gust,
        gust_phase_rad=phase,
        gust_start_s=gust_start,
        gust_period_s=gust_period,
        gust_duration_s=gust_duration,
        gust_altitude_center_m=gust_center,
        gust_altitude_half_width_m=gust_half_width,
        forecast_gust_amplitude_m_s=forecast_gust,
        vertical_wind_profile=(
            ((0.0, vertical), (PROFILE_TOP_M, vertical))
            if vertical_profile is None
            else vertical_profile
        ),
        density_scale=density_scale,
        sea_level_temperature_k=sea_level_temperature,
        sea_level_pressure_pa=sea_level_pressure,
        temperature_profile=temperature_profile,
        pressure_profile=pressure_profile,
    )


def coverage_metadata(development, holdout):
    """Describe sampled inputs, not an assertion about continuous robustness."""
    ranges = {
        "surface_wind_m_s": (-8.0, 8.0),
        "upper_wind_m_s": (-20.0, 20.0),
        "vertical_wind_m_s": (-3.0, 3.0),
        "horizontal_wind_shear_m_s_km": (-4.0, 4.0),
        "vertical_wind_shear_m_s_km": (-3.0, 3.0),
        "forecast_error_m_s": (-2.0, 2.0),
        "gust_amplitude_m_s": (0.0, 5.0),
        "gust_period_s": (8.0, 60.0),
        "gust_phase_rad": (-np.pi, np.pi),
        "gust_start_s": (-120.0, 600.0),
        "gust_duration_s": (8.0, 120.0),
        "gust_altitude_center_m": (0.0, PROFILE_TOP_M),
        "gust_altitude_half_width_m": (100.0, 5000.0),
        "forecast_gust_amplitude_m_s": (0.0, 5.0),
        "density_scale": (0.9, 1.1),
        "sea_level_temperature_k": (273.15, 308.15),
        "sea_level_pressure_pa": (95000.0, 105000.0),
        "latitude_deg": (-90.0, 90.0),
        "thrust_scale": (0.97, 1.03),
        "drag_coefficient_scale": (0.85, 1.15),
        "initial_mass_scale": (0.99, 1.01),
        "inertia_scale": (0.97, 1.03),
        "sensor_position_error_m": (0.0, 2.0),
        "sensor_velocity_error_m_s": (0.0, 1.0),
        "observation_delay_steps": (0, 2),
        "action_delay_steps": (0, 1),
        "engine_lag_s": (0.0, 0.05),
    }

    def values(cases, name):
        if name.endswith("wind_shear_m_s_km"):
            field = "vertical_wind_profile" if name.startswith("vertical") else "wind_profile"
            return [
                1000.0 * (w1 - w0) / (z1 - z0)
                for case in cases
                for (z0, w0), (z1, w1) in zip(
                    case["conditions"][field], case["conditions"][field][1:], strict=False
                )
            ]
        if name == "forecast_error_m_s":
            errors = []
            for case in cases:
                actual = np.asarray(case["conditions"]["wind_profile"])
                forecast = np.asarray(case["conditions"]["forecast_wind_profile"])
                heights = np.union1d(actual[:, 0], forecast[:, 0])
                errors.extend(
                    (
                        np.interp(heights, forecast[:, 0], forecast[:, 1])
                        - np.interp(heights, actual[:, 0], actual[:, 1])
                    ).tolist()
                )
            return errors
        if name in ("surface_wind_m_s", "upper_wind_m_s", "vertical_wind_m_s"):
            field = "vertical_wind_profile" if name == "vertical_wind_m_s" else "wind_profile"
            return [
                wind
                for case in cases
                for height, wind in case["conditions"][field]
                if name != "surface_wind_m_s" or height <= 50.0
            ]
        return [
            case["conditions"][name] if name in case["conditions"] else case["disturbances"][name]
            for case in cases
        ]

    return {
        "kind": "finite_predeclared_cases",
        "continuous_envelope_proven": False,
        "calibrated_weather_probability": False,
        "real_vehicle_safety_certified": False,
        "parameters": {
            name: {
                "configured_range": bounds,
                "development_sampled_range": [
                    min(values(development, name)),
                    max(values(development, name)),
                ],
                "holdout_sampled_range": [min(values(holdout, name)), max(values(holdout, name))],
            }
            for name, bounds in ranges.items()
        },
        "coupled_constraints": [
            "Final density at every altitude remains 0.9-1.1 of fixed standard atmosphere",
            "Horizontal wind shear <=4 m/s/km; vertical wind shear <=3 m/s/km",
            "Surface mean <=8 m/s and gust contribution <=5 m/s; not 13 m/s sustained",
            "Mean-wind forecast error <=2 m/s; instantaneous gust is not supplied to guidance",
            "Temperature and pressure profiles must satisfy the shared input validator",
        ],
        "limitations": [
            "Finite cases do not cover all combinations, gust times, profiles or noise streams",
            "Recorded gust exposure must be checked before claiming disturbance rejection",
            "Lightning, rain and known faults are eligibility exclusions, not simulated loads",
            "Nominal assumed system parameters are not experimentally calibrated hardware",
        ],
    }


def _extended_development_cases():
    """Predeclare additional coupled boundaries without replacing prior inputs.

    Event coordinates are fixed before running these cases. They bracket the
    existing nominal trajectory, but every actual gust encounter must still be
    measured: a named phase is not proof that the vehicle met that disturbance.
    """
    entries = []
    reserve_corner = RecoveryDisturbances(
        thrust_scale=0.97,
        drag_coefficient_scale=1.15,
        initial_mass_scale=1.01,
        inertia_scale=1.03,
        sensor_position_error_m=2.0,
        sensor_velocity_error_m_s=1.0,
        observation_delay_steps=2,
        action_delay_steps=1,
        engine_lag_s=0.05,
    )
    load_corner = replace(
        reserve_corner, thrust_scale=1.03, initial_mass_scale=0.99, inertia_scale=0.97
    )
    for label, sign in (("plus", 1.0), ("minus", -1.0)):
        for phase_name, start, center, duration in (
            ("boostback", 125.0, 70000.0, 30.0),
            ("coast-high", 180.0, 85000.0, 120.0),
        ):
            entries.append(
                (
                    f"gust-{phase_name}-{label}",
                    "late_gust",
                    _boundary_conditions(
                        gust=5.0,
                        phase=sign * np.pi / 2,
                        gust_start=start,
                        gust_center=center,
                        gust_half_width=5000.0,
                        gust_duration=duration,
                    ),
                    RecoveryDisturbances(),
                )
            )
        for event, start, center, duration, width, peak_time in (
            ("short-entry", 319.0, 20000.0, 8.0, 5000.0, 324.8),
            ("short-landing", 398.0, 0.0, 8.0, 5000.0, 400.0),
            ("thin-entry", 318.0, 20000.0, 30.0, 100.0, 324.8),
            ("thin-landing", 390.0, 125.0, 60.0, 100.0, 400.0),
        ):
            phase = (sign * np.pi / 2 - 2 * np.pi * peak_time / 8 + np.pi) % (2 * np.pi) - np.pi
            entries.append(
                (
                    f"gust-{event}-{label}",
                    "late_gust",
                    _boundary_conditions(
                        gust=5.0,
                        phase=float(phase),
                        gust_start=start,
                        gust_center=center,
                        gust_period=8.0,
                        gust_half_width=width,
                        gust_duration=duration,
                    ),
                    RecoveryDisturbances(),
                )
            )
        for corner_name, deviations in (("reserve", reserve_corner), ("load", load_corner)):
            for period_name, period in (("fast", 8.0), ("slow", 60.0)):
                entries.append(
                    (
                        f"late-corner-{corner_name}-{period_name}-{label}",
                        "combined",
                        _boundary_conditions(
                            pad=sign * 8.0,
                            middle=sign * 14.0,
                            upper=sign * 20.0,
                            gust=5.0,
                            phase=sign * np.pi / 2,
                            gust_start=310.0,
                            gust_center=0.0,
                            gust_period=period,
                            gust_half_width=5000.0,
                            vertical=sign * 3.0,
                            density_scale=1.1,
                        ),
                        deviations,
                    )
                )
        for altitude_label, profile in (
            (
                "low",
                ((0.0, 8.0), (50.0, 8.0), (4050.0, -8.0), (7050.0, -20.0)),
            ),
            (
                "upper",
                ((0.0, 0.0), (50.0, 0.0), (5050.0, 20.0), (15050.0, -20.0)),
            ),
        ):
            wind = tuple((height, sign * speed) for height, speed in profile) + (
                (PROFILE_TOP_M, -sign * 20.0),
            )
            conditions = AprConditions(
                wind_profile=wind,
                forecast_wind_profile=wind,
                vertical_wind_profile=(
                    (0.0, -sign * 3.0),
                    (2000.0, sign * 3.0),
                    (PROFILE_TOP_M, sign * 3.0),
                ),
            )
            entries.append(
                (f"shear-reversal-{altitude_label}-{label}", "shear", conditions, reserve_corner)
            )
    standard = AtmosphereSettings()
    levels = (
        0.0,
        2000.0,
        11000.0,
        20000.0,
        32000.0,
        47000.0,
        51000.0,
        71000.0,
        84852.0,
        PROFILE_TOP_M,
    )
    for thermal_label, temperature, pressure, density_target in (
        ("cold-low", 273.15, 95000.0, 0.9),
        ("cold-high", 273.15, 105000.0, 1.1),
        ("warm-low", 308.15, 95000.0, 0.9),
        ("warm-high", 308.15, 105000.0, 1.1),
    ):
        profile = tuple(
            (height, temperature if height == 0.0 else atmosphere(height, standard)[0])
            for height in levels
        )
        minimum, maximum = _density_ratio_extrema(
            AtmosphereSettings(
                sea_level_temperature_k=temperature,
                sea_level_pressure_pa=pressure,
                temperature_profile=profile,
            )
        )
        # The input range constrains both the scale and final density; choose a
        # valid coupled boundary instead of forcing an invalid Cartesian corner.
        scale = density_target / (minimum if density_target < 1.0 else maximum)
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            entries.append(
                (
                    f"thermal-corner-{thermal_label}-{label}",
                    "combined",
                    _boundary_conditions(
                        pad=sign * 8.0,
                        middle=sign * 14.0,
                        upper=sign * 20.0,
                        vertical=sign * 3.0,
                        density_scale=scale,
                        sea_level_temperature=temperature,
                        sea_level_pressure=pressure,
                        temperature_profile=profile,
                    ),
                    reserve_corner if density_target < 1.0 else load_corner,
                )
            )
    return [
        _case(name, group, conditions, deviations, 0, 2000 + index)
        for index, (name, group, conditions, deviations) in enumerate(entries)
    ]


def _touchdown_development_cases(development):
    """Add fixed-window latency and final-approach cases; preserve earlier inputs."""
    originals = {case["id"]: case for case in development}
    reserve = RecoveryDisturbances(**originals["late-corner-reserve-fast-plus"]["disturbances"])
    entries = []
    for label in ("plus", "minus"):
        for event in ("short-entry", "short-landing", "thin-entry", "thin-landing"):
            original = originals[f"gust-{event}-{label}"]
            entries.append(
                (
                    f"gust-{event}-delayed-{label}",
                    "combined",
                    AprConditions(**original["conditions"]),
                    reserve,
                )
            )
        for corner in ("reserve", "load"):
            for speed in ("fast", "slow"):
                original = originals[f"late-corner-{corner}-{speed}-{label}"]
                conditions = replace(
                    AprConditions(**original["conditions"]),
                    gust_start_s=370.0,
                    gust_duration_s=120.0,
                )
                entries.append(
                    (
                        f"touchdown-corner-{corner}-{speed}-{label}",
                        "combined",
                        conditions,
                        RecoveryDisturbances(**original["disturbances"]),
                    )
                )
    return [
        _case(name, group, conditions, deviations, 0, 3000 + index)
        for index, (name, group, conditions, deviations) in enumerate(entries)
    ]


def _late_holdout_cases():
    """Independent healthy/weather draws with predeclared late finite gusts.

    The new stream is separate from all 64 earlier holdout inputs. Windows are
    fixed from the declared input distribution, never moved after seeing a
    flight or holdout result. Actual terminal gust exposure is still measured.
    """
    cases = []
    for index in range(8):
        original = _random_case("combined", 2, index)
        event_rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, 2, index, 370]))
        conditions = replace(
            AprConditions(**original["conditions"]),
            gust_start_s=float(event_rng.uniform(370.0, 390.0)),
            gust_duration_s=120.0,
            gust_altitude_center_m=0.0,
            gust_altitude_half_width_m=5000.0,
        )
        deviations = replace(
            RecoveryDisturbances(**original["disturbances"]),
            observation_delay_steps=2,
            action_delay_steps=1,
            engine_lag_s=0.05,
        )
        cases.append(
            _case(f"late-combined-2-{index:02d}", "combined", conditions, deviations, 2, index)
        )
    return cases


def make_manifest():
    development = [_case("calm", "calm", AprConditions(), RecoveryDisturbances(), 0, 0)]
    example_conditions, example_deviations = load_apr_operating(
        ROOT / "configs/apr-operating-v1.yaml"
    )
    development.append(
        _case("example-weather", "combined", example_conditions, example_deviations, 0, 1)
    )
    for group in GROUPS:
        development.extend(_random_case(group, 0, index) for index in range(2))
    standard = AtmosphereSettings()
    pressure_profile = (
        (0.0, 101325.0),
        (11000.0, atmosphere(11000.0, standard)[1] * 1.005),
        (PROFILE_TOP_M, atmosphere(PROFILE_TOP_M, standard)[1] * 1.002),
    )
    boundary = (
        (
            "surface-plus",
            _boundary_conditions(
                pad=8.0,
                middle=8.0,
                upper=8.0,
                gust=5.0,
                phase=np.pi / 2,
                gust_center=0.0,
                gust_start=-60.0,
                forecast_gust=5.0,
            ),
            RecoveryDisturbances(),
        ),
        (
            "surface-minus",
            _boundary_conditions(
                pad=-8.0,
                middle=-8.0,
                upper=-8.0,
                gust=5.0,
                phase=-np.pi / 2,
                gust_center=0.0,
                gust_start=-60.0,
            ),
            RecoveryDisturbances(),
        ),
        (
            "upper-plus",
            _boundary_conditions(
                middle=10.0, upper=20.0, gust=5.0, phase=np.pi / 2, forecast_gust=5.0
            ),
            RecoveryDisturbances(),
        ),
        (
            "upper-minus",
            _boundary_conditions(middle=-10.0, upper=-20.0, gust=5.0, phase=-np.pi / 2),
            RecoveryDisturbances(),
        ),
        (
            "shear-plus",
            _boundary_conditions(middle=19.8, upper=20.0, forecast_gust=5.0),
            RecoveryDisturbances(),
        ),
        (
            "vertical-plus-inner",
            _boundary_conditions(
                vertical_profile=((0.0, -1.0), (1000.0, 2.0), (PROFILE_TOP_M, 2.0))
            ),
            RecoveryDisturbances(),
        ),
        (
            "vertical-minus-inner",
            _boundary_conditions(
                vertical_profile=((0.0, 1.0), (1000.0, -2.0), (PROFILE_TOP_M, -2.0))
            ),
            RecoveryDisturbances(),
        ),
        ("vertical-plus-outer", _boundary_conditions(vertical=3.0), RecoveryDisturbances()),
        ("vertical-minus-outer", _boundary_conditions(vertical=-3.0), RecoveryDisturbances()),
        ("density-low", _boundary_conditions(density_scale=0.9), RecoveryDisturbances()),
        ("density-high", _boundary_conditions(density_scale=1.1), RecoveryDisturbances()),
        (
            "temperature-warm",
            _boundary_conditions(sea_level_temperature=289.15),
            RecoveryDisturbances(),
        ),
        (
            "temperature-cool",
            _boundary_conditions(sea_level_temperature=287.15),
            RecoveryDisturbances(),
        ),
        (
            "pressure-high",
            _boundary_conditions(sea_level_pressure=102000.0),
            RecoveryDisturbances(),
        ),
        ("pressure-low", _boundary_conditions(sea_level_pressure=100500.0), RecoveryDisturbances()),
        (
            "temperature-profile",
            _boundary_conditions(
                temperature_profile=tuple(
                    (height, atmosphere(height, standard)[0] + (1.0 if height else 0.0))
                    for height in (
                        0.0,
                        11000.0,
                        20000.0,
                        32000.0,
                        47000.0,
                        51000.0,
                        71000.0,
                        84852.0,
                        PROFILE_TOP_M,
                    )
                )
            ),
            RecoveryDisturbances(),
        ),
        (
            "pressure-profile",
            _boundary_conditions(pressure_profile=pressure_profile),
            RecoveryDisturbances(),
        ),
        (
            "drag-minus-inner",
            _boundary_conditions(),
            RecoveryDisturbances(drag_coefficient_scale=0.9),
        ),
        (
            "drag-plus-inner",
            _boundary_conditions(),
            RecoveryDisturbances(drag_coefficient_scale=1.1),
        ),
        (
            "drag-minus-outer",
            _boundary_conditions(),
            RecoveryDisturbances(drag_coefficient_scale=0.85),
        ),
        (
            "drag-plus-outer",
            _boundary_conditions(),
            RecoveryDisturbances(drag_coefficient_scale=1.15),
        ),
        ("thrust-minus-inner", _boundary_conditions(), RecoveryDisturbances(thrust_scale=0.98)),
        ("thrust-plus-inner", _boundary_conditions(), RecoveryDisturbances(thrust_scale=1.02)),
        ("thrust-minus-outer", _boundary_conditions(), RecoveryDisturbances(thrust_scale=0.97)),
        ("thrust-plus-outer", _boundary_conditions(), RecoveryDisturbances(thrust_scale=1.03)),
        ("mass-minus", _boundary_conditions(), RecoveryDisturbances(initial_mass_scale=0.99)),
        ("mass-plus", _boundary_conditions(), RecoveryDisturbances(initial_mass_scale=1.01)),
        (
            "sensor-inner",
            _boundary_conditions(),
            RecoveryDisturbances(sensor_position_error_m=1.0, sensor_velocity_error_m_s=0.5),
        ),
        (
            "sensor-outer",
            _boundary_conditions(),
            RecoveryDisturbances(sensor_position_error_m=2.0, sensor_velocity_error_m_s=1.0),
        ),
    )
    for index, (name, conditions, disturbances) in enumerate(boundary):
        development.append(
            _case(
                name,
                "boundary",
                conditions,
                disturbances,
                0,
                index + 2,
            )
        )
    hidden_gust = _boundary_conditions(middle=-10.0, upper=-20.0, gust=5.0, phase=-np.pi / 2)
    for index, (actual_upper, forecast_upper) in enumerate(
        ((-19.0, -17.0), (-18.0, -16.0), (-14.0, -12.0), (-10.0, -8.0), (-6.0, -4.0), (0.0, 0.0))
    ):
        conditions = replace(
            hidden_gust,
            wind_profile=tuple(
                (z, actual_upper if z >= 10000.0 else wind) for z, wind in hidden_gust.wind_profile
            ),
            forecast_wind_profile=tuple(
                (z, forecast_upper if z >= 10000.0 else wind)
                for z, wind in hidden_gust.wind_profile
            ),
        )
        development.append(
            _case(
                f"hidden-gust-upper-{abs(int(actual_upper)):02d}",
                "boundary",
                conditions,
                RecoveryDisturbances(),
                0,
                len(boundary) + 2 + index,
            )
        )
    forecast_base = AprConditions(
        **next(case["conditions"] for case in development if case["id"] == "hidden-gust-upper-06")
    )
    for index, (label, middle, upper) in enumerate(
        (
            ("00", -8.0, -8.0),
            ("05", -8.8333333333, -7.1666666667),
            ("10", -9.6666666667, -6.3333333333),
            ("15", -10.5, -5.5),
        )
    ):
        conditions = replace(
            forecast_base,
            forecast_wind_profile=tuple(
                (z, middle if z == 5000.0 else upper if z >= 10000.0 else wind)
                for z, wind in forecast_base.forecast_wind_profile
            ),
        )
        development.append(
            _case(
                f"forecast-rise-{label}",
                "boundary",
                conditions,
                RecoveryDisturbances(),
                0,
                len(boundary) + 8 + index,
            )
        )
    for index, (name, sign) in enumerate((("corner-plus", 1.0), ("corner-minus", -1.0))):
        development.append(
            _case(
                name,
                "boundary",
                _boundary_conditions(
                    pad=sign * 8.0,
                    middle=sign * 14.0,
                    upper=sign * 20.0,
                    gust=5.0,
                    phase=sign * np.pi / 2,
                    vertical=sign * 3.0,
                    density_scale=1.1,
                ),
                RecoveryDisturbances(
                    thrust_scale=0.97,
                    drag_coefficient_scale=1.15,
                    initial_mass_scale=1.01,
                    inertia_scale=1.03,
                    sensor_position_error_m=2.0,
                    sensor_velocity_error_m_s=1.0,
                    observation_delay_steps=2,
                    action_delay_steps=1,
                    engine_lag_s=0.05,
                ),
                0,
                900 + index,
            )
        )
    # The fuel-limited corners above do not cover the opposite maximum-load
    # combination: a lighter airframe and a stronger engine in dense air.
    for index, (label, sign) in enumerate((("plus", 1.0), ("minus", -1.0))):
        development.append(
            _case(
                f"corner-load-{label}",
                "boundary",
                _boundary_conditions(
                    pad=sign * 8.0,
                    middle=sign * 14.0,
                    upper=sign * 20.0,
                    gust=5.0,
                    phase=sign * np.pi / 2,
                    vertical=sign * 3.0,
                    density_scale=1.1,
                ),
                RecoveryDisturbances(
                    thrust_scale=1.03,
                    drag_coefficient_scale=1.15,
                    initial_mass_scale=0.99,
                    inertia_scale=0.97,
                    sensor_position_error_m=2.0,
                    sensor_velocity_error_m_s=1.0,
                    observation_delay_steps=2,
                    action_delay_steps=1,
                    engine_lag_s=0.05,
                ),
                0,
                902 + index,
            )
        )
    # Individual healthy-system boundary inputs complement the combined cases.
    additional = [
        ("shear-minus", _boundary_conditions(middle=-19.8, upper=-20.0), RecoveryDisturbances()),
        ("inertia-minus", AprConditions(), RecoveryDisturbances(inertia_scale=0.97)),
        ("inertia-plus", AprConditions(), RecoveryDisturbances(inertia_scale=1.03)),
        (
            "sensor-errors-combined",
            AprConditions(),
            RecoveryDisturbances(sensor_position_error_m=2.0, sensor_velocity_error_m_s=1.0),
        ),
        ("observation-delay", AprConditions(), RecoveryDisturbances(observation_delay_steps=2)),
        ("action-delay", AprConditions(), RecoveryDisturbances(action_delay_steps=1)),
        ("engine-lag", AprConditions(), RecoveryDisturbances(engine_lag_s=0.05)),
        ("pressure-minimum", AprConditions(sea_level_pressure_pa=95000.0), RecoveryDisturbances()),
        ("pressure-maximum", AprConditions(sea_level_pressure_pa=105000.0), RecoveryDisturbances()),
    ]
    for label, temperature in (("minimum", 273.15), ("maximum", 308.15)):
        levels = (
            0.0,
            2000.0,
            11000.0,
            20000.0,
            32000.0,
            47000.0,
            51000.0,
            71000.0,
            84852.0,
            PROFILE_TOP_M,
        )
        additional.append(
            (
                f"temperature-{label}",
                AprConditions(
                    sea_level_temperature_k=temperature,
                    temperature_profile=tuple(
                        (height, temperature if height == 0.0 else atmosphere(height, standard)[0])
                        for height in levels
                    ),
                ),
                RecoveryDisturbances(),
            )
        )
    for label, latitude in (("south-pole", -90.0), ("equator", 0.0), ("north-pole", 90.0)):
        additional.append((label, AprConditions(latitude_deg=latitude), RecoveryDisturbances()))
    for sign, label in ((1.0, "plus"), (-1.0, "minus")):
        for phase_name, start, center, period in (
            ("coast", 100.0, 55000.0, 12.0),
            ("entry", 260.0, 20000.0, 12.0),
            ("landing-fast", 310.0, 0.0, 8.0),
            ("landing-slow", 310.0, 0.0, 60.0),
        ):
            additional.append(
                (
                    f"gust-{phase_name}-{label}",
                    _boundary_conditions(
                        gust=5.0,
                        phase=sign * np.pi / 2,
                        gust_start=start,
                        gust_center=center,
                        gust_half_width=5000.0,
                        gust_period=period,
                    ),
                    RecoveryDisturbances(),
                )
            )
        additional.append(
            (
                f"forecast-bias-{label}",
                AprConditions(
                    forecast_wind_profile=((0.0, sign * 2.0), (PROFILE_TOP_M, sign * 2.0))
                ),
                RecoveryDisturbances(),
            )
        )
    for index, (name, conditions, disturbances) in enumerate(additional):
        group = "late_gust" if name.startswith("gust-") else "boundary"
        development.append(_case(name, group, conditions, disturbances, 0, 1000 + index))
    development.extend(_extended_development_cases())
    development.extend(_touchdown_development_cases(development))
    # Preserve the original 32 inputs and their order before adding new streams.
    holdout = [_random_case(group, 1, index) for group in GROUPS for index in range(4)]
    holdout.extend(_random_case(group, 1, index) for group in GROUPS for index in range(4, 8))
    holdout.extend(_late_holdout_cases())
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "root_seed": ROOT_SEED,
        "vehicle": VEHICLE_ID,
        "operating_scope": OPERATING_SCOPE,
        "nominal_vehicle": asdict(Design()),
        "development": development,
        "holdout": holdout,
        "sampling_note": "Synthetic project envelope; not a calibrated launch-weather distribution",
        "coverage": coverage_metadata(development, holdout),
    }
    manifest["manifest_sha256"] = _sha(manifest)
    return manifest


def _load_manifest(campaign_dir):
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("root_seed") != ROOT_SEED
        or manifest.get("vehicle") != VEHICLE_ID
        or manifest.get("operating_scope") != OPERATING_SCOPE
        or manifest.get("nominal_vehicle") != asdict(Design())
    ):
        raise ValueError("incompatible evaluation manifest")
    if digest != _sha(manifest):
        raise ValueError("evaluation manifest has changed")
    manifest["manifest_sha256"] = digest
    return manifest


def prepare(campaign_dir):
    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest()
    (campaign_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {len(manifest['development'])} development and "
        f"{len(manifest['holdout'])} predeclared holdout inputs; root seed {ROOT_SEED}."
    )


def freeze(campaign_dir, development_tag):
    manifest = _load_manifest(campaign_dir)
    report = campaign_dir / f"{development_tag}-summary.json"
    flights = campaign_dir / f"{development_tag}.jsonl"
    if not report.is_file():
        raise ValueError("development summary is required before freezing")
    if not flights.is_file():
        raise ValueError("development flight records are required before freezing")
    summary = json.loads(report.read_text(encoding="utf-8"))
    if summary["split"] != "development" or summary["total"] != len(manifest["development"]):
        raise ValueError("development sweep is incomplete")
    if summary["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("development summary uses a different manifest")
    if summary["source_hashes"] != _source_hashes():
        raise ValueError("source changed since the development sweep")
    entries = [json.loads(line) for line in flights.read_text(encoding="utf-8").splitlines()]
    if not entries or entries[0] != {
        "record_type": "protocol",
        "split": "development",
        "root_seed": ROOT_SEED,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_hashes": _source_hashes(),
        "case_count": len(manifest["development"]),
    }:
        raise ValueError("development flight protocol does not match the manifest")
    flights_by_case = entries[1:]
    if len(flights_by_case) != len(manifest["development"]) or any(
        entry.get("record_type") != "flight"
        or entry.get("id") != case["id"]
        or entry.get("input_sha256") != _sha(case)
        for entry, case in zip(flights_by_case, manifest["development"], strict=True)
    ):
        raise ValueError("development flight records are incomplete or use different inputs")
    if summary["accepted"] != sum(entry["accepted"] is True for entry in flights_by_case):
        raise ValueError("development summary does not match its flight records")
    frozen = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_hashes": _source_hashes(),
        "development_tag": development_tag,
        "development_report_sha256": hashlib.sha256(flights.read_bytes()).hexdigest(),
        "development_summary_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    path = campaign_dir / "freeze.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2)
        stream.write("\n")
    print(f"Frozen source and predeclared evaluation inputs: {path}")


def _stage(result):
    """Classify reached milestones, not the origin of a peak safety violation."""
    if result["separation"] is None:
        return "ascent_or_separation"
    if result["contact"] is None:
        return "return_or_approach"
    return "touchdown"


def gust_exposure(conditions, steps):
    """Report sampled encounter evidence; the recorded trajectory is not every RK stage."""
    peaks = {}
    encounter = []
    for step in steps:
        height = step["state"][1]
        time_s = step["time_s"]
        gust = abs(conditions.wind(height, time_s) - conditions.mean_wind(height))
        peaks[step["phase"]] = max(peaks.get(step["phase"], 0.0), gust)
        if gust >= 0.1:
            encounter.append(time_s)
    return {
        "configured_amplitude_m_s": conditions.gust_amplitude_m_s,
        "maximum_sampled_contribution_m_s": max(peaks.values(), default=0.0),
        "maximum_sampled_by_phase_m_s": peaks,
        "sample_count_above_0_1_m_s": len(encounter),
        "first_sample_above_0_1_m_s": min(encounter) if encounter else None,
        "last_sample_above_0_1_m_s": max(encounter) if encounter else None,
        "gust_encounter_observed": bool(encounter),
        "sampling_note": "Lower bound on peak encounter from recorded states, not every RK stage",
    }


def failure_evidence(result):
    """Separate measured violations from causal hypotheses and final termination."""
    evidence = []
    for name, passed in result["criteria"].items():
        if passed is not False:
            continue
        item = {"criterion": name, "kind": "observed_acceptance_failure"}
        if name.startswith("ascent_"):
            item["phase"] = "ascent"
        elif name == "angular_acceleration" and result.get("peak_angular_acceleration"):
            item["peak_record"] = result["peak_angular_acceleration"]
            item["phase"] = item["peak_record"]["phase"]
        elif name in (
            "touchdown",
            "contact_load_g",
            "leg_force",
            "leg_stroke",
            "engine_off_standing",
        ):
            item["phase"] = "contact" if result["contact"] is not None else "not_reached"
        elif name == "separation":
            item["phase"] = "separation" if result["separation"] is not None else "not_reached"
        else:
            item["phase"] = "not_localized_by_scalar_peak"
        evidence.append(item)
    if result["contact"] is not None:
        evidence.extend(
            {
                "criterion": f"touchdown.{name}",
                "kind": "observed_acceptance_failure",
                "phase": "first_contact",
            }
            for name, passed in result["contact"]["criteria"].items()
            if passed is False
        )
    return evidence


def _evaluate_case(case):
    conditions = AprConditions(**case["conditions"])
    disturbances = RecoveryDisturbances(**case["disturbances"])
    flight = AprFlight(conditions=conditions, disturbances=disturbances)
    flight.rng = np.random.default_rng(observation_seed_sequence(case))
    result = flight.run()
    mission = result["mission_success"] is True
    safety = result["flight_safety"] is True
    landing = result["landing_success"] is True
    success = result["simulated_acceptance_passed"] is True
    if success != (mission and safety and landing):
        raise ValueError("flight acceptance contradicts mission, safety or landing evidence")
    if success and (result["separation"] is None or result["contact"] is None):
        raise ValueError("successful full flight requires recorded separation and contact")

    separation = result["separation"]
    contact = result["contact"]
    failed_criteria = [name for name, passed in result["criteria"].items() if passed is False]
    exposure = gust_exposure(conditions, result.get("steps", []))
    integration_exposure = result.get("integration_gust_exposure")
    if integration_exposure is not None:
        exposure["integration_endpoints"] = integration_exposure
        exposure["gust_encounter_observed"] = integration_exposure["gust_encounter_observed"]
    entry = {
        "record_type": "flight",
        "id": case["id"],
        "group": case["group"],
        "input_sha256": _sha(case),
        "input": case,
        "vehicle": VEHICLE_ID,
        "operating_scope": OPERATING_SCOPE,
        "campaign_protocol_schema_version": MANIFEST_SCHEMA_VERSION,
        "outcome": result["outcome"],
        "fin_mode": result["fin_mode"],
        "nominal_launch_mass_kg": Design().launch_mass_kg,
        "actual_launch_mass_kg": flight.design.launch_mass_kg,
        "accepted": success,
        "mission_success": mission,
        "flight_safety": safety,
        "landing_success": landing,
        "failed_criteria": failed_criteria,
        "failure_reasons": result["failures"],
        "failure_stage": None if success else _stage(result),
        "failure_stage_note": "Last reached milestone; see failure_evidence for peak violations",
        "failure_evidence": failure_evidence(result),
        "causal_diagnosis_confirmed": False,
        "gust_exposure": exposure,
        "unconnected_systems": result["unconnected_systems"],
        "separation": None
        if separation is None
        else {
            "time_s": separation["time_s"],
            "state": separation["state"][:8],
            "checks_before_separation": separation["checks_before_separation"],
        },
        "contact": None
        if contact is None
        else {
            "time_s": contact["time_s"],
            "criteria": contact["criteria"],
            "state": contact["state"][:8],
            "foot_velocities_m_s": contact["foot_velocities_m_s"],
            "observed_control_state": contact.get("observed_control_state"),
            "navigation_acceleration_bias_m_s2": contact.get("navigation_acceleration_bias_m_s2"),
            "angular_torque_bias_nm": contact.get("angular_torque_bias_nm"),
        },
        "final_state": flight.state[:8].tolist(),
        "final_phase": flight.phase,
    }
    if not success:
        approach = [
            step
            for step in result.get("steps", [])
            if step["phase"] in ("coast", "entry", "landing") and step["state"][1] <= 3500.0
        ]
        sampled = []
        for step in approach:
            if not sampled or step["time_s"] >= sampled[-1]["time_s"] + 2.0 - 1e-7:
                sampled.append(step)
        if approach and (not sampled or approach[-1]["time_s"] != sampled[-1]["time_s"]):
            sampled.append(approach[-1])
        trace_fields = (
            "time_s",
            "phase",
            "state",
            "control",
            "requested_control",
            "observed_control_state",
            "navigation_acceleration_bias_m_s2",
            "angular_torque_bias_nm",
            "engine_throttle",
            "dynamic_pressure_pa",
            "attitude_target_deg",
            "fin_torque_nm",
        )
        entry["failure_approach_trace"] = [
            {name: step[name] for name in trace_fields if name in step} for step in sampled
        ]
        entry["failure_approach_trace_sampling"] = (
            "Recorded coast/entry/landing states below 3500 m COM altitude, "
            "approximately 2 s apart plus the last recorded approach state; "
            "diagnostic trace only, not causal proof or additional integration"
        )
    for key in (
        "landing_assessment",
        "acceptance_version",
        "assessment_scope",
        "system_metrics",
        "systems",
        "systems_settings",
        "systems_diagnostics",
        "systems_model_version",
        "subsystem_design_budget",
        "validation_status",
        "model_validation",
        "peak_angular_acceleration",
        "return_plan",
        "return_plan_history",
    ):
        if key in result:
            entry[key] = result[key]
    metrics = {
        "remaining_main_propellant_kg": "main_propellant_remaining_kg",
        "remaining_rcs_propellant_kg": "rcs_propellant_remaining_kg",
        "main_propellant_used_kg": "main_propellant_used_kg",
        "rcs_propellant_used_kg": "rcs_propellant_used_kg",
        "flight_time_s": "time_s",
        "max_dynamic_pressure_pa": "max_dynamic_pressure_pa",
        "max_recovery_dynamic_pressure_pa": "max_recovery_dynamic_pressure_pa",
        "max_ascent_q_alpha_kpa_deg": "max_ascent_q_alpha_kpa_deg",
        "max_angular_rate_deg_s": "max_angular_rate_deg_s",
        "max_angular_acceleration_deg_s2": "max_angular_acceleration_deg_s2",
        "max_flight_load_g": "max_flight_load_g",
        "max_gimbal_deg": "max_gimbal_deg",
        "max_fin_hinge_nm": "max_fin_hinge_nm",
        "max_mass_balance_error_kg": "max_mass_balance_error_kg",
    }
    for name, source_name in metrics.items():
        entry[name] = result[source_name]
    return entry


def _evaluate_cases(cases, workers):
    if workers == 1:
        yield from map(_evaluate_case, cases)
    else:
        # map yields in manifest order and propagates every worker failure.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(_evaluate_case, cases)


def evaluate(campaign_dir, split, tag, workers=1, *, case_ids=None, groups=None):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    manifest = _load_manifest(campaign_dir)
    source_hashes = _source_hashes()
    if split == "holdout":
        if case_ids or groups:
            raise ValueError(
                "holdout must run in full; case and group filters are development-only"
            )
        frozen = json.loads((campaign_dir / "freeze.json").read_text(encoding="utf-8"))
        if frozen["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("holdout manifest differs from frozen input")
        if frozen["source_hashes"] != source_hashes:
            raise ValueError("source changed after freezing the holdout")
        development_tag = frozen["development_tag"]
        if (
            frozen["development_report_sha256"]
            != hashlib.sha256((campaign_dir / f"{development_tag}.jsonl").read_bytes()).hexdigest()
            or frozen["development_summary_sha256"]
            != hashlib.sha256(
                (campaign_dir / f"{development_tag}-summary.json").read_bytes()
            ).hexdigest()
        ):
            raise ValueError("frozen development evidence has changed")
        if tag != "holdout":
            raise ValueError("holdout has one fixed output tag")
    cases = manifest[split]
    if case_ids:
        unknown = set(case_ids) - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in set(case_ids)]
    if groups:
        unknown = set(groups) - {case["group"] for case in manifest[split]}
        if unknown:
            raise ValueError(f"unknown case groups: {sorted(unknown)}")
        cases = [case for case in cases if case["group"] in set(groups)]
    if not cases:
        raise ValueError("case selection is empty")
    report_path = campaign_dir / f"{tag}.jsonl"
    summary_path = campaign_dir / f"{tag}-summary.json"
    if report_path.exists() or summary_path.exists():
        raise FileExistsError("evaluation output already exists; preserve first-run evidence")
    header = {
        "record_type": "protocol",
        "split": split,
        "root_seed": ROOT_SEED,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_hashes": source_hashes,
        "case_count": len(cases),
    }
    counts = Counter()
    reasons = Counter()
    failures_by_stage = Counter()
    outcomes = Counter()
    group_reasons = {}
    encountered_gusts = []
    unencountered_gusts = []
    remaining_main = []
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(_encoded(header) + "\n")
        for completed, entry in enumerate(_evaluate_cases(cases, workers), start=1):
            success = entry["accepted"]
            counts[(entry["group"], success)] += 1
            outcomes[entry["outcome"]] += 1
            if not success:
                failures_by_stage[entry["failure_stage"]] += 1
                reasons.update(entry["failed_criteria"])
                group_reasons.setdefault(entry["group"], Counter()).update(entry["failed_criteria"])
            else:
                remaining_main.append(entry["remaining_main_propellant_kg"])
            exposure = entry.get("gust_exposure", {})
            if exposure.get("configured_amplitude_m_s", 0.0) > 0.0:
                destination = (
                    encountered_gusts
                    if exposure["gust_encounter_observed"]
                    else unencountered_gusts
                )
                destination.append(entry["id"])
            stream.write(_encoded(entry) + "\n")
            stream.flush()
            print(
                f"[{split} {completed}/{len(cases)}] {entry['id']}: "
                f"{entry['outcome']} / {'pass' if success else 'fail'}",
                flush=True,
            )
    if _source_hashes() != source_hashes:
        raise ValueError("source changed during the evaluation; summary not created")
    summary = {
        "split": split,
        "workers": workers,
        "total": len(cases),
        "complete_split": len(cases) == len(manifest[split]),
        "selected_case_ids": [case["id"] for case in cases],
        "accepted": sum(value for (_, passed), value in counts.items() if passed),
        "by_group": {
            group: {
                "total": sum(counts[(group, passed)] for passed in (False, True)),
                "accepted": counts[(group, True)],
                "failed_criteria": dict(sorted(group_reasons.get(group, {}).items())),
            }
            for group in sorted({case["group"] for case in cases})
        },
        "outcomes": dict(sorted(outcomes.items())),
        "failure_stage": dict(sorted(failures_by_stage.items())),
        "failure_stage_note": "Last reached milestone, not a confirmed causal phase",
        "failed_criteria": dict(sorted(reasons.items())),
        "gust_encounter_case_ids": encountered_gusts,
        "gust_not_encountered_case_ids": unencountered_gusts,
        "continuous_envelope_proven": False,
        "real_vehicle_safety_certified": False,
        "minimum_successful_remaining_main_kg": min(remaining_main) if remaining_main else None,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_hashes": source_hashes,
        "scope": "Synthetic project-eligible conditions, not certified launch clearance",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(_encoded(summary), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", choices=("development", "holdout"))
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--tag", default="development")
    parser.add_argument("--development-tag", default="development")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--case-id", action="append", help="Select development case; repeatable")
    parser.add_argument("--group", action="append", help="Select development group; repeatable")
    args = parser.parse_args(argv)
    if (args.case_id or args.group) and args.run != "development":
        parser.error("case and group filters are only available for development runs")
    if args.workers < 1:
        parser.error("workers must be a positive integer")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.tag):
        parser.error("tag must be a short lowercase label")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.development_tag):
        parser.error("development tag must be a short lowercase label")
    try:
        if args.prepare:
            prepare(args.campaign_dir)
        elif args.freeze:
            freeze(args.campaign_dir, args.development_tag)
        else:
            evaluate(
                args.campaign_dir,
                args.run,
                args.tag,
                workers=args.workers,
                case_ids=args.case_id,
                groups=args.group,
            )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        KeyError,
        TypeError,
        RuntimeError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
