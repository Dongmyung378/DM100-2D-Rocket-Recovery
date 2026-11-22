"""Comprehensive validation and extreme edge-case evaluator for DM100 Reusable RTLS.

Supports:
  - --suite edge-cases       : 32 adversarial boundary corners (including 100%-limit nightmares)
  - --suite validation-study : Full 169-case sweep of configs/validation-study-v1.json
                               (16 development + 96 holdout + 57 sensor-holdout)
  - --suite regression-286   : Full 286-case sweep of configs/apr-regression-v1.json
  - --suite all              : 32 edge-cases + 169 validation-study cases (201 total cases)
  - --suite full-487         : All 32 edge-cases + 169 study + 286 regression (487 total cases)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from powered_landing_guidance.apr_recovery import AprConditions, RecoveryDisturbances
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from powered_landing_guidance.rtls import PlantConfig, RtlsFlight

ROOT = Path(__file__).resolve().parents[1]


def _wind_prof(ground: float, upper: float, forecast_err: float = 0.0) -> tuple:
    actual = (
        (0.0, ground),
        (50.0, ground),
        (10_000.0, upper),
        (100_000.0, upper),
    )
    forecast = (
        (0.0, max(-8.0, min(8.0, ground + forecast_err))),
        (50.0, max(-8.0, min(8.0, ground + forecast_err))),
        (10_000.0, max(-20.0, min(20.0, upper + forecast_err))),
        (100_000.0, max(-20.0, min(20.0, upper + forecast_err))),
    )
    return actual, forecast


def build_extreme_edge_cases() -> list[dict]:
    """Build 24 adversarial boundary cases probing every corner of the DM100 envelope."""
    cases: list[dict] = []

    def add(
        case_id: str,
        description: str,
        *,
        ground_wind: float = 0.0,
        upper_wind: float = 0.0,
        forecast_err: float = 0.0,
        vertical_wind: float = 0.0,
        cond_extra: dict | None = None,
        dist_extra: dict | None = None,
        sens_extra: dict | None = None,
        seed: int = 378,
    ) -> None:
        w_act, w_fc = _wind_prof(ground_wind, upper_wind, forecast_err)
        cond_d = {
            "wind_profile": w_act,
            "forecast_wind_profile": w_fc,
            "vertical_wind_profile": ((0.0, vertical_wind), (100_000.0, vertical_wind)),
        }
        if cond_extra:
            cond_d.update(cond_extra)
        # Validate against AprConditions/RecoveryDisturbances/SensorSettings strictly
        AprConditions(**cond_d)
        RecoveryDisturbances(**(dist_extra or {}))
        SensorSettings(**(sens_extra or {}))
        cases.append(
            {
                "case_id": case_id,
                "split": "edge-cases",
                "description": description,
                "conditions": cond_d,
                "disturbances": dist_extra or {},
                "sensors": sens_extra or {},
                "seed": seed,
            }
        )

    add(
        "edge-01-max-east-wind",
        "Max East wind (+8 ground / +20 upper)",
        ground_wind=8.0,
        upper_wind=20.0,
    )
    add(
        "edge-02-max-west-wind",
        "Max West wind (-8 ground / -20 upper)",
        ground_wind=-8.0,
        upper_wind=-20.0,
    )
    add(
        "edge-03-shear-west-to-east",
        "Opposing shear (-8 ground -> +20 upper)",
        ground_wind=-8.0,
        upper_wind=20.0,
    )
    add(
        "edge-04-shear-east-to-west",
        "Opposing shear (+8 ground -> -20 upper)",
        ground_wind=8.0,
        upper_wind=-20.0,
    )
    add(
        "edge-05-max-updraft-fc-bias",
        "Max updraft (+3 m/s) + +2 m/s wind forecast error",
        ground_wind=4.0,
        upper_wind=12.0,
        forecast_err=2.0,
        vertical_wind=3.0,
    )
    add(
        "edge-06-max-downdraft-fc-bias",
        "Max downdraft (-3 m/s) + -2 m/s wind forecast error",
        ground_wind=-4.0,
        upper_wind=-12.0,
        forecast_err=-2.0,
        vertical_wind=-3.0,
    )
    add(
        "edge-07-gust-ascent-maxq",
        "5 m/s fast gust (8 s) at ascent Max-Q (10 km)",
        ground_wind=3.0,
        upper_wind=12.0,
        cond_extra={
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 8.0,
            "gust_start_s": 30.0,
            "gust_duration_s": 100.0,
            "gust_altitude_center_m": 10_000.0,
            "gust_altitude_half_width_m": 4_000.0,
        },
    )
    add(
        "edge-08-gust-entry-burn",
        "5 m/s gust during entry burn (20 km)",
        ground_wind=-4.0,
        upper_wind=-14.0,
        cond_extra={
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 12.0,
            "gust_start_s": 280.0,
            "gust_duration_s": 100.0,
            "gust_altitude_center_m": 20_000.0,
            "gust_altitude_half_width_m": 5_000.0,
        },
    )
    add(
        "edge-09-gust-aero-glide",
        "5 m/s gust during unpowered aero glide (6 km)",
        ground_wind=4.0,
        upper_wind=14.0,
        forecast_err=-2.0,
        cond_extra={
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 15.0,
            "gust_start_s": 320.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 6_000.0,
            "gust_altitude_half_width_m": 5_000.0,
        },
    )
    add(
        "edge-10-gust-landing-burn",
        "5 m/s gust during terminal landing burn (0-5 km)",
        ground_wind=-6.0,
        upper_wind=-10.0,
        forecast_err=2.0,
        cond_extra={
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 10.0,
            "gust_start_s": 350.0,
            "gust_duration_s": 100.0,
            "gust_altitude_center_m": 500.0,
            "gust_altitude_half_width_m": 3_000.0,
        },
    )
    add(
        "edge-11-worst-maxq-corner",
        "Worst Max-Q corner (density 1.10, drag 0.85, thrust 1.03, mass 0.99, inertia 0.97)",
        cond_extra={"density_scale": 1.10},
        dist_extra={
            "drag_coefficient_scale": 0.85,
            "thrust_scale": 1.03,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
        },
    )
    add(
        "edge-12-heavy-low-thrust",
        "Heavy low-thrust corner (density 0.90, drag 0.85, thrust 0.97, mass 1.01, inertia 1.03)",
        cond_extra={"density_scale": 0.90},
        dist_extra={
            "drag_coefficient_scale": 0.85,
            "thrust_scale": 0.97,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
        },
    )
    add(
        "edge-13-max-aero-torque",
        "Max aero torque (density 1.10, drag 1.15, +6/+18 wind, 5 m/s glide gust)",
        ground_wind=6.0,
        upper_wind=18.0,
        forecast_err=-2.0,
        cond_extra={
            "density_scale": 1.10,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 18.0,
            "gust_start_s": 310.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 3_000.0,
            "gust_altitude_half_width_m": 4_000.0,
        },
        dist_extra={"drag_coefficient_scale": 1.15, "inertia_scale": 0.97},
    )
    add(
        "edge-14-min-aero-braking",
        "Min aero braking (density 0.90, drag 0.85, -6/-18 wind, -3 m/s downdraft)",
        ground_wind=-6.0,
        upper_wind=-18.0,
        forecast_err=2.0,
        vertical_wind=-3.0,
        cond_extra={"density_scale": 0.90},
        dist_extra={"drag_coefficient_scale": 0.85, "thrust_scale": 0.97},
    )
    add(
        "edge-15-cold-dense-day",
        "Cold dense surface (14 C, 104 kPa, density 1.04, drag 1.15)",
        ground_wind=4.0,
        upper_wind=10.0,
        cond_extra={
            "sea_level_temperature_k": 287.15,
            "sea_level_pressure_pa": 104_000.0,
            "density_scale": 1.04,
        },
        dist_extra={"drag_coefficient_scale": 1.15},
    )
    add(
        "edge-16-hot-thin-day",
        "Hot thin surface (16.5 C, 97.5 kPa, density 0.96, drag 0.85, thrust 0.97)",
        ground_wind=-4.0,
        upper_wind=-10.0,
        cond_extra={
            "sea_level_temperature_k": 289.65,
            "sea_level_pressure_pa": 97_500.0,
            "density_scale": 0.96,
        },
        dist_extra={"drag_coefficient_scale": 0.85, "thrust_scale": 0.97},
    )
    add(
        "edge-17-north-pole",
        "Polar latitude (+90 deg, zero horizontal Coriolis) + strong East wind",
        ground_wind=6.0,
        upper_wind=18.0,
        cond_extra={"latitude_deg": 90.0},
    )
    add(
        "edge-18-equator-max-coriolis",
        "Equatorial latitude (0 deg, max Coriolis) + opposing West wind",
        ground_wind=-6.0,
        upper_wind=-18.0,
        cond_extra={"latitude_deg": 0.0},
    )
    add(
        "edge-19-south-hemisphere",
        "Southern hemisphere (-45 deg) + shear + gust",
        ground_wind=-5.0,
        upper_wind=15.0,
        cond_extra={
            "latitude_deg": -45.0,
            "gust_amplitude_m_s": 4.0,
            "gust_start_s": 300.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 4_000.0,
            "gust_altitude_half_width_m": 4_000.0,
        },
    )
    add(
        "edge-20-max-nav-and-delays",
        "Max nav error (2 m, 1 m/s) + max obs(2)/act(1) delays + max engine lag (0.05 s)",
        ground_wind=5.0,
        upper_wind=15.0,
        dist_extra={
            "sensor_position_error_m": 2.0,
            "sensor_velocity_error_m_s": 1.0,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.05,
        },
        seed=2020,
    )
    add(
        "edge-21-max-imu-pos-bias-dropout",
        "Max IMU noise/pos bias (+0.05 deg, +0.05 deg/s) + worst dropout (2/20) + max delays",
        ground_wind=-4.0,
        upper_wind=-12.0,
        dist_extra={
            "sensor_position_error_m": 1.5,
            "sensor_velocity_error_m_s": 0.8,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.04,
        },
        sens_extra={
            "attitude_error_deg": 0.1,
            "gyro_error_deg_s": 0.1,
            "attitude_bias_deg": 0.05,
            "gyro_bias_deg_s": 0.05,
            "dropout_period_steps": 20,
            "dropout_length_steps": 2,
        },
        seed=2121,
    )
    add(
        "edge-22-max-imu-neg-bias-dropout",
        "Max IMU noise/neg bias (-0.05 deg, -0.05 deg/s) + worst dropout (2/20) + max delays",
        ground_wind=4.0,
        upper_wind=12.0,
        dist_extra={
            "sensor_position_error_m": 2.0,
            "sensor_velocity_error_m_s": 1.0,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.05,
        },
        sens_extra={
            "attitude_error_deg": 0.1,
            "gyro_error_deg_s": 0.1,
            "attitude_bias_deg": -0.05,
            "gyro_bias_deg_s": -0.05,
            "dropout_period_steps": 20,
            "dropout_length_steps": 2,
        },
        seed=2222,
    )
    add(
        "edge-23-combined-nightmare-low",
        "Nightmare #1: heavy+low thrust+low drag+West wind+downdraft+landing gust+max delays+IMU",
        ground_wind=-6.0,
        upper_wind=-16.0,
        forecast_err=2.0,
        vertical_wind=-2.5,
        cond_extra={
            "density_scale": 0.92,
            "gust_amplitude_m_s": 4.5,
            "gust_period_s": 14.0,
            "gust_start_s": 330.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 1_500.0,
            "gust_altitude_half_width_m": 3_500.0,
        },
        dist_extra={
            "thrust_scale": 0.97,
            "drag_coefficient_scale": 0.87,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
            "sensor_position_error_m": 1.8,
            "sensor_velocity_error_m_s": 0.9,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.045,
        },
        sens_extra={
            "attitude_error_deg": 0.08,
            "gyro_error_deg_s": 0.08,
            "attitude_bias_deg": 0.04,
            "gyro_bias_deg_s": -0.04,
            "dropout_period_steps": 30,
            "dropout_length_steps": 2,
        },
        seed=2323,
    )
    add(
        "edge-24-combined-nightmare-high",
        "Nightmare #2: light+high thrust+dense+high drag+East wind+glide gust+max delays+IMU",
        ground_wind=6.0,
        upper_wind=16.0,
        forecast_err=-2.0,
        vertical_wind=2.5,
        cond_extra={
            "density_scale": 1.08,
            "gust_amplitude_m_s": 4.5,
            "gust_period_s": 16.0,
            "gust_start_s": 310.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 4_000.0,
            "gust_altitude_half_width_m": 4_500.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.14,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            "sensor_position_error_m": 1.8,
            "sensor_velocity_error_m_s": 0.9,
            "observation_delay_steps": 2,
            "action_delay_steps": 1,
            "engine_lag_s": 0.045,
        },
        sens_extra={
            "attitude_error_deg": 0.08,
            "gyro_error_deg_s": 0.08,
            "attitude_bias_deg": -0.04,
            "gyro_bias_deg_s": 0.04,
            "dropout_period_steps": 30,
            "dropout_length_steps": 2,
        },
        seed=2424,
    )
    max_nav_del = {
        "sensor_position_error_m": 2.0,
        "sensor_velocity_error_m_s": 1.0,
        "observation_delay_steps": 2,
        "action_delay_steps": 1,
        "engine_lag_s": 0.05,
    }
    max_imu_pos = {
        "attitude_error_deg": 0.1,
        "gyro_error_deg_s": 0.1,
        "attitude_bias_deg": 0.05,
        "gyro_bias_deg_s": -0.05,
        "dropout_period_steps": 20,
        "dropout_length_steps": 2,
    }
    max_imu_neg = {
        "attitude_error_deg": 0.1,
        "gyro_error_deg_s": 0.1,
        "attitude_bias_deg": -0.05,
        "gyro_bias_deg_s": 0.05,
        "dropout_period_steps": 20,
        "dropout_length_steps": 2,
    }
    add(
        "edge-25-max-everything-low",
        "100% low corner: max mass/min thrust/min drag/min density/-8..-20 wind/gust/max sensors",
        ground_wind=-8.0,
        upper_wind=-20.0,
        forecast_err=2.0,
        vertical_wind=-3.0,
        cond_extra={
            "density_scale": 0.90,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 12.0,
            "gust_start_s": 330.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 1_500.0,
            "gust_altitude_half_width_m": 3_500.0,
        },
        dist_extra={
            "thrust_scale": 0.97,
            "drag_coefficient_scale": 0.85,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
            **max_nav_del,
        },
        sens_extra=max_imu_pos,
        seed=2525,
    )
    add(
        "edge-26-max-everything-high",
        "100% high corner: min mass/max thrust/max drag/max density/+8..+20 wind/gust/max sensors",
        ground_wind=8.0,
        upper_wind=20.0,
        forecast_err=-2.0,
        vertical_wind=3.0,
        cond_extra={
            "density_scale": 1.10,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 14.0,
            "gust_start_s": 310.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 4_000.0,
            "gust_altitude_half_width_m": 4_500.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.15,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            **max_nav_del,
        },
        sens_extra=max_imu_neg,
        seed=2626,
    )
    add(
        "edge-27-max-shear-heavy-nightmare",
        "Opposing extreme shear (-8/+20) + 5 m/s fast gust + heavy low-thrust + max sensors",
        ground_wind=-8.0,
        upper_wind=20.0,
        forecast_err=-2.0,
        vertical_wind=-3.0,
        cond_extra={
            "density_scale": 0.90,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 8.0,
            "gust_start_s": 340.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 1_500.0,
            "gust_altitude_half_width_m": 3_000.0,
        },
        dist_extra={
            "thrust_scale": 0.97,
            "drag_coefficient_scale": 0.85,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
            **max_nav_del,
        },
        sens_extra=max_imu_pos,
        seed=2727,
    )
    add(
        "edge-28-reverse-shear-light-nightmare",
        "Reverse extreme shear (+8/-20) + 5 m/s touchdown gust + light high-thrust + max sensors",
        ground_wind=8.0,
        upper_wind=-20.0,
        forecast_err=2.0,
        vertical_wind=3.0,
        cond_extra={
            "density_scale": 1.10,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 8.0,
            "gust_start_s": 350.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 500.0,
            "gust_altitude_half_width_m": 2_500.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.15,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            **max_nav_del,
        },
        sens_extra=max_imu_neg,
        seed=2828,
    )
    add(
        "edge-29-maxq-nightmare",
        "Max-Q corner (density 1.10, drag 0.85, thrust 1.03, mass 0.99) + gust + max sensors",
        ground_wind=8.0,
        upper_wind=20.0,
        forecast_err=2.0,
        cond_extra={
            "density_scale": 1.10,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 8.0,
            "gust_start_s": 30.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 10_000.0,
            "gust_altitude_half_width_m": 4_000.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 0.85,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            **max_nav_del,
        },
        sens_extra=max_imu_pos,
        seed=2929,
    )
    add(
        "edge-30-equator-coriolis-heavy",
        "Equator (lat 0, max Coriolis) + heavy low-thrust + -8/-20 wind + downdraft + max sensors",
        ground_wind=-8.0,
        upper_wind=-20.0,
        forecast_err=2.0,
        vertical_wind=-3.0,
        cond_extra={"latitude_deg": 0.0, "density_scale": 0.90},
        dist_extra={
            "thrust_scale": 0.97,
            "drag_coefficient_scale": 0.85,
            "initial_mass_scale": 1.01,
            "inertia_scale": 1.03,
            **max_nav_del,
        },
        sens_extra=max_imu_neg,
        seed=3030,
    )
    add(
        "edge-31-south-pole-cold-dense",
        "South pole (lat -90) + cold dense (14C, 104 kPa) + +8/+20 wind + glide gust + max sensors",
        ground_wind=8.0,
        upper_wind=20.0,
        forecast_err=-2.0,
        cond_extra={
            "latitude_deg": -90.0,
            "sea_level_temperature_k": 287.15,
            "sea_level_pressure_pa": 104_000.0,
            "density_scale": 1.04,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 12.0,
            "gust_start_s": 310.0,
            "gust_duration_s": 120.0,
            "gust_altitude_center_m": 5_000.0,
            "gust_altitude_half_width_m": 4_000.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.15,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            **max_nav_del,
        },
        sens_extra=max_imu_pos,
        seed=3131,
    )
    add(
        "edge-32-ground-gust-touchdown-shock",
        "5 m/s fast (8 s) ground-level gust at touchdown + +8 m/s ground wind + max sensors",
        ground_wind=8.0,
        upper_wind=16.0,
        forecast_err=-2.0,
        cond_extra={
            "density_scale": 1.08,
            "gust_amplitude_m_s": 5.0,
            "gust_period_s": 8.0,
            "gust_start_s": 360.0,
            "gust_duration_s": 90.0,
            "gust_altitude_center_m": 0.0,
            "gust_altitude_half_width_m": 1_500.0,
        },
        dist_extra={
            "thrust_scale": 1.03,
            "drag_coefficient_scale": 1.15,
            "initial_mass_scale": 0.99,
            "inertia_scale": 0.97,
            **max_nav_del,
        },
        sens_extra=max_imu_neg,
        seed=3232,
    )
    return cases


def _load_regression_286() -> list[dict]:
    cfg = json.loads((ROOT / "configs" / "apr-regression-v1.json").read_text(encoding="utf-8"))
    out = []
    for idx, job in enumerate(cfg["jobs"]):
        c = job["case"]
        cond_raw = dict(c["conditions"])
        for k in (
            "wind_profile",
            "forecast_wind_profile",
            "vertical_wind_profile",
            "temperature_profile",
            "pressure_profile",
        ):
            if k in cond_raw and cond_raw[k] is not None:
                cond_raw[k] = tuple(tuple(pair) for pair in cond_raw[k])
        out.append(
            {
                "case_id": f"reg-{idx:03d}-{c['id']}",
                "split": f"regression-{job['split']}",
                "description": job["job_id"],
                "conditions": cond_raw,
                "disturbances": c["disturbances"],
                "sensors": {},
                "seed": 378 + idx,
            }
        )
    return out


def _load_validation_160() -> list[dict]:
    cfg = json.loads((ROOT / "configs" / "validation-study-v1.json").read_text(encoding="utf-8"))
    out = []
    for c in cfg["cases"]:
        cid = c["case_id"]
        if cid.startswith("development-"):
            split = "development"
        elif cid.startswith("holdout-"):
            split = "holdout"
        else:
            split = "sensor-holdout"
        out.append(
            {
                "case_id": cid,
                "split": split,
                "description": cid,
                "conditions": c["conditions"],
                "disturbances": c["disturbances"],
                "sensors": c.get("sensors") or {},
                "seed": c["seed"],
            }
        )
    return out


def _run_case(case: dict) -> dict:
    cond = AprConditions(**case["conditions"])
    dist = RecoveryDisturbances(**case["disturbances"])
    sens = SensorSettings(**case["sensors"]) if case.get("sensors") else SensorSettings()
    res = RtlsFlight(
        conditions=cond,
        disturbances=dist,
        sensor_settings=sens,
        controller_options=ControllerOptions(),
        plant_config=PlantConfig(
            seed=case["seed"],
            sensor_seed=case["seed"] + 1000,
            record_replay=False,
        ),
    ).run()
    ag = res["phases"].get("aero_glide", {})
    glide_km = max(0.0, (ag.get("start_z_m", 0.0) - ag.get("end_z_m", 0.0)) / 1000.0)
    sep = res.get("separation") or {}
    orb = res.get("upper_stage_orbit") or {}
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "description": case.get("description", case["case_id"]),
        "passed": res["requirements_passed"],
        "outcome": res["outcome"],
        "failed_criteria": res["failed_criteria"],
        "restarts_used": res["restarts_used"],
        "main_remaining_kg": res["main_remaining_kg"],
        "rcs_remaining_kg": res["rcs_remaining_kg"],
        "position_error_m": res["position_error_m"],
        "vertical_foot_speed_m_s": res["vertical_foot_speed_m_s"],
        "horizontal_foot_speed_m_s": res["horizontal_foot_speed_m_s"],
        "tilt_deg": res["tilt_deg"],
        "contact_rate_deg_s": res["contact_rate_deg_s"],
        "peak_q_pa": res["peak_dynamic_pressure_pa"],
        "peak_heat_flux_w_m2": res["peak_heat_flux_w_m2"],
        "peak_q_alpha_kpa_deg": res["peak_q_alpha_kpa_deg"],
        "peak_ascent_high_q_alpha_deg": res["peak_ascent_high_q_alpha_deg"],
        "peak_ascent_axial_g": res["peak_ascent_axial_g"],
        "peak_recovery_g": res["peak_recovery_g"],
        "peak_rate_deg_s": res["peak_angular_rate_deg_s"],
        "peak_accel_deg_s2": res["peak_angular_acceleration_deg_s2"],
        "peak_fin_deg": res["peak_fin_deflection_deg"],
        "peak_fin_hinge_nm": res["peak_fin_hinge_nm"],
        "separation_altitude_km": sep.get("altitude_m", 0.0) / 1000.0,
        "separation_downrange_km": sep.get("downrange_m", 0.0) / 1000.0,
        "separation_speed_m_s": sep.get("speed_m_s", 0.0),
        "separation_pitch_deg": sep.get("pitch_deg", 0.0),
        "orbit_inserted": bool(orb.get("orbit_inserted", False)),
        "orbit_altitude_km": orb.get("final_apogee_km", 0.0),
        "upper_propellant_remaining_kg": orb.get("upper_propellant_remaining_kg", 0.0),
        "glide_duration_s": ag.get("duration_s", 0.0),
        "glide_distance_km": glide_km,
        "glide_main_used_kg": ag.get("main_used_kg", 0.0),
        "standing_stable": res["post_contact"]["standing_stable"],
        "max_leg_stroke_m": res["post_contact"]["max_leg_stroke_m"],
        "max_contact_load_g": res["post_contact"]["max_contact_load_g"],
        "wall_seconds": res["wall_seconds"],
    }


def _summarize_group(rows: list[dict]) -> dict:
    passed = [r for r in rows if r["passed"]]
    pos = [r["position_error_m"] for r in rows if r["position_error_m"] is not None]
    vv = [r["vertical_foot_speed_m_s"] for r in rows if r["vertical_foot_speed_m_s"] is not None]
    vh = [
        r["horizontal_foot_speed_m_s"] for r in rows if r["horizontal_foot_speed_m_s"] is not None
    ]
    return {
        "total": len(rows),
        "passed": len(passed),
        "pass_rate_pct": 100.0 * len(passed) / max(len(rows), 1),
        "failed_cases": [
            {
                "case_id": r["case_id"],
                "outcome": r["outcome"],
                "failed_criteria": r["failed_criteria"],
                "position_error_m": r["position_error_m"],
                "peak_q_pa": r["peak_q_pa"],
                "peak_accel_deg_s2": r["peak_accel_deg_s2"],
            }
            for r in rows
            if not r["passed"]
        ],
        "extrema": {
            "main_remaining_kg": [
                min(r["main_remaining_kg"] for r in rows),
                max(r["main_remaining_kg"] for r in rows),
            ],
            "rcs_remaining_kg": [
                min(r["rcs_remaining_kg"] for r in rows),
                max(r["rcs_remaining_kg"] for r in rows),
            ],
            "position_error_m": [min(pos), max(pos)] if pos else None,
            "vertical_foot_speed_m_s": [min(vv), max(vv)] if vv else None,
            "horizontal_foot_speed_m_s": [min(vh), max(vh)] if vh else None,
            "peak_q_kpa": [
                min(r["peak_q_pa"] for r in rows) / 1000.0,
                max(r["peak_q_pa"] for r in rows) / 1000.0,
            ],
            "peak_heat_flux_kw_m2": [
                min(r["peak_heat_flux_w_m2"] for r in rows) / 1000.0,
                max(r["peak_heat_flux_w_m2"] for r in rows) / 1000.0,
            ],
            "orbit_altitude_km": [
                min(r["orbit_altitude_km"] for r in rows),
                max(r["orbit_altitude_km"] for r in rows),
            ],
            "orbit_inserted_count": sum(1 for r in rows if r["orbit_inserted"]),
            "separation_altitude_km": [
                min(r["separation_altitude_km"] for r in rows),
                max(r["separation_altitude_km"] for r in rows),
            ],
            "separation_downrange_km": [
                min(r["separation_downrange_km"] for r in rows),
                max(r["separation_downrange_km"] for r in rows),
            ],
            "peak_accel_deg_s2": [
                min(r["peak_accel_deg_s2"] for r in rows),
                max(r["peak_accel_deg_s2"] for r in rows),
            ],
            "glide_distance_km": [
                min(r["glide_distance_km"] for r in rows),
                max(r["glide_distance_km"] for r in rows),
            ],
            "standing_stable_count": sum(1 for r in rows if r["standing_stable"] is True),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "edge-cases",
            "validation-160",
            "validation-study",
            "regression-286",
            "all",
            "full-487",
        ),
        default="edge-cases",
    )
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    jobs: list[dict] = []
    if args.suite in ("edge-cases", "all", "full-487"):
        jobs.extend(build_extreme_edge_cases())
    if args.suite in ("validation-160", "validation-study", "all", "full-487"):
        jobs.extend(_load_validation_160())
    if args.suite in ("regression-286", "full-487"):
        jobs.extend(_load_regression_286())

    t0 = perf_counter()
    print(f"Running {len(jobs)} DM100 RTLS cases across {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_run_case, jobs))
    elapsed = perf_counter() - t0

    splits: dict[str, list[dict]] = {}
    for r in rows:
        splits.setdefault(r["split"], []).append(r)

    summary = {
        "schema_version": "dm100_rtls_validation_1",
        "created_utc": datetime.now(UTC).isoformat(),
        "suite": args.suite,
        "elapsed_wall_s": elapsed,
        "overall": _summarize_group(rows),
        "by_split": {k: _summarize_group(v) for k, v in splits.items()},
        "cases": rows,
    }

    ov = summary["overall"]
    print(
        f"\n=== OVERALL: {ov['passed']}/{ov['total']} PASSED "
        f"({ov['pass_rate_pct']:.1f}%) in {elapsed:.1f} s ==="
    )
    for sp, grp in summary["by_split"].items():
        ext = grp["extrema"]
        pos_s = (
            f"{ext['position_error_m'][0]:.3f}-{ext['position_error_m'][1]:.3f}m"
            if ext["position_error_m"]
            else "N/A"
        )
        print(
            f"  [{sp:15s}] {grp['passed']:3d}/{grp['total']:3d} ({grp['pass_rate_pct']:5.1f}%) | "
            f"main={ext['main_remaining_kg'][0]:.0f}-{ext['main_remaining_kg'][1]:.0f}kg | "
            f"pos={pos_s} | q={ext['peak_q_kpa'][1]:.2f}kPa | "
            f"acc={ext['peak_accel_deg_s2'][1]:.2f}deg/s2 | "
            f"glide={ext['glide_distance_km'][0]:.1f}-{ext['glide_distance_km'][1]:.1f}km"
        )

    if args.suite == "edge-cases" or ov["failed_cases"]:
        print("\nCase details:")
        for r in rows:
            if args.suite == "edge-cases" or not r["passed"]:
                pe = (
                    f"{r['position_error_m']:.3f}m" if r["position_error_m"] is not None else "None"
                )
                vv = (
                    f"{r['vertical_foot_speed_m_s']:.2f}"
                    if r["vertical_foot_speed_m_s"] is not None
                    else "None"
                )
                vh = (
                    f"{r['horizontal_foot_speed_m_s']:.2f}"
                    if r["horizontal_foot_speed_m_s"] is not None
                    else "None"
                )
                print(
                    f"  {r['case_id']:32s} pass={str(r['passed']):5s} out={r['outcome']:10s} "
                    f"main={r['main_remaining_kg']:6.0f} rcs={r['rcs_remaining_kg']:5.1f} "
                    f"pos={pe:>7s} vv={vv:>4s} vh={vh:>4s} q={r['peak_q_pa'] / 1000:5.2f}k "
                    f"g={r['peak_recovery_g']:4.2f} acc={r['peak_accel_deg_s2']:4.2f} "
                    f"fin={r['peak_fin_deg']:4.1f} glide={r['glide_distance_km']:4.1f}km "
                    f"fail={r['failed_criteria']}"
                )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved validation results -> {args.out_json}")

    return 0 if ov["passed"] == ov["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
