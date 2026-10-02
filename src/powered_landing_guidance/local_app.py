"""Offline, loopback-only DM100 browser application.

Run ``python -m powered_landing_guidance.local_app`` after installing the ui extra.
The web process remains responsive while an owned child process computes a flight
or renders the already saved flight. No network services are used by the UI.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import multiprocessing
import os
import queue
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, fields
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from powered_landing_guidance.apr_recovery import (
    FINS,
    PHYSICS,
    SAFETY,
    AprConditions,
    AprFlight,
    Design,
    RecoveryDisturbances,
)

_FIELD_ROWS = (
    ("ground_wind_m_s", "지상 수평풍", "Ground horizontal wind", "m/s", -8, 8, 0.5, 0),
    ("upper_wind_m_s", "상층 수평풍", "Upper horizontal wind", "m/s", -20, 20, 0.5, 0),
    ("vertical_wind_m_s", "수직풍", "Vertical wind", "m/s", -3, 3, 0.1, 0),
    (
        "forecast_error_m_s",
        "수평풍 예보 오차",
        "Horizontal wind forecast error",
        "m/s",
        -2,
        2,
        0.1,
        0,
    ),
    ("sea_level_temperature_c", "지상 기온", "Ground temperature", "°C", 0, 35, 0.5, 15),
    ("sea_level_pressure_kpa", "지상 기압", "Ground pressure", "kPa", 95, 105, 0.1, 101.325),
    ("density_scale", "대기밀도 배율", "Atmospheric density scale", "×", 0.9, 1.1, 0.01, 1),
    ("latitude_deg", "위도", "Latitude", "°", -90, 90, 0.1, 37),
    ("gust_amplitude_m_s", "돌풍 진폭", "Gust amplitude", "m/s", 0, 5, 0.1, 0),
    ("gust_period_s", "돌풍 주기", "Gust period", "s", 8, 60, 1, 12),
    ("gust_start_s", "돌풍 시작 시간", "Gust start time", "s", -120, 600, 1, 0),
    ("gust_duration_s", "돌풍 지속 시간", "Gust duration", "s", 8, 120, 1, 60),
    (
        "gust_altitude_center_m",
        "돌풍 중심 고도",
        "Gust center altitude",
        "m",
        0,
        100000,
        100,
        10000,
    ),
    (
        "gust_altitude_half_width_m",
        "돌풍 고도 반폭",
        "Gust altitude half-width",
        "m",
        100,
        5000,
        100,
        2500,
    ),
    ("gust_phase_rad", "돌풍 위상", "Gust phase", "rad", -math.pi, math.pi, 0.1, 0),
    ("thrust_scale", "추력 배율", "Thrust scale", "×", 0.97, 1.03, 0.001, 1),
    ("drag_coefficient_scale", "항력계수 배율", "Drag coefficient scale", "×", 0.85, 1.15, 0.01, 1),
    ("initial_mass_scale", "초기 질량 배율", "Initial mass scale", "×", 0.99, 1.01, 0.001, 1),
    ("inertia_scale", "관성 배율", "Inertia scale", "×", 0.97, 1.03, 0.001, 1),
    ("sensor_position_error_m", "위치 측정 오차", "Position measurement error", "m", 0, 2, 0.1, 0),
    (
        "sensor_velocity_error_m_s",
        "속도 측정 오차",
        "Velocity measurement error",
        "m/s",
        0,
        1,
        0.1,
        0,
    ),
    ("observation_delay_steps", "관측 지연", "Observation delay", "steps", 0, 2, 1, 0),
    ("action_delay_steps", "제어 지연", "Action delay", "steps", 0, 1, 1, 0),
    ("engine_lag_s", "엔진 응답 지연", "Engine response lag", "s", 0, 0.05, 0.001, 0),
)
DEFAULTS = {row[0]: row[-1] for row in _FIELD_ROWS}
LIMITS = {row[0]: dict(min=row[4], max=row[5], step=row[6]) for row in _FIELD_ROWS}
_DISTURBANCE_KEYS = {field.name for field in fields(RecoveryDisturbances)}
_PROFILE_KEYS = {
    "wind_profile",
    "forecast_wind_profile",
    "vertical_wind_profile",
    "temperature_profile",
    "pressure_profile",
}
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_FILE_NAMES = frozenset({"result.json", "trajectory.png", "flight.mp4"})
_FAILURE_LABELS = {
    "dynamic_pressure": ("동압 한계 초과", "Dynamic pressure limit exceeded"),
    "ascent_q_alpha": ("상승 동압·받음각 한계 초과", "Ascent q-alpha limit exceeded"),
    "ascent_high_q_alpha": (
        "고동압 구간 받음각 한계 초과",
        "High-q angle-of-attack limit exceeded",
    ),
    "ascent_axial_g": ("상승 축방향 하중 한계 초과", "Ascent axial load limit exceeded"),
    "flight_load_g": ("비행 하중 한계 초과", "Flight load limit exceeded"),
    "angular_rate": ("비행 각속도 한계 초과", "Flight angular-rate limit exceeded"),
    "angular_acceleration": (
        "비행 각가속도 한계 초과",
        "Flight angular-acceleration limit exceeded",
    ),
    "gimbal": ("짐벌각 한계 초과", "Gimbal-angle limit exceeded"),
    "fin_hinge": ("핀 힌지 모멘트 한계 초과", "Fin hinge-moment limit exceeded"),
    "fin_model_envelope": ("핀 모델 적용 범위 초과", "Fin model envelope exceeded"),
    "mass_balance": ("추진제 질량 보존 오차 초과", "Propellant mass-balance tolerance exceeded"),
    "weather_profile_domain": (
        "기상 프로필 고도 범위 초과",
        "Weather profile altitude domain exceeded",
    ),
    "feed_inventory": ("공급계 추진제 재고 기준 미충족", "Feed inventory check failed"),
    "feed_supply": ("추진제 공급 기준 미충족", "Propellant supply check failed"),
    "helium_mass_balance": ("헬륨 질량 보존 오차 초과", "Helium mass-balance tolerance exceeded"),
    "slosh_model_domain": ("슬로시 모델 적용 범위 초과", "Slosh model domain exceeded"),
    "separation_momentum": ("분리 운동량 보존 기준 미충족", "Separation momentum check failed"),
    "separation_pulse_estimate": (
        "분리 펄스 각가속도 한계 초과",
        "Separation pulse estimate limit exceeded",
    ),
    "separation": ("단 분리 조건 미충족", "Stage separation check failed"),
    "touchdown": ("착륙 판정 기준 미충족", "Touchdown acceptance checks failed"),
    "contact.position": ("착륙 위치 오차 한계 초과", "Landing position-error limit exceeded"),
    "contact.vertical_foot_speed": (
        "착륙 수직속도 한계 초과",
        "Landing vertical-speed limit exceeded",
    ),
    "contact.horizontal_foot_speed": (
        "착륙 수평속도 한계 초과",
        "Landing horizontal-speed limit exceeded",
    ),
    "contact.tilt": ("착륙 기울기 한계 초과", "Landing tilt limit exceeded"),
    "contact.angular_rate": ("착륙 각속도 한계 초과", "Landing angular-rate limit exceeded"),
    "contact.main_propellant": (
        "주엔진 추진제 잔량 기준 미충족",
        "Main propellant reserve check failed",
    ),
    "contact.rcs_propellant": (
        "RCS 추진제 잔량 기준 미충족",
        "RCS propellant reserve check failed",
    ),
    "contact.gear_deployed": ("착륙다리 전개 기준 미충족", "Landing gear deployment check failed"),
    "contact.feet_on_pad": (
        "접지 위치의 착륙장 범위 초과",
        "Foot contact outside landing pad limits",
    ),
    "return_plan_infeasible": (
        "복귀 계획 실행 가능 조건 미충족",
        "Return plan feasibility check failed",
    ),
    "separation_gate_missed": ("단 분리 조건 도달 실패", "Stage separation gate missed"),
    "touchdown_limits_exceeded": ("착륙 판정 기준 미충족", "Touchdown acceptance limits exceeded"),
    "body_ground_collision": ("기체와 지면의 충돌 발생", "Rocket body contacted the ground"),
    "feed_inventory_violation": ("공급계 추진제 재고 기준 위반", "Feed inventory violation"),
    "propellant_feed_starvation": (
        "주엔진 추진제 공급 부족",
        "Main engine propellant supply shortage",
    ),
    "slosh_model_out_of_domain": ("슬로시 모델 적용 범위 초과", "Slosh model domain exceeded"),
    "numerical_failure": (
        "수치 적분의 유효성 기준 미충족",
        "Numerical integration validity check failed",
    ),
    "main_propellant_depleted": ("주엔진 추진제 소진", "Main propellant depleted"),
    "timeout": (
        "계산 시간 범위 내 비행 종료 실패",
        "Flight did not finish within the simulation deadline",
    ),
    "source_changed": (
        "계산 중 소스 파일 변경으로 결과 확인 필요",
        "Source files changed during calculation",
    ),
    "nonfinite_contact": (
        "접지 데이터의 수치 유효성 기준 미충족",
        "Contact data contains invalid numeric values",
    ),
    "nonfinite_metrics": (
        "결과 수치의 유효성 기준 미충족",
        "Result metrics contain invalid numeric values",
    ),
    "inconsistent_acceptance": (
        "결과 판정 정보 불일치",
        "Acceptance result is internally inconsistent",
    ),
}


def _is_finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _failure_labels(failures: list[str], failed_criteria: list[str]) -> list[dict]:
    labels = []
    for key in dict.fromkeys(failures + failed_criteria):
        ko, en = _FAILURE_LABELS.get(key, (f"판정 기준 미충족: {key}", f"Check failed: {key}"))
        group = (
            "landing"
            if key.startswith("contact.") or key == "touchdown"
            else ("outcome" if key in failures else "flight")
        )
        if key in ("separation", "separation_momentum", "separation_pulse_estimate"):
            group = "mission"
        elif key in ("feed_inventory", "feed_supply", "helium_mass_balance", "slosh_model_domain"):
            group = "subsystem"
        labels.append(dict(key=key, label=dict(ko=ko, en=en), group=group))
    return labels


def _error(field: str, message: str, *, code: str = "input_invalid") -> dict:
    korean = {
        "Run not found.": "계산 결과를 찾을 수 없습니다.",
        "File not found.": "파일을 찾을 수 없습니다.",
        "File is not ready.": "파일 생성이 완료되지 않았습니다.",
        "Loopback Host required.": "이 컴퓨터의 로컬 주소로 접속해야 합니다.",
        "Same-origin requests required.": "현재 화면과 같은 로컬 주소에서 요청해야 합니다.",
        "Invalid local session token.": "로컬 세션이 만료되었습니다. 화면을 새로고침하세요.",
        "Settings are limited to 64 KiB.": "설정 데이터는 64 KiB 이내여야 합니다.",
        "Could not create the output directory.": "결과 저장 폴더를 만들 수 없습니다.",
        "UI package files are missing.": (
            "화면 파일이 누락되었습니다. UI 구성요소를 다시 설치하세요."
        ),
    }.get(message, message)
    english = (
        f"{field}: enter a finite number within the supported range."
        if field in LIMITS
        else "The input combination is outside the supported operating range."
    )
    if message.isascii():
        english = message
    if field in LIMITS:
        bounds = LIMITS[field]
        english = f"{field}: enter a number in [{bounds['min']}, {bounds['max']}]."
        if field in ("observation_delay_steps", "action_delay_steps"):
            code = "integer_delay"
            english = f"{field}: enter an integer in [{bounds['min']}, {bounds['max']}]."
    translations = (
        ("지원하지 않는", "unsupported_field", f"{field}: this input is not supported."),
        (
            "동시에",
            "profile_scalar_conflict",
            f"{field}: use either a profile or its scalar inputs.",
        ),
        (
            "대기밀도",
            "density_constraint",
            "Final density must remain within ±10% of the fixed reference atmosphere "
            "at every altitude. "
            "Adjust temperature, pressure and density scale together.",
        ),
        (
            "예보 오차",
            "forecast_constraint",
            "Actual and forecast horizontal wind must differ by no more than 2 m/s "
            "at every altitude.",
        ),
        ("0~100000", "profile_span", f"{field}: the profile must span 0 to 100000 m."),
        ("2~100", "profile_size", f"{field}: supply 2 to 100 altitude/value pairs."),
        ("쌍", "profile_pairs", f"{field}: supply finite numeric [altitude, value] pairs."),
        ("증가", "profile_order", f"{field}: altitudes must increase without duplicates."),
        ("윈드시어", "wind_shear", "Horizontal wind shear must not exceed 4 m/s per km."),
        (
            "수직풍 기울기",
            "vertical_wind_shear",
            "Vertical wind shear must not exceed 3 m/s per km.",
        ),
        (
            "상층 기본풍",
            "upper_wind_limit",
            "Actual and forecast upper wind must remain within ±20 m/s.",
        ),
        (
            "지상 지속풍",
            "ground_wind_limit",
            "Actual and forecast horizontal wind below 50 m must remain within ±8 m/s. "
            "Adjust wind and forecast error together.",
        ),
        ("수직풍 ±3", "vertical_wind_limit", "Vertical wind must remain within ±3 m/s."),
        (
            "지상값",
            "profile_ground_value",
            f"{field}: the first value must match the sea-level input.",
        ),
        ("온도 기울기", "temperature_shear", "Temperature gradients must not exceed 0.02 K/m."),
        (
            "정상 대기",
            "temperature_profile_constraint",
            "Temperature must remain within 150–330 K and within 25 K of the reference profile.",
        ),
        (
            "정역학",
            "pressure_profile_constraint",
            "Pressure must remain within ±10% of the hydrostatic atmosphere "
            "for the supplied temperature.",
        ),
        (
            "기압은 양수",
            "pressure_profile_order",
            "Pressure must be positive and decrease with altitude.",
        ),
        (
            "기압 기울기",
            "pressure_profile_gradient",
            "Pressure gradients must be consistent with the supplied temperature profile.",
        ),
        ("0~600", "gust_window", "An enabled gust must overlap the 0–600 s flight window."),
        ("JSON 객체", "json_object", "Settings must be a JSON object."),
        ("JSON 입력", "json_syntax", "Check the JSON syntax."),
        ("영상 너비", "video_width", "Video width must be 640, 1280 or 1920."),
        ("영상 시간", "video_time", "Video interval bounds must be finite numeric seconds."),
        (
            "영상 구간",
            "video_interval",
            "The clip interval must have start < end and lie within the saved flight bounds.",
        ),
        (
            "start_s, end_s, width",
            "video_fields",
            "Video settings support only start_s, end_s and width.",
        ),
        (
            "저장 비행 데이터",
            "video_data",
            "The saved flight does not have enough samples for video.",
        ),
        ("FFmpeg", "ffmpeg_missing", "Install FFmpeg or the video extra to generate an MP4."),
        (
            "전체 판정",
            "video_requires_success",
            "Video rendering requires a flight that passed all checks.",
        ),
        (
            "이미 생성",
            "video_already_created",
            "The video already exists; download the saved file.",
        ),
        (
            "진행 중",
            "compute_busy",
            "Calculation or video rendering is in progress. Wait until it finishes.",
        ),
    )
    for fragment, mapped_code, translated in translations:
        if fragment in message:
            code, english = mapped_code, translated
            break
    if code == "video_interval":
        bounds = re.search(r"([\d.]+)~([\d.]+)", message)
        english = "Use start < end inside the saved flight interval."
        if bounds:
            english += f" Saved bounds: {bounds[1]}–{bounds[2]} s."
    return dict(field=field, message=message, code=code, label=dict(ko=korean, en=english))


class InputError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(item["message"] for item in errors))
        self.errors = errors


def build_conditions(settings: dict) -> tuple[AprConditions, RecoveryDisturbances]:
    """Map browser units to the existing model, without clipping any inputs."""
    if not isinstance(settings, dict):
        raise InputError([_error("settings", "설정은 JSON 객체로 입력해야 합니다.")])
    errors = []
    for key in settings.keys() - (DEFAULTS.keys() | _PROFILE_KEYS):
        errors.append(_error(str(key), f"{key}: 지원하지 않는 입력 항목입니다."))
    for profile, overridden in (
        ("wind_profile", {"ground_wind_m_s", "upper_wind_m_s"}),
        ("forecast_wind_profile", {"forecast_error_m_s"}),
        ("vertical_wind_profile", {"vertical_wind_m_s"}),
    ):
        if profile in settings and overridden & settings.keys():
            errors.append(
                _error(profile, f"{profile}: 프로필과 기본 입력을 동시에 지정할 수 없습니다.")
            )
    values = DEFAULTS | settings
    for key, limits in LIMITS.items():
        value = values[key]
        valid = _is_finite_number(value) and limits["min"] <= value <= limits["max"]
        if key in ("observation_delay_steps", "action_delay_steps"):
            valid = valid and type(value) is int
        if not valid:
            errors.append(_error(key, f"{key}: 유한한 숫자와 허용 범위를 확인하세요."))
    if errors:
        raise InputError(errors)
    actual = values.get(
        "wind_profile",
        (
            (0, values["ground_wind_m_s"]),
            (50, values["ground_wind_m_s"]),
            (10000, values["upper_wind_m_s"]),
            (100000, values["upper_wind_m_s"]),
        ),
    )
    # A scalar error applies at every supplied altitude. The model validates the
    # resulting forecast too; a boundary value is never silently reduced.
    forecast = values.get("forecast_wind_profile")
    if "forecast_wind_profile" not in values:
        if not isinstance(actual, list | tuple) or any(
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or not _is_finite_number(pair[0])
            or not _is_finite_number(pair[1])
            for pair in actual
        ):
            raise InputError(
                [_error("wind_profile", "wind_profile: 유한한 고도·바람 쌍이 필요합니다.")]
            )
        forecast = tuple((z, wind + values["forecast_error_m_s"]) for z, wind in actual)
    vertical = values.get(
        "vertical_wind_profile",
        (
            (0, values["vertical_wind_m_s"]),
            (100000, values["vertical_wind_m_s"]),
        ),
    )
    environment = {key: values[key] for key in DEFAULTS if key.startswith("gust_")}
    environment.update(
        wind_profile=actual,
        forecast_wind_profile=forecast,
        vertical_wind_profile=vertical,
        sea_level_temperature_k=values["sea_level_temperature_c"] + 273.15,
        sea_level_pressure_pa=values["sea_level_pressure_kpa"] * 1000,
        density_scale=values["density_scale"],
        latitude_deg=values["latitude_deg"],
        temperature_profile=values.get("temperature_profile", ()),
        pressure_profile=values.get("pressure_profile", ()),
    )
    try:
        conditions = AprConditions(**environment)
        disturbances = RecoveryDisturbances(**{key: values[key] for key in _DISTURBANCE_KEYS})
    except (ValueError, TypeError, OverflowError) as exc:
        message = str(exc)
        field = next((key for key in values if key in message), "conditions")
        raise InputError([_error(field, message, code="conditions_invalid")]) from exc
    return conditions, disturbances


def validate_settings(settings: dict) -> dict:
    try:
        conditions, disturbances = build_conditions(settings)
    except InputError as exc:
        return dict(
            eligible=False,
            effective_conditions=None,
            effective_disturbances=None,
            errors=exc.errors,
        )
    return dict(
        eligible=True,
        effective_conditions=asdict(conditions),
        effective_disturbances=asdict(disturbances),
        errors=[],
    )


def _json_value(value: Any) -> Any:
    """Keep saved JSON standards-compliant, including optional unavailable metrics."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _chart_data(raw: dict, maximum: int = 1200) -> dict:
    steps = raw.get("steps", [])
    if not steps:
        return dict(samples=[], events=[])
    required = {0, len(steps) - 1}
    events = [dict(time_s=steps[0]["time_s"], phase=steps[0]["phase"])]
    for index in range(1, len(steps)):
        if steps[index]["phase"] != steps[index - 1]["phase"]:
            required.update((index - 1, index))
            events.append(dict(time_s=steps[index]["time_s"], phase=steps[index]["phase"]))
    for key in ("separation", "contact", "peak_angular_acceleration"):
        event = raw.get(key)
        if event is not None and "time_s" in event:
            nearest = min(
                range(len(steps)), key=lambda i: abs(steps[i]["time_s"] - event["time_s"])
            )
            required.add(nearest)
    # Preserve extrema as well as event-adjacent samples when reducing transport.
    for state_index in (0, 1, 2, 3, 4, 5):
        required.add(min(range(len(steps)), key=lambda i: steps[i]["state"][state_index]))
        required.add(max(range(len(steps)), key=lambda i: steps[i]["state"][state_index]))
    room = max(0, maximum - len(required))
    candidates = (
        {round(i * (len(steps) - 1) / max(1, room - 1)) for i in range(room)}
        if len(steps) > maximum
        else set(range(len(steps)))
    )
    selected = sorted(required | candidates)
    if len(selected) > maximum:
        selected = [
            selected[round(i * (len(selected) - 1) / (maximum - 1))] for i in range(maximum)
        ]
    samples = []
    for index in selected:
        step = steps[index]
        state = step["state"]
        samples.append(
            dict(
                time_s=step["time_s"],
                phase=step["phase"],
                x_m=state[0],
                z_m=state[1],
                vx_m_s=state[2],
                vz_m_s=state[3],
                attitude_deg=math.degrees(state[4]),
                angular_rate_deg_s=math.degrees(state[5]),
                throttle=step.get("engine_throttle", state[12] if len(state) > 12 else None),
                main_propellant_kg=state[6],
                rcs_propellant_kg=state[7],
                mass_kg=step.get("mass_kg"),
                dynamic_pressure_pa=step.get("dynamic_pressure_pa"),
            )
        )
    return dict(samples=samples, events=events)


def summarize_result(raw: dict, job_id: str) -> dict:
    """Expose contact foot speeds and independent safety/mission decisions."""
    contact = raw.get("contact")
    state = contact["state"] if contact else None
    velocity = contact["foot_velocities_m_s"] if contact else None
    acceptance = raw.get("physics", {}).get("acceptance", asdict(PHYSICS.acceptance))
    safety = raw.get("safety_limits", asdict(SAFETY))
    criteria = raw.get("criteria", {})
    contact_criteria = contact.get("criteria", {}) if contact else {}
    metrics = []

    def metric(key, ko, en, value, limit, unit, passed, group="landing", operator="<="):
        numeric = _json_value(value)
        if value is not None and not _is_finite_number(numeric):
            numeric, passed = None, False if passed is not None else None
        if numeric is not None and passed is True and _is_finite_number(limit):
            epsilon = 1e-7 if group == "flight" else 0.0
            actual_passed = (
                numeric <= limit + epsilon
                if operator == "<="
                else numeric > limit
                if operator == ">"
                else numeric >= limit
            )
            passed = bool(actual_passed)
        metrics.append(
            dict(
                key=key,
                label=dict(ko=ko, en=en),
                value=numeric,
                limit=limit,
                unit=unit,
                passed=passed,
                group=group,
                operator=operator,
                stage="first_contact" if group == "landing" else "through_first_contact",
                time_s=contact["time_s"] if contact and group == "landing" else None,
            )
        )

    def contact_speed(component):
        if not velocity:
            return None
        if any(
            not isinstance(pair, list | tuple)
            or len(pair) != 2
            or not _is_finite_number(pair[component])
            for pair in velocity
        ):
            return float("nan")
        return max(abs(pair[component]) for pair in velocity)

    metric(
        "landing_position_m",
        "착륙 위치 오차",
        "Landing position error",
        abs(state[0]) if state else None,
        acceptance["max_position_error_m"],
        "m",
        contact_criteria.get("position"),
    )
    metric(
        "landing_horizontal_speed_m_s",
        "착륙 수평속도",
        "Landing horizontal speed",
        contact_speed(0),
        acceptance["max_horizontal_contact_speed_m_s"],
        "m/s",
        contact_criteria.get("horizontal_foot_speed"),
    )
    metric(
        "landing_vertical_speed_m_s",
        "착륙 수직속도",
        "Landing vertical speed",
        contact_speed(1),
        acceptance["max_vertical_foot_speed_m_s"],
        "m/s",
        contact_criteria.get("vertical_foot_speed"),
    )
    metric(
        "landing_tilt_deg",
        "착륙 기울기",
        "Landing tilt",
        abs(math.degrees(state[4])) if state else None,
        acceptance["max_contact_tilt_deg"],
        "°",
        contact_criteria.get("tilt"),
    )
    metric(
        "landing_angular_rate_deg_s",
        "착륙 각속도",
        "Landing angular rate",
        abs(math.degrees(state[5])) if state else None,
        acceptance["max_contact_rate_deg_s"],
        "°/s",
        contact_criteria.get("angular_rate"),
    )
    metric(
        "main_propellant",
        "주엔진 추진제 잔량",
        "Main propellant remaining",
        state[6] if state else None,
        0,
        "kg",
        contact_criteria.get("main_propellant"),
        operator=">",
    )
    metric(
        "rcs_propellant",
        "RCS 추진제 잔량",
        "RCS propellant remaining",
        state[7] if state else None,
        0,
        "kg",
        contact_criteria.get("rcs_propellant"),
        operator=">=",
    )
    for key, ko, en, raw_key, limit_key, unit, scale in (
        (
            "dynamic_pressure",
            "최대 동압",
            "Peak dynamic pressure",
            "max_dynamic_pressure_pa",
            "max_q_pa",
            "kPa",
            0.001,
        ),
        (
            "ascent_q_alpha",
            "최대 동압·받음각",
            "Peak q-alpha",
            "max_ascent_q_alpha_kpa_deg",
            "max_q_alpha_kpa_deg",
            "kPa·°",
            1,
        ),
        (
            "angular_rate",
            "최대 각속도",
            "Peak angular rate",
            "max_angular_rate_deg_s",
            "max_angular_rate_deg_s",
            "°/s",
            1,
        ),
        (
            "angular_acceleration",
            "최대 각가속도",
            "Peak angular acceleration",
            "max_angular_acceleration_deg_s2",
            "max_angular_acceleration_deg_s2",
            "°/s²",
            1,
        ),
        (
            "flight_load_g",
            "최대 비행 하중",
            "Peak flight load",
            "max_flight_load_g",
            "max_recovery_flight_g",
            "g",
            1,
        ),
    ):
        value = raw.get(raw_key)
        metric(
            key,
            ko,
            en,
            value * scale if value is not None else None,
            safety[limit_key] * scale,
            unit,
            criteria.get(key),
            "flight",
        )
    metric(
        "gimbal",
        "최대 짐벌각",
        "Peak gimbal angle",
        raw.get("max_gimbal_deg"),
        raw.get("design", {}).get("gimbal_limit_deg", Design().gimbal_limit_deg),
        "°",
        criteria.get("gimbal"),
        "flight",
    )
    metric(
        "fin_hinge",
        "최대 핀 힌지 모멘트",
        "Peak fin hinge moment",
        raw.get("max_fin_hinge_nm"),
        FINS.per_fin_hinge_limit_nm,
        "N·m",
        criteria.get("fin_hinge"),
        "flight",
    )
    failures = list(raw.get("failures", []))
    failed_criteria = []
    for key, passed in criteria.items():
        if passed is False:
            failed_criteria.append(key)
    if contact:
        failed_criteria.extend(
            f"contact.{key}" for key, passed in contact_criteria.items() if passed is False
        )
    integrity_keys = []
    source_unchanged = raw.get("local_app_provenance", {}).get("source_unchanged")
    if source_unchanged is False:
        integrity_keys.append("source_changed")
    engine_accepted = raw.get("simulated_acceptance_passed") is True
    contact_finite = bool(
        contact
        and isinstance(state, list | tuple)
        and len(state) >= 9
        and all(_is_finite_number(value) for value in state)
        and isinstance(velocity, list | tuple)
        and len(velocity) >= 2
        and all(
            isinstance(pair, list | tuple)
            and len(pair) == 2
            and all(_is_finite_number(value) for value in pair)
            for pair in velocity
        )
        and _is_finite_number(contact.get("time_s"))
    )
    positions = contact.get("foot_positions_m") if contact else None
    if positions is not None:
        contact_finite = contact_finite and bool(
            isinstance(positions, list | tuple)
            and positions
            and all(
                isinstance(pair, list | tuple)
                and len(pair) == 2
                and all(_is_finite_number(value) for value in pair)
                for pair in positions
            )
        )
    if (contact or engine_accepted) and not contact_finite:
        integrity_keys.append("nonfinite_contact")
    invalid_metrics = [row for row in metrics if row["value"] is None and row["passed"] is False]
    if invalid_metrics:
        integrity_keys.append("nonfinite_metrics")
    category_passed = all(
        raw.get(key) is True for key in ("mission_success", "flight_safety", "landing_success")
    )
    if engine_accepted and (
        not category_passed
        or failed_criteria
        or any(row["passed"] is False for row in metrics)
        or not contact
        or raw.get("outcome") != "landed"
    ):
        integrity_keys.append("inconsistent_acceptance")
    result_valid = not integrity_keys
    integrity_errors = _failure_labels([], integrity_keys)
    failure_labels = _failure_labels(failures, failed_criteria + integrity_keys)
    base = f"/api/runs/{job_id}/files"
    return _json_value(
        dict(
            success=bool(engine_accepted and category_passed and contact_finite and result_valid),
            result_valid=result_valid,
            integrity_errors=integrity_errors,
            simulated_acceptance_passed=bool(raw.get("simulated_acceptance_passed", False)),
            mission_success=bool(raw.get("mission_success", False)),
            flight_safety=bool(raw.get("flight_safety", False)),
            landing_success=bool(raw.get("landing_success", False)),
            outcome=raw.get("outcome"),
            criteria=criteria,
            contact_criteria=contact_criteria,
            failures=failures,
            failed_criteria=failed_criteria,
            failure_labels=failure_labels,
            metrics=metrics,
            time_s=raw.get("time_s"),
            first_subsystem_failure=raw.get("systems_diagnostics", {}).get("first_failure"),
            peak_angular_acceleration=raw.get("peak_angular_acceleration"),
            max_altitude_m=raw.get("max_altitude_m"),
            main_propellant_remaining_kg=raw.get("main_propellant_remaining_kg"),
            rcs_propellant_remaining_kg=raw.get("rcs_propellant_remaining_kg"),
            contact_time_s=contact["time_s"] if contact else None,
            charts=_chart_data(raw),
            files=dict(result_json=f"{base}/result.json", trajectory_png=f"{base}/trajectory.png"),
            description=raw.get("description"),
            assessment_scope=raw.get("assessment_scope"),
            conditions=raw.get("conditions"),
            disturbances=raw.get("disturbances"),
            source_unchanged=source_unchanged,
        )
    )


def _save_plot(raw: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    samples = _chart_data(raw)["samples"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    try:
        axes[0, 0].plot(
            [s["x_m"] / 1000 for s in samples], [s["z_m"] / 1000 for s in samples], color="#259487"
        )
        axes[0, 0].set(xlabel="Horizontal distance (km)", ylabel="Altitude (km)")
        time = [s["time_s"] for s in samples]
        axes[0, 1].plot(time, [s["vx_m_s"] for s in samples], label="Horizontal")
        axes[0, 1].plot(time, [s["vz_m_s"] for s in samples], label="Vertical")
        axes[0, 1].set(xlabel="Time (s)", ylabel="Velocity (m/s)")
        axes[0, 1].legend()
        axes[1, 0].plot(time, [s["attitude_deg"] for s in samples])
        axes[1, 0].set(xlabel="Time (s)", ylabel="Attitude (deg)")
        axes[1, 1].plot(time, [s["dynamic_pressure_pa"] / 1000 for s in samples])
        axes[1, 1].axhline(SAFETY.max_q_pa / 1000, color="#cf685c", linestyle="--")
        axes[1, 1].set(xlabel="Time (s)", ylabel="Dynamic pressure (kPa)")
        for axis in axes.flat:
            axis.grid(alpha=0.2)
        passed = raw.get("local_app_result_validation", {}).get(
            "success", raw["simulated_acceptance_passed"]
        )
        figure.suptitle("DM100 · " + ("PASS" if passed else "FAIL"))
        with destination.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=130)
    finally:
        plt.close(figure)


def _write_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            _json_value(data), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )


def _source_hashes() -> dict[str, str]:
    source = Path(__file__).parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("*.py"))
    }


def _run_flight_worker(settings: dict, directory: str, messages, cancel) -> None:
    try:
        hashes_before = _source_hashes()
        conditions, disturbances = build_conditions(settings)
        flight = AprFlight(
            conditions=conditions,
            disturbances=disturbances,
            fin_mode="active",
            integration_dt_s=0.01,
        )

        def progress(current):
            if cancel.is_set():
                raise InterruptedError("Calculation stopped because the application closed.")
            messages.put(
                dict(
                    kind="progress",
                    progress=dict(
                        phase=current.phase,
                        time_s=current.time_s,
                        deadline_s=current.deadline_s,
                        fraction=min(current.time_s / current.deadline_s, 0.99),
                    ),
                )
            )

        raw = flight.run(progress=progress)
        hashes_after = _source_hashes()
        raw["local_app_provenance"] = dict(
            created_at_utc=datetime.now(UTC).isoformat(),
            requested_settings=settings,
            source_sha256_before=hashes_before,
            source_sha256_after=hashes_after,
            source_unchanged=hashes_before == hashes_after,
        )
        destination = Path(directory)
        summary = summarize_result(raw, destination.name)
        raw["local_app_result_validation"] = dict(
            result_valid=summary["result_valid"],
            success=summary["success"],
            integrity_errors=summary["integrity_errors"],
        )
        _write_json(destination / "result.json", raw)
        _save_plot(raw, destination / "trajectory.png")
        messages.put(dict(kind="completed", result=summary))
    except Exception as exc:
        messages.put(dict(kind="execution_error", error=f"{type(exc).__name__}: {exc}"))


def _video_available() -> bool:
    from powered_landing_guidance.visualization import _ffmpeg_available

    return _ffmpeg_available()


def validate_video_params(params: dict, raw: dict) -> dict:
    if not isinstance(params, dict) or params.keys() - {"start_s", "end_s", "width"}:
        raise ValueError("영상 설정은 start_s, end_s, width만 지원합니다.")
    width = params.get("width", 1280)
    if type(width) is not int or width not in (640, 1280, 1920):
        raise ValueError("영상 너비는 640, 1280, 1920 중 하나여야 합니다.")
    steps = raw.get("steps", [])
    if len(steps) < 2:
        raise ValueError("영상에 필요한 저장 비행 데이터가 없습니다.")
    first, last = steps[0]["time_s"], steps[-1]["time_s"]
    start, end = params.get("start_s", first), params.get("end_s", last)
    if any(not _is_finite_number(value) for value in (start, end)):
        raise ValueError("영상 시간은 유한한 숫자여야 합니다.")
    if not first <= start < end <= last:
        raise ValueError(f"영상 구간은 저장 비행 시간 {first:.3f}~{last:.3f}초 안에 있어야 합니다.")
    return dict(start_s=float(start), end_s=float(end), width=width)


def _video_worker(params: dict, directory: str, messages, cancel) -> None:
    try:
        if cancel.is_set():
            raise InterruptedError("Video rendering stopped.")
        from powered_landing_guidance.visualization import save_nominal_flight_replay

        destination = Path(directory)
        raw = json.loads((destination / "result.json").read_text(encoding="utf-8"))
        summary = summarize_result(raw, destination.name)
        if not summary["success"] or not summary["result_valid"]:
            raise ValueError("Video requires a valid saved flight that passed all checks.")
        save_nominal_flight_replay(raw, destination / "flight.mp4", fps=24, **params)
        messages.put(dict(kind="completed"))
    except Exception as exc:
        messages.put(dict(kind="execution_error", error=f"{type(exc).__name__}: {exc}"))


def _worker_entry(kind, payload, directory, messages, cancel):
    if os.name != "nt":
        os.setsid()
    target = _run_flight_worker if kind == "flight" else _video_worker
    target(payload, directory, messages, cancel)


def default_output_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    app_name = "DM100" if os.name == "nt" or sys.platform == "darwin" else "dm100"
    return base / app_name / "runs"


def _terminate_owned_process(process) -> None:
    if not process.is_alive():
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    process.join(timeout=2)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


class BusyError(RuntimeError):
    pass


class JobManager:
    """One owned compute process, with isolated never-overwritten output folders."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = (output_dir or default_output_dir()).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._active: dict | None = None
        self._closed = False
        self._context = multiprocessing.get_context("spawn")

    def _job(self, job_id: str) -> dict:
        if not _RUN_ID.fullmatch(job_id) or job_id not in self._jobs:
            raise KeyError(job_id)
        return self._jobs[job_id]

    def get(self, job_id: str) -> dict:
        with self._lock:
            return _json_value(self._job(job_id))

    def submit(self, settings: dict) -> dict:
        build_conditions(settings)
        with self._lock:
            if self._closed or self._active is not None:
                raise BusyError("계산 또는 영상 생성이 진행 중입니다. 완료 후 다시 실행하세요.")
            while len(self._jobs) >= 64:
                # Retain output files; only expire old, completed response caches.
                self._jobs.pop(next(iter(self._jobs)))
            job_id = uuid4().hex
            directory = self.output_dir / job_id
            directory.mkdir()
            _write_json(
                directory / "input.json",
                dict(settings=settings, seed=378, integration_dt_s=0.01, fin_mode="active"),
            )
            self._jobs[job_id] = dict(
                id=job_id,
                status="running",
                error=None,
                result=None,
                progress=dict(phase="starting", time_s=0, deadline_s=600, fraction=0),
                video=dict(status="not_requested", error=None, url=None),
            )
            self._spawn(job_id, "flight", settings)
            return dict(id=job_id, status=self._jobs[job_id]["status"])

    def _spawn(self, job_id: str, kind: str, payload: dict) -> None:
        messages = self._context.Queue()
        cancel = self._context.Event()
        process = self._context.Process(
            target=_worker_entry,
            args=(kind, payload, str(self.output_dir / job_id), messages, cancel),
            name=f"DM100-{kind}-{job_id[:8]}",
        )
        active = dict(process=process, messages=messages, cancel=cancel, id=job_id, kind=kind)
        self._active = active
        try:
            process.start()
        except Exception as exc:
            self._set_terminal(job_id, kind, dict(kind="execution_error", error=str(exc)))
            messages.close()
            self._active = None
            return
        watcher = threading.Thread(target=self._monitor, args=(active,), daemon=True)
        active["watcher"] = watcher
        watcher.start()

    def _set_terminal(self, job_id: str, kind: str, message: dict) -> None:
        job = self._job(job_id)
        if kind == "video":
            job["video"] = dict(
                status=message["kind"],
                error=message.get("error"),
                url=f"/api/runs/{job_id}/files/flight.mp4"
                if message["kind"] == "completed"
                else None,
            )
        else:
            job.update(
                status=message["kind"], error=message.get("error"), result=message.get("result")
            )
            if message["kind"] == "completed":
                job["progress"].update(
                    phase="completed", time_s=job["result"]["time_s"], fraction=1
                )

    def _monitor(self, active: dict) -> None:
        process, messages = active["process"], active["messages"]
        terminal = False
        try:
            while not terminal:
                try:
                    message = messages.get(timeout=0.25)
                except queue.Empty:
                    if process.is_alive():
                        continue
                    message = dict(
                        kind="execution_error",
                        error=f"Calculation process exited unexpectedly ({process.exitcode}).",
                    )
                with self._lock:
                    if message["kind"] == "progress":
                        self._job(active["id"])["progress"] = message["progress"]
                    else:
                        self._set_terminal(active["id"], active["kind"], message)
                        terminal = True
            process.join(timeout=2)
            if process.is_alive():
                _terminate_owned_process(process)
        finally:
            with self._lock:
                if self._active is active:
                    self._active = None
            messages.close()

    def start_video(self, job_id: str, params: dict) -> dict:
        with self._lock:
            job = self._job(job_id)
            if job["status"] != "completed" or not job["result"]["success"]:
                raise BusyError("전체 판정을 통과한 비행 결과만 영상으로 생성할 수 있습니다.")
            if self._active is not None or self._closed:
                raise BusyError("계산 또는 영상 생성이 진행 중입니다.")
            if job["video"]["status"] == "completed":
                raise BusyError("영상이 이미 생성되었습니다. 기존 영상 파일을 내려받으세요.")
            if not _video_available():
                raise BusyError("FFmpeg가 없습니다. video 추가 구성요소 또는 FFmpeg를 설치하세요.")
            raw = json.loads((self.output_dir / job_id / "result.json").read_text(encoding="utf-8"))
            verified = summarize_result(raw, job_id)
            if not verified["success"] or not verified["result_valid"]:
                raise BusyError(
                    "저장된 결과가 전체 판정 또는 데이터 유효성 기준을 통과하지 못했습니다."
                )
            normalized = validate_video_params(params, raw)
            job["video"] = dict(status="rendering", error=None, url=None)
            self._spawn(job_id, "video", normalized)
            return _json_value(job["video"])

    def close(self) -> None:
        with self._lock:
            self._closed = True
            active = self._active
        if active is None:
            return
        process = active["process"]
        active["cancel"].set()
        process.join(timeout=2)
        if process.is_alive():
            # This is exclusively the child launched by this manager. Its video
            # encoder also belongs to it, so terminate the owned process tree.
            _terminate_owned_process(process)
        watcher = active.get("watcher")
        if watcher and watcher is not threading.current_thread():
            watcher.join(timeout=2)


def _loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(output_dir: Path | None = None, manager: JobManager | None = None):
    """Create an application; injecting a manager permits short deterministic checks."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError('Install the UI extra: pip install -e ".[ui]"') from exc

    jobs = manager if manager is not None else JobManager(output_dir)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(jobs.close)

    app = FastAPI(
        title="DM100 local application",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.manager = jobs
    app.state.token = token

    @app.middleware("http")
    async def local_requests_only(request: Request, call_next):
        try:
            host = urlsplit("http://" + request.headers.get("host", ""))
            valid = bool(host.hostname and _loopback_host(host.hostname))
            valid = (
                valid
                and not host.username
                and not host.password
                and host.path == ""
                and not host.query
                and not host.fragment
            )
            valid = valid and (host.port is None or 0 < host.port <= 65535)
        except ValueError:
            valid = False
        if not valid:
            message = "Loopback Host required."
            return JSONResponse(
                dict(detail=message, error=_error("host", message)), status_code=403
            )
        origin = request.headers.get("origin")
        if origin is not None:
            expected = f"{request.url.scheme}://{request.headers['host']}"
            if origin.rstrip("/").lower() != expected.lower():
                message = "Same-origin requests required."
                return JSONResponse(
                    dict(detail=message, error=_error("origin", message)), status_code=403
                )
        if request.method not in ("GET", "HEAD", "OPTIONS") and not secrets.compare_digest(
            request.headers.get("x-dm100-token", ""), token
        ):
            message = "Invalid local session token."
            return JSONResponse(
                dict(detail=message, error=_error("token", message)), status_code=403
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self'; media-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    async def body_object(request: Request) -> dict:
        body = await request.body()
        if len(body) > 65536:
            raise HTTPException(413, detail="Settings are limited to 64 KiB.")
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(
                422, detail=[_error("settings", "JSON 입력을 확인하세요.")]
            ) from exc
        if not isinstance(value, dict):
            raise HTTPException(422, detail=[_error("settings", "JSON 객체가 필요합니다.")])
        return value

    # FastAPI resolves annotations against module globals. Request is imported
    # locally so core users can import the module without the optional UI extra.
    globals()["Request"] = Request

    @app.exception_handler(HTTPException)
    async def readable_errors(request: Request, exc: HTTPException):
        content = dict(detail=exc.detail)
        if isinstance(exc.detail, str):
            content["error"] = _error("request", exc.detail)
        return JSONResponse(content, status_code=exc.status_code, headers=exc.headers)

    @app.get("/api/config")
    def config():
        metadata = []
        for row in _FIELD_ROWS:
            key = row[0]
            group = (
                "disturbances"
                if key in _DISTURBANCE_KEYS
                else ("gust" if key.startswith("gust_") else "environment")
            )
            metadata.append(
                dict(
                    key=key,
                    label=dict(ko=row[1], en=row[2]),
                    unit=row[3],
                    group=group,
                    **LIMITS[key],
                )
            )
        return dict(
            token=token,
            defaults=DEFAULTS,
            limits=LIMITS,
            fields=metadata,
            seed=378,
            integration_dt_s=0.01,
            fin_mode="active",
            vehicle=asdict(Design()),
            video_available=_video_available(),
            presets=[
                dict(id="calm", label=dict(ko="무풍 기준", en="Calm baseline"), settings=DEFAULTS),
                dict(
                    id="light_wind",
                    label=dict(ko="약한 바람", en="Light wind"),
                    settings=DEFAULTS | dict(ground_wind_m_s=3, upper_wind_m_s=8),
                ),
                dict(
                    id="gust",
                    label=dict(ko="돌풍 조건", en="Gust condition"),
                    settings=DEFAULTS | dict(gust_amplitude_m_s=2, gust_start_s=240),
                ),
            ],
        )

    @app.post("/api/validate")
    async def validate(request: Request):
        return validate_settings(await body_object(request))

    @app.post("/api/runs", status_code=202)
    async def submit(request: Request):
        settings = await body_object(request)
        validated = validate_settings(settings)
        if not validated["eligible"]:
            raise HTTPException(422, detail=validated["errors"])
        try:
            return await asyncio.to_thread(jobs.submit, settings)
        except BusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, detail="Could not create the output directory.") from exc

    def get_job(job_id):
        if not _RUN_ID.fullmatch(job_id):
            raise HTTPException(404, detail="Run not found.")
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Run not found.") from exc

    @app.get("/api/runs/{job_id}")
    def status(job_id: str):
        return get_job(job_id)

    @app.get("/api/runs/{job_id}/video/status")
    def video_status(job_id: str):
        return get_job(job_id)["video"]

    @app.post("/api/runs/{job_id}/video", status_code=202)
    async def video(job_id: str, request: Request):
        get_job(job_id)
        params = await body_object(request)
        try:
            return await asyncio.to_thread(jobs.start_video, job_id, params)
        except BusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, detail=[_error("video", str(exc))]) from exc

    @app.get("/api/runs/{job_id}/files/{filename}")
    def download(job_id: str, filename: str):
        job = get_job(job_id)
        if filename not in _FILE_NAMES:
            raise HTTPException(404, detail="File not found.")
        ready = (
            job["video"]["status"] == "completed"
            if filename == "flight.mp4"
            else job["status"] == "completed"
        )
        target = jobs.output_dir / job_id / filename
        contained = target.resolve() == jobs.output_dir.resolve() / job_id / filename
        if not ready or not contained or not target.is_file() or target.is_symlink():
            raise HTTPException(404, detail="File is not ready.")
        media = {
            "result.json": "application/json",
            "trajectory.png": "image/png",
            "flight.mp4": "video/mp4",
        }[filename]
        return FileResponse(target, media_type=media, filename=f"DM100-{job_id[:8]}-{filename}")

    ui = Path(str(resources.files("powered_landing_guidance").joinpath("ui")))

    @app.get("/")
    def index():
        target = ui / "index.html"
        if not target.is_file():
            raise HTTPException(503, detail="UI package files are missing.")
        return FileResponse(target, media_type="text/html")

    if ui.is_dir():
        app.mount("/static", StaticFiles(directory=ui), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DM100 offline browser application.")
    parser.add_argument(
        "--port", type=int, default=0, help="Local port; 0 chooses an available port."
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for new uniquely named runs.")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the default browser."
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    try:
        import uvicorn

        app = create_app(args.output_dir)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"DM100: {exc}", file=sys.stderr)
        return 1
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", args.port))
        listener.listen(128)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=listener.getsockname()[1],
            proxy_headers=False,
            access_log=False,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        print(f"DM100: {url}\nResults: {app.state.manager.output_dir}\nStop: Ctrl+C", flush=True)
        if not args.no_browser:

            def open_when_ready():
                while not server.started and not server.should_exit:
                    threading.Event().wait(0.05)
                if server.started:
                    webbrowser.open(url)

            threading.Thread(target=open_when_ready, daemon=True).start()
        server.run(sockets=[listener])
    except OSError as exc:
        print(f"DM100: {exc}", file=sys.stderr)
        return 1
    finally:
        listener.close()
        app.state.manager.close()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
