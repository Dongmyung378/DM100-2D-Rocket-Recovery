"""High-level Reusable RTLS flight runner for DM100."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from powered_landing_guidance.apr_recovery import (
    AprConditions,
    AscentGuidanceSettings,
    Design,
    RecoveryDisturbances,
)
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings
from powered_landing_guidance.rtls.contact import ContactSettings
from powered_landing_guidance.rtls.forces import (
    RtlsAerodynamics,
    RtlsFinSettings,
    RtlsPropulsionSettings,
)
from powered_landing_guidance.rtls.guidance import RtlsController, RtlsGuidanceSettings
from powered_landing_guidance.rtls.plant_2d import Plant2D, PlantConfig
from powered_landing_guidance.rtls.replay import export_replay_json


class RtlsFlight:
    """Orchestrates Plant2D <-> RtlsController strictly through Telemetry/Command."""

    def __init__(
        self,
        *,
        design: Design | None = None,
        conditions: AprConditions | None = None,
        disturbances: RecoveryDisturbances | None = None,
        sensor_settings: SensorSettings | None = None,
        controller_options: ControllerOptions | None = None,
        aero: RtlsAerodynamics | None = None,
        fins: RtlsFinSettings | None = None,
        propulsion: RtlsPropulsionSettings | None = None,
        contact_settings: ContactSettings | None = None,
        ascent_settings: AscentGuidanceSettings | None = None,
        guidance_settings: RtlsGuidanceSettings | None = None,
        plant_config: PlantConfig | None = None,
    ) -> None:
        self.design = design or Design()
        self.conditions = conditions or AprConditions()
        self.disturbances = disturbances or RecoveryDisturbances()
        self.sensor_settings = sensor_settings or SensorSettings()
        self.controller_options = controller_options or ControllerOptions()
        self.aero = aero or RtlsAerodynamics()
        self.fins = fins or RtlsFinSettings()
        self.propulsion = propulsion or RtlsPropulsionSettings()
        self.contact_settings = contact_settings or ContactSettings()
        self.ascent_settings = ascent_settings or AscentGuidanceSettings()
        self.guidance_settings = guidance_settings or RtlsGuidanceSettings()
        self.plant_config = plant_config or PlantConfig()

        self.plant = Plant2D(
            design=self.design,
            conditions=self.conditions,
            disturbances=self.disturbances,
            sensor_settings=self.sensor_settings,
            aero=self.aero,
            fins=self.fins,
            propulsion=self.propulsion,
            contact_settings=self.contact_settings,
            ascent_settings=self.ascent_settings,
            config=self.plant_config,
        )
        self.controller = RtlsController(
            design=self.design,
            conditions=self.conditions,
            aero=self.aero,
            fins=self.fins,
            propulsion=self.propulsion,
            ascent_settings=self.ascent_settings,
            guidance_settings=self.guidance_settings,
            position_error_m=self.disturbances.sensor_position_error_m,
            velocity_error_m_s=self.disturbances.sensor_velocity_error_m_s,
            observation_delay_steps=self.disturbances.observation_delay_steps,
            action_delay_steps=self.disturbances.action_delay_steps,
            sensor_settings=self.sensor_settings,
            controller_options=self.controller_options,
        )
        self.last_result: dict | None = None

    def run(self) -> dict:
        t0 = perf_counter()
        dt_ctrl = self.guidance_settings.control_dt_s
        while not self.plant.done:
            telemetry = self.plant.telemetry()
            command = self.controller.step(telemetry)
            self.plant.step(command, dt_ctrl)
        wall_s = perf_counter() - t0
        result = self.plant.summary()
        result["wall_seconds"] = wall_s
        result["replanning_count"] = self.controller.replanning_count
        result["convex_reference_feasible"] = bool(
            self.controller.convex_solution is not None and self.controller.convex_solution.feasible
        )
        result["sensor_diagnostics"] = {
            "generated_samples": self.plant.sensor_sample_count,
            "dropped_packets": self.plant.sensor_dropout_count,
        }
        self.last_result = result
        return result

    def export_replay(
        self,
        output_path: str | Path,
        *,
        metadata: dict | None = None,
    ) -> Path:
        meta = {
            "version": "dm100_rtls",
            "landing_mode": self.plant_config.landing_mode,
            "replay_hz": self.plant_config.replay_hz,
            "requirements_passed": (
                self.last_result["requirements_passed"] if self.last_result else None
            ),
            "outcome": self.plant.outcome,
        }
        if metadata:
            meta.update(metadata)
        return export_replay_json(self.plant.replay_frames, output_path, metadata=meta)
