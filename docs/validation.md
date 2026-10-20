# Validation and experiment design

[한국어](validation.ko.md) | [Model specification](spec.md) | [README](../README.md)

The evidence distinguishes repeatable software behavior, numerical accuracy,
performance inside a declared synthetic distribution and physical-model validity.
Passing one of these checks does not establish the others.

## How to read the experiments

A scenario specifies weather, vehicle assumptions and sensor behavior. A seed
sets the starting point for repeatable random draws, while the manifest records
the exact inputs and variants to run. A paired comparison executes both methods
under the same declared inputs and random streams. This controls the comparison
conditions even though each controller's decisions can lead to different flight
paths and measurement times.

An **ablation** removes a feature while keeping the other declared settings fixed
to examine that feature's contribution. A **step-size comparison** makes the
physics calculation's time increments smaller to check whether the answer
remains within the declared tolerance. **Sensitivity analysis** instead changes
a model assumption, such as thrust or drag, to inspect its effect on the result.
These experiments answer different questions and are reported separately.

## Evaluation populations

The original 286 inputs in `configs/apr-regression-v1.json` remain unchanged.
They measure regression on previously observed conditions. They are separate
from the new study in `configs/validation-study-v1.json`:

| Population | Declared conditions | Purpose |
|---|---:|---|
| Development | 16, with two controller variants | Diagnose and select the implementation |
| New-condition evaluation | 96 | Evaluate the original observation assumptions on unseen coupled inputs |
| Extended-sensor evaluation | 48, with two controller variants | Paired comparison under added angular instrumentation errors |
| Ablation | 12 development conditions, five variants | Isolate individual feature contributions |
| Step-size study | Four conditions, three integration intervals | Compare the hybrid full-flight calculation with fixed 0.02 s control periods |
| Sensitivity | Nominal plus eight one-factor perturbations | Measure response to thrust, drag, inertia and initial mass assumptions |

Development outcomes may inform implementation changes. Before either final
population is executed, `freeze.json` binds the source, manifest, dependency
environment and completed development evidence. Final outcomes are not used to
retune that frozen version. A later revision needs a new evaluation population.

All attempted flights remain in the denominator, including execution errors.
Mission completion, flight safety, first-contact acceptance and fuel conservation
must all pass. Original physical acceptance thresholds are never relaxed.

## New condition sampling

The new study combines independent bounded draws instead of repeating the
original three-knot wind profiles. Wind knots occur at 0, 4, 10, 22 and 100 km.
Ground and upper-level wind draws are bounded by 6 and 18 m/s; intermediate
knots use the declared interpolation plus uniform perturbations. Forecast offsets
are within 1.8 m/s. Other draws include vertical wind within 2.5 m/s, gusts up to
5 m/s, density scaling 0.97-1.03 and all previously declared healthy vehicle
deviations. Three predeclared gust windows target ascent, high-altitude return or
late descent. Actual gust exposure is a property of the resulting flight, not
guaranteed by the configured amplitude alone.

The generator and every exact input, seed and controller variant are public.
Inputs must satisfy the existing eligibility model before execution. This
distribution samples only part of the declared operating envelope. It cannot
establish success for all continuous combinations or for real flights.

Wilson 95% intervals describe empirical success under the study's synthetic
sampling recipe. They are not confidence bounds for rocket reliability.
Metric summaries include every completed flight with that measurement, including
failed flights. Contact-only metrics report their own measurement counts.

A Wilson interval expresses uncertainty from a finite set of successes and
failures. Even a sample with no observed failures leaves uncertainty about the
underlying success rate. Its interpretation remains limited to the declared
synthetic sampling assumptions.

## Added instrumentation

Attitude is the vehicle's orientation, and a gyro measures how quickly it rotates.
Noise changes between measurements; bias shifts readings persistently in one
direction. A sensor packet carries a measurement and its timestamp. Packet
omission means a scheduled observation is unavailable, so reusing an old value
must not be mistaken for obtaining a fresh measurement.

Position and velocity noise, latency and a causal navigation estimator already
existed. The new experiment adds the following synthetic channels:

| Quantity | Extended-sensor evaluation draw |
|---|---:|
| Uniform attitude noise radius | 0.01-0.05 degrees |
| Uniform gyro noise radius | 0.005-0.03 degrees/s |
| Constant attitude bias | -0.02 to +0.02 degrees |
| Constant gyro bias | -0.03 to +0.03 degrees/s |
| Scheduled packet omission | One or two samples per 60-120 control samples |

These ranges are explicit modeling assumptions, not selected sensor specifications
or measured hardware calibration. Omitted packets retain their original
timestamps. The estimators never assimilate the same packet twice. A separate
random stream keeps angular noise from changing the legacy position/velocity
noise sequence. Propellant and actuator telemetry remain ideal delayed channels.

The attitude estimator uses angle, angular rate and gyro bias as its states.
It wraps angle innovations and uses a Joseph covariance update. The configured
true bias is not supplied to the estimator. An independent attitude reference is
needed to identify gyro bias; absolute attitude bias remains a limitation.

The `raw-attitude` comparison uses the same sensor settings and declared random
streams with the existing control method and no attitude filter. Closed-loop
trajectories and observation timestamps can differ between methods.
It represents applying the previous
measurement-processing approach to the new instrumentation assumptions. It is
not a claim that the previous 286-condition model had these sensor errors.

## Controlled ablations

`ControllerOptions` disables one feature at a time: translation-bias compensation,
torque-bias compensation, boostback replanning or attitude filtering. The plant,
inputs, random streams, limits and acceptance thresholds remain identical.
Paired pass/fail outcomes and metric differences expose both benefits and
tradeoffs. Twelve development cases are a diagnostic sample, not a separate
independent final evaluation population.

## Numerical verification

Full-flight cases use 0.01, 0.005 and 0.0025 s integration intervals with the same
declared sensor seeds and 0.02 s control period. Event detection still handles
contact and propellant exhaustion. Staging resets the control clock to its
detected event time, so small event-time changes can shift subsequent observation
timestamps. The finest result is a comparison reference, not an exact solution.
Tolerances are declared in the manifest before execution; failures remain visible.

`scripts/check_numerical_reference.py` separately compares RK4 with SciPy DOP853
on three smooth, one-second, held-control coupled-plant segments. Fins remain
stowed in these segments to exclude deployment switches and saturation boundaries.
The full-flight study exercises those hybrid events. Solver agreement checks
implementation consistency; it does not validate assumed physics against reality.

The existing analytical freefall, mass accounting, staging momentum and contact
event tests remain part of the software suite. The sensitivity study reports
changes in landing and fuel metrics under one-factor parameter perturbations.
Vehicle coefficients and sensor error scales have no measured-data calibration.

## Reproduce

To configure an individual experiment through the Python API:

```python
from powered_landing_guidance.apr_recovery import AprFlight
from powered_landing_guidance.observation_models import ControllerOptions, SensorSettings

flight = AprFlight(
    sensor_settings=SensorSettings(
        attitude_error_deg=0.03,
        gyro_error_deg_s=0.02,
        attitude_bias_deg=0.01,
        gyro_bias_deg_s=0.01,
        dropout_period_steps=80,
        dropout_length_steps=1,
    ),
    controller_options=ControllerOptions(attitude_filter=True),
)
result = flight.run()
print(result["simulated_acceptance_passed"], result["rcs_propellant_remaining_kg"])
```

This is a custom experiment, separate from the fixed published evaluation.
Use `ControllerOptions(attitude_filter=False)` for the raw angular processing
variant. The other options are documented by the ablation table and source.

Install Python 3.12 and the tested dependency snapshot from the repository root:

```bash
python -m pip install -c requirements/constraints.txt -e ".[ui,video,dev]"
python -m scripts.evaluate_study --prepare --output-dir artifacts/study
python -m scripts.evaluate_study --run development --workers 4 --output-dir artifacts/study
python -m scripts.evaluate_study --freeze --output-dir artifacts/study
python -m scripts.evaluate_study --run final --workers 4 --output-dir artifacts/study
python -m scripts.evaluate_study --run ablation --workers 4 --output-dir artifacts/study
python -m scripts.evaluate_study --run convergence --workers 4 --output-dir artifacts/study
python -m scripts.evaluate_study --run sensitivity --workers 4 --output-dir artifacts/study
python -m scripts.analyze_study --campaign-dir artifacts/study --output-dir artifacts/evidence
python scripts/validate_evidence.py artifacts/evidence
```

Each campaign needs a new output directory. `--resume` continues an interrupted
phase using its existing completed rows and unchanged source. It does not rerun
or replace completed outcomes. The evidence validator uses only the Python
standard library and recomputes all aggregate results without running a flight.
It checks recorded evidence consistency, not the authenticity of physical data.

## References

- [NASA Standard 7009: Models and Simulations](https://standards.nasa.gov/standard/nasa/nasa-std-7009)
- [NASA 6DOF benchmark check cases](https://www.nasa.gov/centers-and-facilities/nesc/6dof-check-cases/)

These references inform verification practice. DM100 does not claim compliance,
hardware qualification or 3D capability.
