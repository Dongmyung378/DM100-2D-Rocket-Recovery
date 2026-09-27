# Model and Environment Specification

[Korean specification](spec.ko.md)

This document defines the state, control, coordinate, unit, event, and episode-record contracts shared by the simulator, controllers, datasets, and future learned policies. Any change to a sign or unit must update both the implementation and this specification.

## 1. Model scope

- 2D planar rigid body
- Translational degrees of freedom: `x`, `z`
- Rotational degree of freedom: `theta`
- Variable vehicle mass: `mass`
- Control inputs: throttle and thrust-vector gimbal angle
- Implemented effects: gravity, thrust, propellant consumption, point-position ground contact, and optional quadratic aerodynamic drag with wind
- Evaluation-only effects: Gaussian sensor noise, actual thrust-scale error, and first-order throttle lag
- Fixed parameters: moment of inertia and engine lever arm

## 2. Inertial frame

~~~text
                       +z (up)
                        ^
                        |       body axis
                        |      /   theta > 0
                        |     /
                        |    /
                  COM   o----------------> +x (right)
                        |
                        |
                   ground: z = 0
~~~

- Origin: landing target on the ground
- `+x`: right on screen
- `+z`: upward
- Gravity vector: `(0, -g)`
- Ground: `z = ground_z_m`, with a default of `0 m`

## 3. Angles and signs

- `theta = 0`: the vehicle longitudinal axis is aligned with `+z`
- `theta > 0`: the nose tilts clockwise toward `+x`
- `omega = d(theta)/dt`: positive in the same direction as `theta`
- `gimbal_angle = 0`: thrust is aligned with the vehicle axis
- `gimbal_angle > 0`: the thrust vector rotates from the vehicle axis toward `+theta`
- `gimbal_angle` describes the force-vector direction, not mechanical nozzle deflection

The inertial thrust-vector angle is `theta + gimbal_angle`:

~~~text
thrust_x = thrust * sin(theta + gimbal_angle)
thrust_z = thrust * cos(theta + gimbal_angle)
~~~

The equations of motion are:

~~~text
x_dot     = vx
z_dot     = vz
vx_dot    = thrust_x / mass
vz_dot    = thrust_z / mass - g
theta_dot = omega
omega_dot = torque / moment_of_inertia
mass_dot  = -thrust / (specific_impulse * standard_gravity)
~~~

Applied thrust, torque, and mass flow become zero when the vehicle reaches dry mass:

~~~text
if mass <= dry_mass:
    applied_thrust = 0
    torque = 0
    mass_dot = 0
~~~

`standard_gravity` converts specific impulse in seconds to effective exhaust velocity in meters per second. It is distinct from the environmental `gravity` value.

The engine application point lies below the center of mass on the vehicle axis. For engine lever arm `l`, thrust `T`, and gimbal angle `delta`, the torque convention is:

~~~text
torque    = -l * T * sin(delta)
omega_dot = torque / moment_of_inertia
~~~

A positive gimbal therefore produces negative angular acceleration, and a negative gimbal produces positive angular acceleration.

## 4. State vector

The array order is part of the public interface and must not change silently.

| Index | Name | Meaning | Internal unit |
|---:|---|---|---|
| 0 | `x` | Horizontal position | m |
| 1 | `z` | Vertical position | m |
| 2 | `vx` | Horizontal velocity | m/s |
| 3 | `vz` | Vertical velocity | m/s |
| 4 | `theta` | Attitude from the vertical axis | rad |
| 5 | `omega` | Angular velocity | rad/s |
| 6 | `mass` | Current vehicle mass | kg |

~~~text
state = [x, z, vx, vz, theta, omega, mass]
~~~

## 5. Control vector

| Index | Name | Meaning | Internal unit or range |
|---:|---|---|---|
| 0 | `throttle` | Fraction of maximum thrust | dimensionless, `[0, 1]` |
| 1 | `gimbal_angle` | Thrust-vector angle from the vehicle axis | rad |

~~~text
control = [throttle, gimbal_angle]
~~~

Human-readable angle limits and initial values in YAML use the `_deg` suffix. Conversion to radians happens once at the simulator boundary; internal dynamics, logs, and future datasets use radians.

Controller commands are limited before entering the dynamics:

~~~text
applied_throttle = clip(commanded_throttle, throttle_min, throttle_max)
applied_gimbal   = clip(commanded_gimbal, -gimbal_limit, +gimbal_limit)
~~~

## 6. Invariants

- `mass >= dry_mass`
- `0 <= throttle <= 1`
- `abs(gimbal_angle) <= gimbal_limit`
- Every state and control value is finite
- Time step, maximum time, gravity, mass, and thrust parameters are positive
- Initial mass is at or above dry mass

## 7. Landing-success criteria

The pre-contact state must satisfy every configured limit:

| Criterion | Default limit |
|---|---:|
| Horizontal position error | `abs(x) <= 1.0 m` |
| Horizontal speed | `abs(vx) <= 1.0 m/s` |
| Vertical speed | `abs(vz) <= 2.0 m/s` |
| Attitude | `abs(theta) <= 5 deg` |
| Angular speed | `abs(omega) <= 5 deg/s` |
| Mass | `mass >= dry_mass` |

These values are initial project criteria stored in `configs/default.yaml`. Controller comparisons use the same seeds and initial conditions.

## 8. Environment API

Create `RocketLandingEnv(config)` with a dictionary returned by `load_config`.

- `observation, info = reset(seed=None, options=None)`
- `next_state, reward, terminated, truncated, info = step(action)`
- `episode_log`: isolated, JSON-compatible copy of the current episode record
- `save_episode(path)`: write a new record without overwriting an existing file

Calling `step` before `reset` or after termination raises an error. Calling `reset` starts a new episode, so save any required record first. The environment rejects non-finite values, invalid action shapes, below-ground initial states, and initial mass below dry mass.

Observations are seven-element `float64` state arrays. Actions are `float64` `[throttle, gimbal_angle]` arrays in SI-compatible units. Position and velocity deliberately have no arbitrary finite bounds, so Gymnasium's checker can emit advisory warnings about infinite observation-space limits. Non-finite observations are never allowed.

## 9. Initial conditions

By default, `reset` independently samples configured fields from `initial_state_sampling`. YAML stores the same units as `initial_state`, except that attitude and angular velocity are expressed in degrees before conversion to radians.

- `options={"randomize": False}`: use the nominal initial state
- `options={"initial_state": [...]}`: use an exact seven-element SI/radian state
- `reset(seed=n)`: initialize the environment random generator
- `reset(seed=None)`: continue its existing random-generator state

The same seed and action sequence produce the same record. Callers that use `action_space.sample()` must seed the action space separately. An exact initial state and recorded action sequence can reproduce the physics without the original reset seed.

## 10. Termination

Ground contact uses point-position `z`. The returned contact observation is never below ground. Attitude is wrapped with `atan2(sin(theta), cos(theta))` only for contact evaluation; internal states and records retain the unwrapped angle.

| Outcome | Rule |
|---|---|
| `running` | No terminal event |
| `success` | Position, velocity, attitude, and angular-speed limits all pass |
| `hard_landing` | Position, attitude, and angular speed pass, but `vx` or `vz` fails |
| `crash` | Position, attitude, or angular speed fails |
| `fuel_depletion` | Dry mass is reached in flight |
| `timeout` | Maximum simulation time is reached |

A state that begins on the ground or at dry mass is evaluated at `t=0` on the first step. Simultaneous events use ground contact, fuel depletion, then timeout priority. Sequential events stop at the earliest event time. Fuel depletion terminates the environment immediately; the dynamics function itself can still model a later ballistic segment when called directly.

`success`, `hard_landing`, `crash`, and `fuel_depletion` set `terminated=True`. Only `timeout` sets `truncated=True`. Success reward is `+1`, physical failure reward is `-1`, and running or timeout reward is `0`. A final learning reward has not been designed yet.

## 11. Events inside a step

One action remains constant for `simulation.dt_s`. Internal integration intervals are at most 0.02 seconds, and the final interval is shortened to hit the maximum simulation time or fuel-depletion time exactly.

Each internal interval checks for ground crossing. Brent root search locates the contact time. Because a powered trajectory can descend and rise while both interval endpoints remain above ground, a conservative acceleration bound identifies intervals that also need a minimum-altitude search.

The root-search time tolerance is `1e-12 s`; final physical accuracy still depends on the selected Euler or RK4 integrator and its integration interval. A descending endpoint within floating-point tolerance of the ground is treated as contact to prevent a coincident event from being misclassified as timeout.

## 12. Episode record schema version 1

| Field | Contents |
|---|---|
| `config` | Complete configuration used for the episode |
| `seed_argument` | Seed passed to reset, or `null` |
| `rng_state_after_reset` | Generator state immediately after initial-state creation |
| `state_names`, `state_units` | State order and units |
| `action_names`, `action_units` | Action order and units |
| `initial_state` | Seven-element SI state at `t=0` |
| `steps` | Ordered step results |
| `outcome` | Latest outcome |

Each step contains `index`, `time_s`, `state`, `commanded_action`, `clipped_action`, `thrust_start_n`, `thrust_end_n`, `fuel_used_kg`, `reward`, `terminated`, `truncated`, `outcome`, and `is_success`.

`fuel_used_kg` is the difference between initial and current mass. `clipped_action` contains actuator-limited commands; `thrust_start_n` and `thrust_end_n` show whether dry-mass cutoff actually removed thrust. Returned states, info dictionaries, and logs are copied so external modification cannot mutate environment state.

## 13. Replay and visualization

`save_episode_data` writes the same episode record as JSON or compressed NPZ. NPZ stores the source JSON plus validation arrays for `times_s`, `states`, `actions`, and `thrust_n`, without object pickle. `load_episode_data` checks those arrays against the source record. Both formats replay stored states directly rather than rerunning physics.

Time-series plots show position, velocity, attitude, angular speed, mass, throttle, and gimbal. The 2D animation shows the x-z trajectory, vehicle attitude, thrust vector, ground, and the `x=0` landing target. Vehicle geometry is illustrative and does not model real dimensions or landing legs. GIF is supported directly; MP4 uses a system FFmpeg executable or the bundled executable from the `video` extra. Output functions never overwrite an existing file.

## 14. Vertical suicide-burn baseline

`SuicideBurnController` handles vertical descent with zero horizontal and attitude error. The first ignition-height estimate holds mass constant during the burn:

~~~text
net_deceleration = max_thrust / mass - gravity
stopping_distance = (downward_speed^2 - target_speed^2) / (2 * net_deceleration)
~~~

`estimate_suicide_burn` exposes that reference calculation. The executable controller refines ignition height with the analytic variable-mass vertical solution. If available propellant is insufficient or maximum thrust cannot overcome gravity, the estimate is `infeasible` and the controller commands immediate maximum thrust.

When ideal ignition lies within one 0.02-second control interval, the first burn command uses fractional throttle to reduce timing quantization. Maximum thrust remains latched after ignition. Ignition above the requirement is `early_burn`, below it is `late_burn`, and within tolerance it is `on_time`.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

This baseline does not compensate for wind, horizontal error, tilt, sensor delay, or engine delay. The 2 m/s contact limit is a simulation criterion, not a hardware structural-safety guarantee.

## 15. Vertical-velocity PID baseline

`VerticalVelocityPIDController` generates an altitude-dependent vertical-speed reference and tracks it with throttle PID. Its present evaluation scope holds horizontal position, horizontal velocity, attitude, and angular velocity at zero.

~~~text
speed_limit = sqrt(touchdown_speed^2 + 2 * profile_deceleration * altitude)
target_vz = -min(max_descent_speed, speed_limit)
error = target_vz - current_vz
~~~

The default profile uses a 1 m/s touchdown target, 30 m/s maximum descent speed, and 3 m/s² nominal deceleration. Feed-forward throttle compensates for gravity and reference acceleration before PID correction:

~~~text
feed_forward = mass * (gravity + target_acceleration) / max_thrust
raw_throttle = feed_forward + Kp * error + Ki * integral + Kd * error_rate
throttle = clip(raw_throttle, throttle_min, throttle_max)
~~~

Default gains are `Kp=0.08`, `Ki=0.001`, and `Kd=0.01`. Integrated error is limited to ±10 m. Integration is cancelled for a step when positive error would push an upper-saturated command higher or negative error would push a lower-saturated command lower. `reset` clears the integral, previous error, and last-command telemetry.

The configured evaluation range independently samples altitude from 80 to 120 m, vertical speed from -25 to -15 m/s, and mass from 950 to 1000 kg. With seed 20260910, all 100 sampled cases landed successfully. Mean touchdown speed was 1.1010 m/s, the 95th percentile was 1.1938 m/s, and mean propellant use was 37.4923 kg.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

These results apply only to the current deterministic, vertical, wind-free simulation range. Horizontal and attitude control, wind, model error, sensor noise, and actuator delay remain outside this baseline.

## 16. Horizontal-position and attitude controller

`HorizontalAttitudeController` is a cascaded, non-learning controller. It accepts the current state and a base throttle from a separate vertical controller. This interface keeps horizontal-attitude logic independent until the integrated-controller stage.

The outer loop calculates horizontal acceleration from position and velocity error:

~~~text
raw_ax = Kp_x * (target_x - x) - Kd_x * vx
target_ax = clip(raw_ax, -max_ax, max_ax)
target_theta = clip(atan2(target_ax, gravity), -max_tilt, max_tilt)
~~~

The default outer-loop gains are `Kp_x=0.05 s^-2` and `Kd_x=0.45 s^-1`. Horizontal acceleration is limited to 2.5 m/s² and target tilt to 15 degrees.

The inner loop calculates desired angular acceleration and inverts the documented torque convention. Positive gimbal produces negative angular acceleration:

~~~text
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
sin(gimbal) = -target_alpha * inertia / (lever_arm * thrust)
~~~

The default inner-loop gains are `Kp_theta=4.0 s^-2` and `Kd_theta=3.0 s^-1`. The inverse is clipped to its mathematical domain and then to the 15-degree vehicle gimbal limit. If base throttle is zero, gimbal is zero because thrust vectoring cannot produce torque.

Tilting reduces vertical thrust by `cos(theta + gimbal)`. With coupling compensation enabled, the output throttle is iteratively adjusted so its vertical component matches the requested base throttle, subject to the 0 to 1 actuator range.

The reproducible evaluation uses 100 wind-free, fixed-duration hover trials at 100 m. Initial horizontal error is at least 5 m within -20 to 20 m; horizontal speed, attitude, angular rate, and mass are sampled from the configured ranges. With seed 20260914, all 100 trials completed and reduced absolute horizontal error. Mean error fell from 12.4712 m to 5.4894 m. Maximum target tilt was 9.7647 degrees, maximum gimbal was 7.0393 degrees, and maximum altitude deviation was 0.0149 m.

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

This isolated result establishes wind-free convergence toward the pad without claiming touchdown success. The integrated controller below combines these commands and evaluates complete landings.

## 17. Integrated landing controller

`IntegratedLandingController` combines the vertical-velocity PID throttle with horizontal-position and attitude gimbal control. The simulator advances at 0.02-second intervals, while the controller updates every 0.1 seconds. The latest command is held for the four intermediate simulation steps. The controller rejects a control interval that is shorter than, or not an integer multiple of, the simulation interval.

Altitude selects one of two independently stateful control phases. Above 30 m, the approach phase uses a 1.0 m/s touchdown target, 1.5 m/s² profile deceleration, vertical PID gains `(0.08, 0.001, 0.01)`, horizontal gains `(0.5, 1.7)`, and a 15-degree target-tilt limit. At or below 30 m, the terminal phase uses a 0.8 m/s touchdown target, 1.0 m/s² profile deceleration, vertical PID gains `(0.1, 0.001, 0.015)`, horizontal gains `(0.3, 1.35)`, and a 5-degree target-tilt limit. The terminal limits prioritize an upright, low-rate contact.

The vertical controller first produces base throttle. The selected horizontal-attitude controller then calculates gimbal and compensates the tilted thrust's vertical component. The combined command is finally limited relative to the previous control update:

~~~text
abs(throttle[k] - throttle[k-1]) <= 2.0 * control_interval
abs(gimbal[k] - gimbal[k-1]) <= deg2rad(60) * control_interval
~~~

The first calculated command initializes the held action because no previous controller output exists. Later updates expose separate throttle and gimbal slew-limit flags. Time must be finite, nonnegative, and nondecreasing. `reset` clears both phases, held commands, update counters, and scheduling state.

The medium-difficulty evaluation samples 100 wind-free cases with horizontal error from 3 to 15 m on either side of the pad, altitude from 80 to 120 m, horizontal speed from -2 to 2 m/s, vertical speed from -25 to -15 m/s, attitude from -5 to 5 degrees, angular rate from -2 to 2 deg/s, and mass from 950 to 1000 kg. With seed 20260914, all 100 cases satisfied every touchdown limit.

Mean and maximum absolute touchdown position were 0.4516 m and 0.9440 m. Mean and maximum horizontal speed were 0.1476 m/s and 0.3526 m/s. Mean and maximum vertical speed were 0.7819 m/s and 0.7827 m/s. Mean and maximum attitude were 0.7559 degrees and 1.4386 degrees, while angular-rate values were 1.1004 deg/s and 2.4399 deg/s. Mean propellant use was 53.0159 kg. Maximum observed slew was 1.7784 throttle units per second and 60.0000 deg/s gimbal.

~~~powershell
python scripts/evaluate_integrated_control.py --episodes 100 --seed 20260914
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid
~~~

This controller is a deterministic wind-free baseline. Disturbance rejection, uncertainty, sensor noise, actuator delay, and hardware safety remain outside this result.

## 18. Baseline tuning automation

`tune_integrated_controller.py` searches three scale factors applied to both integrated-controller phases: horizontal position gain, horizontal velocity gain, and descent-profile deceleration. It supports a Cartesian `grid` search and a uniform seeded `random` search. Both methods always include the unscaled `(1, 1, 1)` reference. Scaling is performed on an isolated configuration copy, so the input configuration is not mutated.

Each candidate is evaluated on the same training batch. Selection minimizes the following scalar objective:

~~~text
failure_rate = 1 - success_rate
fuel_fraction = mean_fuel_used / (initial_mass - dry_mass)
normalized_landing_error = mean(
    mean_abs_x / x_limit,
    mean_abs_vx / vx_limit,
    mean_abs_vz / vz_limit,
    mean_abs_theta / theta_limit,
    mean_abs_omega / omega_limit,
)
score = 1000 * failure_rate + 5 * fuel_fraction + 10 * normalized_landing_error
~~~

The failure term dominates the two secondary terms, while normalized fuel and touchdown error make their scales explicit. Objective weights, search ranges, candidate count, episode counts, and seeds are configuration values. Invalid ranges, nonpositive counts, equal train and validation seeds, and an objective with neither secondary term are rejected during configuration loading.

The default Cartesian grid contains 12 scale combinations and adds the unscaled reference for 13 total candidates. Training uses 16 initial conditions from seed 20260912. Only the training objective selects the winner. The winner is then evaluated once on 100 independently sampled initial conditions from seed 20260913.

The selected scales were `(1.10, 0.95, 1.10)`. All 16 training cases landed safely; the objective was 3.716630 and mean propellant use was 51.1532 kg. The unscaled candidate also landed all 16 cases but scored 4.179125 and used 52.9862 kg. Independent validation produced 100 safe landings in 100 cases, 51.9394 kg mean propellant use, 0.3942 m mean absolute touchdown position, and 0.7802 m/s mean absolute vertical touchdown speed.

~~~powershell
python scripts/tune_integrated_controller.py --report artifacts/tuning.json --best-config artifacts/best-integrated.yaml
~~~

The JSON report contains all candidate evaluations, objective components, seeds, and the independent validation result. The YAML output contains the complete selected configuration plus selection metadata and passes the same configuration validation as the tracked default. Existing output files are never overwritten.

## 19. Isolated PID disturbance evaluation

`evaluate_pid_disturbances.py` applies the selected Day 12 gain scales `(1.10, 0.95, 1.10)` and changes one disturbance model at a time. Every strength within a curve reuses the same 30 initial conditions from seed 20260915. Random wind direction, gust start time, and sensor-noise stream are also paired by episode across strength levels. Zero-strength results are required to match across all five curves.

### 19.1 Wind and aerodynamic drag

Wind is expressed as a two-dimensional inertial air-velocity vector. With vehicle velocity `v` and air velocity `w`, the relative velocity and quadratic drag force are:

~~~text
v_relative = v - w
F_drag = -0.5 * air_density * drag_coefficient * reference_area
         * norm(v_relative) * v_relative
~~~

The force acts through the modeled center of mass and therefore produces no aerodynamic torque. It is evaluated at each RK4 stage. Passing no wind model preserves the earlier ideal dynamics exactly and applies no aerodynamic force.

A constant-wind episode holds its sampled positive or negative horizontal direction for the entire flight. A gust is a horizontal half-sine pulse with a two-second duration, a start time sampled uniformly from 2 to 6 seconds, and a direction sampled independently for each episode. The same direction and start time are reused when comparing amplitudes.

### 19.2 Sensor noise

Sensor noise is added only to the state observed by the controller; contact detection and physical propagation use the true state. Independent zero-mean Gaussian samples use the following one-times standard deviations: x and z `0.5 m`, vx and vz `0.1 m/s`, attitude and angular rate `0.25 deg` and `0.25 deg/s`, and mass `0.5 kg`. A dimensionless strength multiplies all seven values. Noisy altitude and mass are clipped at ground and dry mass so the observation remains inside the controller's physical domain.

### 19.3 Thrust error and engine lag

Thrust scale changes the simulator's actual maximum thrust while the controller retains its nominal model. Propellant flow uses the actual thrust and the unchanged specific impulse. A scale of `0.95` therefore means that every throttle setting produces 95% of the thrust expected by the controller.

Engine lag is a first-order throttle response integrated exactly over each 0.02-second simulation interval:

~~~text
response = 1 - exp(-dt / time_constant)
actual_throttle += response * (commanded_throttle - actual_throttle)
~~~

The initial actual throttle is zero. Gimbal remains immediate because gimbal-servo dynamics are not part of this disturbance. A zero time constant reproduces the command exactly.

### 19.4 Performance curves and collapse rule

The first tested nonzero disturbance strength with a safe-landing rate below 95% is reported as the collapse point. With 30 episodes per level, each case changes the observed success rate by 3.33 percentage points. Results at the measured collapse points were:

| Disturbance | Last passing strength | Collapse strength | Success rate | Outcomes at collapse |
|---|---:|---:|---:|---|
| Constant wind | 10 m/s | 20 m/s | 86.7% | 26 success, 4 crash |
| Gust amplitude | 40 m/s | 60 m/s | 86.7% | 26 success, 4 crash |
| Sensor-noise scale | 1x | 2x | 86.7% | 26 success, 4 crash |
| Actual thrust scale | 1.00x | 0.95x | 93.3% | 28 success, 2 crash |
| Throttle-lag time constant | 0.2 s | 0.5 s | 90.0% | 27 success, 3 crash |

The shared zero-disturbance result was 29 safe landings and one crash. This new seed exposes a boundary case that was absent from the separate Day 12 validation batch. The values above are tested grid thresholds, not exact physical limits or hardware certifications.

~~~powershell
python scripts/evaluate_pid_disturbances.py --report artifacts/disturbances.json --plot artifacts/disturbances.png
~~~

The JSON contains every level's outcome counts, terminal-state errors, normalized error, and propellant use. The PNG contains five safe-landing-rate curves, the 95% threshold, and the detected collapse strength. Existing files are never overwritten.

## 20. Frozen Week 2 baseline protocol

`configs/pid-baseline-v1.yaml` is the comparison baseline for all later methods. It materializes the Day 12 selected scales in the integrated-controller values. The approach-phase position gain, velocity gain, and profile deceleration are `0.55`, `1.615`, and `1.65 m/s^2`; the terminal values are `0.33`, `1.2825`, and `1.1 m/s^2`. The controller mapping has the frozen digest `80d4cbedbc545457072dc08f308ee408fd25996e3df500520135ec0452818a8a`.

The fixed evaluation set uses sampler `integrated_uniform_v1`, 1,000 episodes, and seed 20260914. The sampler draws the sign and magnitude of horizontal position separately so every state begins at least 3 m from the target. It then draws the remaining six state components independently from the ranges in the integrated-controller evaluation configuration. Angles are converted to radians before hashing or simulation.

`initial_condition_sha256` hashes a schema prefix, the two matrix dimensions as little-endian unsigned 64-bit integers, and the contiguous state matrix as little-endian IEEE 754 float64 values. The expected digest is `a9593a383acc0017738ed383a123ca743a32bce46c64d249b08ca577a448a952`. `frozen_baseline_initial_states` regenerates the matrix and verifies both the controller and state digests. It returns a read-only array and rejects controller edits, seed or range drift, a changed sampler, or a changed episode count.

The nominal evaluation produced the following result:

| Metric | Value |
|---|---:|
| Safe landings | 999 / 1,000 (99.9%) |
| Crash | 1 / 1,000 |
| Mean / maximum absolute touchdown x | 0.4098 / 1.3494 m |
| Mean / maximum absolute touchdown vx | 0.1656 / 0.5915 m/s |
| Mean / p95 / maximum absolute touchdown vz | 0.7802 / 0.7810 / 0.7814 m/s |
| Mean / maximum absolute touchdown attitude | 0.5991 / 3.2498 deg |
| Mean / maximum absolute touchdown angular rate | 1.3178 / 3.6500 deg/s |
| Mean propellant use | 51.4460 kg |

The 99.9% result passes the configured 70% Week 2 acceptance gate. The single crash exceeded the 1 m horizontal position limit; the reported maxima for horizontal speed, vertical speed, attitude, and angular rate remained inside their respective contact limits. This finite simulation result is not a universal robustness claim.

~~~powershell
python scripts/evaluate_frozen_baseline.py --output-dir artifacts/week2-baseline
~~~

The command writes a JSON benchmark report, two replayable episode logs, and two GIFs. The representative success is initial-condition index 0 under nominal wind. The representative failure reuses index 0 with a 40 m/s constant horizontal wind, which produces a 2.5911 m touchdown position error and a crash. This stress case demonstrates a known failure mode and is not counted in the nominal success rate. The command checks all destination names first and never overwrites an existing output.

## 21. Day 15 optimal-control teacher formulation

The seven-state teacher problem is defined in `LandingOptimalControlProblem`. Day 16 first solved its vertical subset, Day 17 solved an ideal-vector translational subset, and Day 18 solves the complete planar problem. The model is nominal, wind-free, sensor-perfect, and uses instantaneous actuators. It uses the same planar state ordering, units, force direction, torque sign, variable-mass law, and landing thresholds as the simulator. The simulator's optional aerodynamic drag is absent because no wind model is supplied in nominal rollouts. The nonlinear-program equations do not clip actions or switch thrust off at dry mass; hard actuator bounds and a 1 kg propellant reserve keep feasible trajectories strictly inside that smooth region.

The state is `X = [x, z, vx, vz, theta, omega, m]` and the control is `U = [q, delta]`, where `q` is dimensionless throttle and `delta` is the gimbal angle in radians. The optimization uses `N = 100` piecewise-constant control intervals. Final time `T` is a decision variable in `[5, 25] s`, with mesh step `h = T/N`. The initial state is fixed to the supplied state. Units are SI throughout; `theta` and `delta` are positive clockwise toward `+x` under the project convention.

| Mathematical quantity | Equation or bound | Code input and output |
|---|---|---|
| Thrust | `F = T_max q` | `dynamics(X, U)` receives throttle `q`; returns a 7-vector derivative |
| Position | `dx/dt = vx`, `dz/dt = vz` | derivative elements 0-1 |
| Translation | `dvx/dt = F sin(theta + delta)/m`; `dvz/dt = F cos(theta + delta)/m - g` | derivative elements 2-3; matches `state_derivative` in the smooth feasible region |
| Rotation | `dtheta/dt = omega`; `domega/dt = -L F sin(delta)/I` | derivative elements 4-5; negative torque sign matches the simulator |
| Propellant | `dm/dt = -F/(Isp g0)` | derivative element 6 |
| Multiple-shooting defect | `X[k+1] - RK4(X[k], U[k], T/N) = 0` | `rk4_defects(states, controls, T)` returns an `N x 7` matrix |
| Path margins | `z-ground >= 0`; `m-dry_mass-1 kg >= 0`; `q_min <= q <= q_max`; `|delta| <= delta_max`; `|theta| <= 20 deg`; `|omega| <= 30 deg/s` | `path_margins(X, U)` returns ten nonnegative-required values; state bounds also apply at the final node |
| Terminal landing | `z(T)=ground`; `|x(T)-target_x|<=1 m`; `|vx(T)|<=1 m/s`; `-2<=vz(T)<=0 m/s`; `|theta(T)|<=5 deg`; `|omega(T)|<=5 deg/s` | `terminal_violations(X_T)` returns six normalized nonnegative values; zero means each terminal bound holds |

The Stage A violation of the terminal-height equality is normalized by a fixed 1 m scale. The terminal descent constraint intentionally excludes upward ground contact. The simulator uses point contact and has no landing-leg or impact-load model. The path constraints apply at shooting nodes; a solver result must still be rolled out in the event-driven simulator because node feasibility alone cannot rule out between-node ground penetration or other transcription error.

The objective is staged rather than relying on a single large weighted penalty. In **Stage A**, initial state, dynamics, duration, actuator, altitude, and propellant constraints are hard; terminal constraints are temporarily relaxed. The solver minimizes the sum of squares of the six normalized positive terminal violations. Stage A passes only when the maximum normalized terminal violation is at most `0.001`. This gives a measurable feasibility target and prevents low fuel use from compensating for a missed landing. Stage B starts from that solution, imposes the terminal constraints as hard bounds, and minimizes:

~~~text
fuel_fraction = (m_initial - m_final) / (m_initial - dry_mass)
touchdown_error = mean([
    ((x_final - target_x) / x_limit)^2,
    (vx_final / vx_limit)^2,
    ((vz_final - target_vz) / vz_limit)^2,
    (theta_final / theta_limit)^2,
    (omega_final / omega_limit)^2,
])
control_step = T / N
smoothness = mean over k=1..N-1 of [
    ((q[k] - q[k-1]) / (control_step * 2 s^-1))^2
    + ((delta[k] - delta[k-1]) / (control_step * deg2rad(60) s^-1))^2
]
J = 1.0 * fuel_fraction + 0.1 * touchdown_error + 0.01 * smoothness
~~~

The target terminal vertical velocity is `-0.8 m/s`, within the simulator's `[-2, 0] m/s` safe-descending interval. Every cost term is dimensionless. Stage B's fuel fraction uses the propellant available in the particular initial state, not the vehicle's nominal 250 kg capacity. The two control-rate scales come from the integrated controller's configured slew references. Dividing by the mesh step makes the same physical control ramp carry the same smoothness cost at different node counts. These references are soft objective scales, not hard rate bounds. A candidate is not a teacher trajectory until a solver reports convergence, constraint residuals are checked, and independent simulator rollout agrees.

## 22. Day 16 vertical CasADi/IPOPT teacher

Day 16 solves only the vertical subset of the Day 15 problem. Its state is `Y=[z,vz,m]`, control is throttle `q`, and dynamics are `dz/dt=vz`, `dvz/dt=T_max*q/m-g`, and `dm/dt=-T_max*q/(Isp*g0)`. The initial horizontal and attitude state must be zero. The 100 constant-control intervals, free final time in `[5,25] s`, dry-mass-plus-1-kg reserve, actuator bounds, and landing vertical-speed limit come from the same configuration as the planar formulation. Each interval has a CasADi RK4 continuity equality, making this direct multiple shooting rather than single shooting.

A constant-acceleration estimate supplies the initial duration and throttle profile; a numerical rollout supplies the initial state nodes. Stage A keeps the initial state, RK4 dynamics, duration, altitude, fuel, and throttle bounds hard. Two nonnegative terminal slacks permit temporary height and vertical-speed violations, and the solver minimizes their normalized squares. Stage B starts from the Stage A solution, fixes final height to ground, constrains final speed to `[-2,0] m/s`, and minimizes the Day 15 vertical restriction of fuel fraction, touchdown-speed error, and throttle change. A stage must meet the configured `0.001` normalized feasibility threshold before its output is accepted.

The JSON report records each IPOPT return status, iteration count, objective, maximum normalized hard and terminal violation, and each RK4 state defect in physical units. The optimized controls are separately integrated with the existing planar simulator at `0.02 s` steps. The report checks ground height, terminal descent speed, fuel reserve, minimum altitude, and disagreement at every optimization node; the PNG overlays this rollout and the NLP nodes. Solver or feasibility failures write a diagnostic JSON report without a plot. This independent nominal rollout does not establish performance under wind, sensor error, engine lag, or a population of initial states.

~~~bash
python -m pip install -e ".[optimization]"
python scripts/solve_vertical_landing.py --output-dir artifacts/day16-vertical
python scripts/solve_vertical_landing.py --initial-z 110 --initial-vz -18 --initial-mass 980 --output-dir artifacts/day16-custom
~~~

The command refuses to overwrite existing files. The nominal case converges with a `5.000 s` duration, about `-0.832 m/s` simulator terminal velocity, and about `27.435 kg` of fuel use. These are one-case outputs, not a general success-rate estimate. Day 17 adds horizontal translation; Day 18 adds rotation and gimbal torque.

## 23. Day 17 planar translation teacher

Day 17 extends the solved vertical problem to `Y=[x,z,vx,vz,m]` with controls `U=[q,alpha]`. Here `alpha` is the **absolute thrust-vector angle from vertical**, not the physical gimbal angle. The model deliberately omits body attitude and angular velocity until Day 18. Its equations are `dx/dt=vx`, `dz/dt=vz`, `dvx/dt=T_max*q*sin(alpha)/m`, `dvz/dt=T_max*q*cos(alpha)/m-g`, and `dm/dt=-T_max*q/(Isp*g0)`. The physical state/action ordering and signs match the full simulator when its body angle is set to `alpha` and gimbal to zero for one interval.

The direct multiple-shooting transcription uses the same 100 intervals and free `[5,25] s` horizon as Day 16. Initial state, RK4 continuity, `z>=ground`, `m>=dry_mass+1 kg`, `q_min<=q<=q_max`, and `|alpha|<=15 deg` are hard constraints at shooting nodes. The 15-degree direct-vector limit is an explicit reduced-model assumption; it is not evidence that the body can rotate or that a gimbal can realize this command. Stage A relaxes only terminal position, height, horizontal speed, and descending vertical-speed limits with four nonnegative normalized slacks. Stage B fixes height to ground, imposes `|x-target_x|<=1 m`, `|vx|<=1 m/s`, `-2<=vz<=0 m/s`, and minimizes the vertical-planar restriction of the Day 15 fuel, touchdown, and control-change cost. Each stage is accepted only if its relevant normalized violations are at most `0.001`.

The default analytic seed estimates descent time from altitude and vertical speed, uses a cubic horizontal-position curve ending at the target with zero horizontal velocity, maps its required accelerations to bounded throttle and thrust angle, and integrates the resulting controls to supply consistent state nodes. With `--guess pid`, the existing full-model integrated PID controller runs from the same initial state. Its applied throttle and instantaneous net thrust direction (`theta+gimbal`) are sampled onto the optimization grid, clipped to the reduced-model control bounds, and re-integrated with the five-state model. The PID trajectory is only an initial guess; the NLP solution is checked independently and does not inherit a PID success claim.

After IPOPT converges, the saved controls are replayed through `simulate_planar` at the configured fine step. For each control interval, the replay sets body heading to the optimized `alpha` and gimbal to zero, then compares all five translational states at every shooting node. This tests the ideal-vector translational dynamics and between-node altitude/fuel behavior, **not** physical rotation, gimbal slew, or touchdown attitude. The report contains both solver statuses, objectives, iteration counts, hard/terminal violations, five physical-unit RK4 defects, and the replay comparison. Solver or feasibility failures save a diagnostic JSON without a plot. Existing output files are never overwritten.

~~~bash
python scripts/solve_translation_landing.py --output-dir artifacts/day17-translation
python scripts/solve_translation_landing.py --guess pid --output-dir artifacts/day17-pid
python scripts/solve_translation_landing.py --initial-x -12 --initial-vx 1 --output-dir artifacts/day17-custom
~~~

The default single-case start is `x=10 m`, `z=100 m`, `vx=0`, `vz=-20 m/s`, `m=1000 kg`. Both the analytic and PID-seeded runs converge. The analytic-seeded ideal-vector replay ends at approximately `x=0.003 m`, `vx=-0.005 m/s`, `vz=-0.902 m/s`, consuming `27.605 kg` in `5.000 s`. This demonstrates horizontal-error correction for one initial condition, not a distribution-level success rate. Day 18 removes the heading-reset assumption and solves the physical attitude and gimbal trajectory.

## 24. Day 18 full planar 3-DoF teacher

Day 18 solves the complete state `X=[x,z,vx,vz,theta,omega,m]` with physical controls `U=[q,delta]`. Unlike Day 17, `delta` is the gimbal angle relative to the body. Translation uses the inertial thrust direction `theta+delta`; rotation uses `dtheta/dt=omega` and `domega/dt=-L*T_max*q*sin(delta)/I`. The symbolic CasADi derivative was checked against the existing simulator derivative under the same state and action.

The direct multiple-shooting problem retains 100 piecewise-constant control intervals and a free final time in `[5,25] s`. Every shooting node enforces altitude, dry-mass-plus-1-kg reserve, body tilt `|theta|<=20 deg`, and angular rate `|omega|<=30 deg/s`. Every control enforces physical throttle and `|delta|<=15 deg`. Stage A temporarily relaxes all six landing conditions with normalized nonnegative slacks. Stage B fixes height to ground and imposes the configured position, velocity, attitude, and angular-rate landing limits as hard constraints before minimizing the original fuel, touchdown, and control-change objective.

The integrated PID controller runs once from the requested initial state to provide a practical initial control sequence. Those controls are sampled on the optimization mesh and reintegrated with the seven-state RK4 model, so the initial node sequence is dynamically consistent. The PID outcome is saved as provenance only; an accepted teacher still requires IPOPT convergence, independently recomputed hard and terminal residuals below `0.001`, and a separate fine-step simulator replay.

Failure reports identify the failed stage, IPOPT status, largest named violation, the full hard-constraint breakdown, all six terminal violations, and state-by-state RK4 defects. A solver failure therefore distinguishes missed altitude, fuel, actuator, tilt, angular-rate, duration, dynamics, and terminal conditions instead of recording only a generic infeasible status.

The independent replay carries `theta` and `omega` continuously between intervals and applies the optimized gimbal directly through `simulate_planar`; it never resets heading. Validation checks all terminal conditions, between-node altitude, fuel reserve, tilt, angular rate, gimbal magnitude, and the disagreement at every shooting node. The report and plot refuse to overwrite existing files.

~~~bash
python scripts/solve_planar_landing.py --output-dir artifacts/day18-planar-3dof
python scripts/solve_planar_landing.py --initial-x -8 --initial-theta-deg -2 --initial-omega-deg-s 1 --output-dir artifacts/day18-custom
~~~

The default case starts at `x=10 m`, `z=100 m`, `vx=0`, `vz=-20 m/s`, `theta=3 deg`, `omega=-1 deg/s`, and `m=1000 kg`. Both stages return `Solve_Succeeded`. With the current Day 19 objective, the full-model replay ends after `5.000 s` at approximately `x=0.012 m`, `vx=-0.023 m/s`, `vz=-0.974 m/s`, `theta=0.101 deg`, and `omega=-0.028 deg/s`, using `27.907 kg` of propellant. Peak body tilt is `15.749 deg`, peak angular rate is `21.680 deg/s`, and peak gimbal is `14.671 deg`. Maximum state-node disagreement is below `9e-7` in physical units. This is one nominal trajectory, not a success-rate, disturbance-robustness, actuator-bandwidth, structural-load, or hardware-safety claim.

## 25. Day 19 objective and numerical-stability study

Day 19 turns the full-planar Stage B objective into a mesh-comparable experiment. `ObjectiveScales` records the available fuel, five terminal landing limits, and the two configured control-rate references. The solver and the post-solve evaluator use the same normalized fuel, touchdown, and rate-based smoothness definitions. A regression test applies the same linear control ramp on two meshes and verifies equal smoothness cost.

The configured sweep keeps the fuel and touchdown weights fixed at `1.0` and `0.1`, then tests smoothness weights `0.001`, `0.01`, and `0.1`. The default initial condition is the Day 18 case. The measured trade-off was:

| Profile | Fuel used | Smoothness term | Peak throttle rate | Peak gimbal rate |
|---|---:|---:|---:|---:|
| Fuel priority | 27.858 kg | 0.11080 | 1.479/s | 50.013 deg/s |
| Baseline | 27.907 kg | 0.04708 | 0.697/s | 23.824 deg/s |
| Smooth control | 28.003 kg | 0.02661 | 0.478/s | 11.685 deg/s |

The baseline weights are then solved at `N=25, 50, 100, 200`. Each optimized control sequence is replayed through the existing event-driven simulator at `0.005 s`, rather than accepted from transcription nodes alone. Final-state differences between the optimization and replay must be no greater than `0.001 m` for `x,z`, `0.001 m/s` for `vx,vz`, `0.01 deg` for `theta`, `0.01 deg/s` for `omega`, and `0.001 kg` for mass. The replay must also satisfy the landing constraints and preserve altitude and propellant reserve.

| Intervals | Measured solve time | Largest final error / tolerance | Replay result |
|---:|---:|---:|---|
| 25 | 0.613 s | 0.231 | pass |
| 50 | 0.872 s | 0.0140 | pass |
| 100 | 1.476 s | 0.000870 | pass |
| 200 | 2.618 s | 0.00101 | pass |

All configured cases passed the Day 19 completion gate. The error is not required to decrease strictly at every refinement because solver tolerances, adaptive replay integration, and interpolation also contribute; both 100- and 200-interval disagreements are roughly one-thousandth of the allowed error. Solve times are measurements from one development run and are not performance guarantees. The study demonstrates nominal transcription agreement for one initial condition. It does not impose hard throttle/gimbal rate constraints; Day 20 extends the solver to a reproducible initial-condition batch.

~~~bash
python scripts/analyze_optimal_control.py --output-dir artifacts/day19-study
~~~

The command writes a JSON report containing all settings, objective components, solver iterations, physical final/node errors, and the completion gate. It also writes a four-panel PNG for the trade-off, normalized peak rates, mesh timing, and normalized replay error. Existing output files are never overwritten.

## 26. Day 20 multi-initial-condition teacher pipeline

Day 20 converts the single-case full-planar solver into a deterministic batch pipeline. `integrated_uniform_v1` samples the same seven-state medium-difficulty ranges used by the integrated-controller evaluation: 3-15 m absolute horizontal offset with both signs, 80-120 m altitude, -2 to 2 m/s horizontal velocity, -25 to -15 m/s vertical velocity, -5 to 5 deg attitude, -2 to 2 deg/s angular rate, and 950-1000 kg mass. The default batch contains 100 cases from seed `20260920`; its canonical matrix hash is `7a58dc505c4535e8f081544cea22de17743bff0d80f34debc9454b2029a9f9b9`.

The runner is deliberately sequential because warm-start provenance must be deterministic. Optimization executes inside a reusable spawned process instead of the parent process. The parent waits at most 30 seconds for each attempt and terminates the worker on timeout, so the policy is enforced rather than merely measured after a blocking solve. An idle worker is also recycled after 25 attempts to limit long-run solver memory. The worker returns only NumPy-backed trajectory data and structured diagnostics; its solver console output is suppressed.

The first case uses the integrated PID trajectory as its initial guess. After a case succeeds, the pipeline copies its throttle/gimbal sequence and duration, then reintegrates those controls from the next case's initial state to obtain dynamically consistent warm-start nodes. If that attempt fails solver or replay validation, or times out, the case is retried once with a new PID guess. A successful case is accepted only after the Stage A and B checks in the solver and a `0.01 s` independent replay. Replay acceptance uses the Day 19 per-state agreement tolerances and also checks terminal constraints, altitude, propellant reserve, body tilt, angular rate, and gimbal bounds.

Every attempt has an explicit status. Solver failures retain the failed stage, IPOPT return status, named worst constraint, complete hard/terminal residuals, and RK4 defects when available. Timeout records contain the configured limit and elapsed time. Worker crashes, protocol mismatches, unexpected exception types, and replay-validation failures are represented separately. The top-level `failures` list contains each failed case with all attempts, even when a run has zero failures.

~~~bash
python scripts/generate_teacher_pipeline.py --output-dir artifacts/day20-teacher-pipeline
~~~

The command refuses to overwrite either output and writes only two final files:

| File | Contents |
|---|---|
| `teacher-pipeline.json` | Settings, batch hash, completion gate, per-case attempts, compact solver metrics, replay checks, summary, and failure metadata |
| `teacher-trajectories.npz` | Case indices, initial states, `101 x 7` state nodes, `100 x 2` controls, 101 time nodes, and durations for accepted cases |

The default run executed all 100 requested cases in `126.246 s`. All 100 cases converged and passed replay on the first attempt. Case 0 used the PID guess and cases 1-99 used the previous successful trajectory, so retries, timeouts, and failures were all zero. Mean Stage A and B iteration counts were `37.54` and `29.09`. Mean propellant use was `27.349 kg`, with a `24.669-30.571 kg` range; optimized durations ranged from `5.000` to `5.762 s`. The largest final replay error was `0.00331` of its configured tolerance. The compressed dataset contains arrays with shapes `(100,101,7)` for state nodes and `(100,100,2)` for controls.

The roadmap completion gate asks only whether at least 100 initial conditions were processed, so solver success and the execution gate are reported separately. This run happened to produce 100 accepted nominal trajectories, but it does not establish robustness under wind, sensing error, engine lag, model mismatch, or initial conditions outside the configured ranges. Day 21 evaluates teacher quality against the PID baseline and selects representative trajectories.

## 27. Day 21 teacher validation and freeze

Day 21 defines `planar-teacher-v1` as the immutable Week 3 comparison protocol. Its canonical configuration digest covers the coordinate conventions, simulation and vehicle parameters, nominal environment, landing limits, integrated PID settings used for initial guesses, optimal-control formulation, and Teacher pipeline policy. The `pid-baseline-v1.yaml` digest is `dfea2326159997cc0fccc8977e13e316856c6eb2ec5e26f97cbb07196f648793`. The nominal set remains the 100-case, seed-`20260920` matrix with SHA-256 `7a58dc505c4535e8f081544cea22de17743bff0d80f34debc9454b2029a9f9b9`. The source report and trajectory archive are also frozen as `11ef6d9f823686483dcf1d4106415097cb57db43a6899973d757fe1aa7dfe414` and `c19f897d63c51e6a8b5f79f5dbe8594e7bbd2578221ea4873f4652149905a9dd`. Validation stops when any digest changes or when the test-set sampler, episode count, or seed differs from the generation pipeline.

Before numerical evaluation, the Day 20 bundle loader checks the report problem and schema versions, exact batch identity, complete ordered case records, explicit one-to-one failure accounting, summary counts, NPZ field set, array dimensions and dtypes, successful case indices, initial states, strictly increasing time nodes, durations, actuator limits, and JSON array metadata. The NPZ is opened with pickled objects disabled. This prevents a stale, reordered, incomplete, or manually changed dataset from being silently compared with PID.

Every accepted control sequence is then reintegrated again through `simulate_planar` at `0.01 s`. This is a new Day 21 calculation from the saved initial states, controls, and durations, not a reuse of the Boolean replay result stored on Day 20. The validator recomputes final and node disagreement, normalized terminal residuals, between-node altitude, propellant reserve, body tilt, angular rate, throttle, and gimbal checks. It also reports failure counts for every named check and the minimum physical margin over the batch.

All 100 Teacher solutions passed the fresh replay. The largest final and node error was `0.00331085` of its state-specific tolerance, and the largest normalized terminal violation was `2.63555e-6`, below the `0.001` feasibility threshold. The largest Stage B hard-constraint residual stored by the solver was `8.89050e-9`. Every named replay failure count was zero. The minimum raw tilt margin was `-0.00938 deg`; this small replay overshoot remains inside the explicit `0.01 deg` attitude tolerance used by the acceptance check. The minimum angular-rate margin was `1.84e-6 deg/s`, the minimum gimbal margin was `2.50e-6 deg`, and the minimum upper-throttle margin was `4.62e-8`. These values show that several solutions lie very close to bounds and should not be treated as actuator or modeling robustness margins.

The integrated PID is evaluated from the same 100 initial states with zero wind. It landed 99 cases and crashed in one. That failure exceeded the position limit by `1.582 m`, the horizontal-speed limit by `0.277 m/s`, and the angular-rate limit by `1.851 deg/s`; all other cases had no terminal-limit violation. The Teacher mean fuel use over all accepted cases was `27.349 kg`. On the 99 paired cases where PID also landed, Teacher averaged `27.347 kg` and PID averaged `51.907 kg`, giving a mean paired difference of `-24.560 kg` for Teacher.

Teacher pipeline wall time was `126.246 s`, total accepted-attempt time was `123.566 s`, and mean accepted-attempt time was `1.236 s`. The PID evaluation required `20.661 s` including physics simulation. PID command calculations used `5.609 s` in total, or approximately `0.483 ms` per controller update on this run. Offline nonlinear-program solve time and online controller-command time are different measurements, so the report keeps them separate and does not claim that their ratio is a real-time speedup.

The representative selection rule is fixed as the successful case whose fuel use is closest to the accepted-set median, with the smaller case index breaking a tie. It selected case 22: median fuel was `27.3636 kg` and selected fuel was `27.3641 kg`. Its controls are freshly replayed into the standard episode-log schema, then rendered as a 240-frame GIF. The last replay frame reaches `x=-0.0091 m`, `vx=0.0194 m/s`, `vz=-0.9645 m/s`, `theta=-0.0878 deg`, and `omega=0.0243 deg/s` after `5.000 s`, using `27.3641 kg` of propellant.

~~~bash
python scripts/validate_teacher.py
~~~

The command refuses to overwrite any output and writes exactly three files: `teacher-validation.json` with protocol verification, Teacher/PID metrics, constraint analysis, per-case PID results, independent replay results, failure accounting, comparison, and gate decisions; `representative-teacher.json` with the replayable selected episode; and `representative-teacher.gif` with its animation. The Week 3 gate requires at least 90% solver success, every accepted solution to pass both its stored and fresh replay checks, and all failed optimization cases to be reported separately. All six protocol and acceptance checks passed. The result is limited to the frozen, wind-free nominal sample; disturbance robustness, actuator bandwidth, model mismatch, hard control-rate constraints, and hardware safety remain unproven.

## 28. Day 22 dataset contract and split design

Day 22 defines the inputs to offline Behavior Cloning before producing a large trajectory collection. `planar-bc-v1` uses the ordered state `[x,z,vx,vz,theta,omega,mass]` in `m,m,m/s,m/s,rad,rad/s,kg` and action `[throttle,gimbal_angle]` in `1,rad`. An accepted trajectory with `T+1` states has `T` interval actions. Its first `T` states are supervised inputs and its terminal state is validation-only. Physical values are stored as float64 and may be cast to float32 only after schema validation.

Variable-length trajectories use one `packed_npz_v1` shard per split with separate state and action offsets and `allow_pickle=False`. Each record includes stable trajectory and initial-condition IDs, source index, duration, time, states, actions, both solver-stage iteration counts, hard and terminal residuals, replay checks, normalized replay error ratios, fuel use, and attempt count. Failed attempts remain explicit in a manifest and are excluded from shards. NaN, infinite values, malformed offsets, nonincreasing time, state-action count mismatch, failed replay, and duplicate initial-condition IDs are rejection conditions.

The split unit is a complete trajectory. The initial-condition ID is SHA-256 over canonical little-endian float64 bytes of the seven-state vector and is calculated without a split label. Generation fails on any duplicate within one split or overlap across train, validation, and test. Distinct deterministic seeds are used for all splits. Train and validation share the easy envelope so validation measures IID model selection; validation still never contributes gradients or normalization statistics.

| Field | Train and validation | Hard OOD test |
|---|---:|---:|
| Absolute horizontal offset | 2-10 m | 12-20 m |
| Altitude | 70-105 m | 110-140 m |
| Horizontal velocity | -1.5 to 1.5 m/s | -4 to 4 m/s |
| Vertical velocity | -22 to -15 m/s | -32 to -24 m/s |
| Attitude | -3 to 3 deg | -10 to 10 deg |
| Angular rate | -1.5 to 1.5 deg/s | -5 to 5 deg/s |
| Mass | 970-1000 kg | 900-960 kg |

The train, validation, and test plans contain 800, 100, and 200 trajectories with seeds `20260922`, `20261022`, and `20261122`. The test range is strictly separated from train in absolute horizontal offset, altitude, descent speed, and mass. Therefore every planned hard-test case starts farther away, higher, descending faster, and with less available propellant than every train case. Its other dynamic envelopes are also wider. Configuration validation rejects a plan that loses any of these relations.

Normalization is a standard score fitted only to aligned nonterminal state-action rows from accepted train trajectories. Each stored scale is `max(population_standard_deviation, 1e-6)`. Count, mean, population standard deviation, applied scale, minimum, and maximum are accumulated and stored as float64. Angles remain in radians. At inference, the state is normalized, the predicted action is denormalized, and physical actuator clipping is applied. Final numeric statistics do not exist until the train trajectories are generated on Days 23 and 24.

~~~bash
python scripts/design_offline_dataset.py
~~~

The command writes a JSON contract report and compressed NPZ containing the six deterministic initial-state and ID arrays. It refuses to overwrite either output. The Day 22 completion gate checks the full state, action, trajectory, initial-condition, and solver-quality schema; all split ranges; absence of trajectory leakage; train-only normalization; and a strictly harder test range. Generated files remain under the Git-ignored `artifacts/day22-dataset-design/` directory. The human-readable [dataset card](dataset-card.md) and its [Korean version](dataset-card.ko.md) document intended use and limitations.

## 29. Day 23 large-scale easy Teacher generation

Day 23 solves the complete 800-condition easy train plan from Day 22. The existing timeout-isolated sequential worker now exposes a reusable caller-provided batch interface, while the frozen Day 20 entry point retains its original sampler and policy. The Day 23 policy permits 30 seconds per attempt, recycles the worker after 25 attempts, uses the previous accepted control sequence as the next warm start, and retries once from a new PID trajectory. Every control warm start is reintegrated from the current initial state before optimization.

The generator reports elapsed wall time, average throughput, and whole-run ETA after the first case and every 10 cases thereafter. Timing samples remain descriptive measurements from the development machine. They are stored in the manifest and are not performance guarantees. The output writer first creates temporary files and atomically replaces both final destinations. It refuses to run if either destination already exists.

Solver success does not directly imply admission to the training shard. Each candidate is filtered for duplicate split-independent initial-condition and trajectory IDs; NaN or infinite state, action, time, duration, and quality values; fixed-mesh shape and state-action count errors; nonincreasing or inconsistent time grids; initial-state mismatch; actuator bounds; Stage B hard and terminal residuals; and independently recomputed terminal, altitude, propellant, tilt, and angular-rate limits. The packed shard is then validated again for its exact field set and dtype, offsets, finite values, unique IDs, `T+1` state to `T` action relation, per-trajectory time grids, reconstructed initial-condition IDs, and accepted-only solver/replay flags.

The shard uses the Day 22 `packed_npz_v1` contract. It contains 797 trajectory and initial-condition IDs, source indices, 798 state and action offsets, 80,497 state/time rows, 79,700 action rows, duration and solver-iteration values, hard and terminal residuals, replay error ratios, fuel use, and attempt counts. A canonical array digest includes each sorted field name, little-endian dtype, shape, and raw array bytes, making it independent of NPZ ZIP timestamps.

~~~bash
python scripts/generate_offline_dataset.py
~~~

All 800 planned cases were executed. The run accepted 797 trajectories and rejected three after both the previous-success and PID attempts exceeded the replay tilt bound. Case 637 failed its first optimization attempt but passed its PID retry, so four cases were retried in total. There were no timeouts and no post-solver filter rejections. Wall time was `947.487 s`, or `1.184 s` per executed case, and acceptance was `99.625%`.

Accepted fuel use averaged `26.594 kg` with a `24.785-28.704 kg` range. Duration averaged `5.003 s` with a `5.000-5.230 s` range. The largest stored Stage B hard residual was `6.758e-9`, the largest terminal residual was `1.770e-6`, and the largest fine-replay state error was `0.002351` of its state-specific tolerance. The shard canonical SHA-256 is `04c0f2c0ff3c2e77a28276b37cd8005d2274dad041fbca2e145cd0aec9fe0876`.

The completion gate requires the entire planned split to be executed, at least 500 trajectories to pass all filters, and the final packed shard to pass every structural and quality check. All checks passed. `train-manifest.json` and `train-trajectories.npz` remain under the Git-ignored `artifacts/day23-easy-dataset/` directory. This is the easy train shard, not the final frozen offline dataset. Day 24 adds wider and harder conditions, visualizes coverage, and fills underrepresented regions before finalization.

## 30. Day 24 final offline dataset

Day 24 produces `planar-bc-v1-final` without modifying the verified Day 23 source shard. Before generation, the finalizer loads the Day 23 manifest and NPZ with `allow_pickle=False`, verifies the dataset-configuration digest, completion gate, packed schema, reconstructed initial-condition identities, and canonical shard SHA-256. Any mismatch stops the run before a new solver process starts.

The challenge-train envelope expands the easy range to absolute horizontal offsets of `8-11 m`, altitude `90-108 m`, horizontal speed `-3-3 m/s`, vertical speed `-23.5` to `-20 m/s`, attitude `-7-7 deg`, angular rate `-3-3 deg/s`, and mass `962-980 kg`. The hard OOD test remains strictly separated in horizontal distance, altitude, descent speed, and mass, and retains wider horizontal-speed, attitude, and angular-rate envelopes.

Hard-case oversampling first draws 640 deterministic challenge candidates with seed `20260924`. Each candidate receives the mean of seven normalized difficulty components: absolute horizontal offset, altitude, absolute horizontal speed, downward speed, absolute attitude, absolute angular rate, and inverse mass. The selected set contains the 120 highest scores and 40 random candidates from the remainder. The mean score rises from `0.5010` in the candidate pool to `0.6079` in the selected set. This preserves some distribution breadth while deliberately increasing hard examples.

Coverage uses the union of the easy and challenge train envelopes. It converts horizontal position to absolute target distance and attitude and angular rate to degrees, then forms eight fixed bins for each of seven state features. Every marginal bin must contain at least 30 accepted train trajectories. After challenge generation, the summed deficit was 51. The supplement sampler interleaves deficient features, fixes one targeted feature inside its deficient bin, samples the remaining features over the final train envelope, rejects duplicate initial-condition IDs, and solves the resulting states with the same Teacher policy. After each round, coverage is recomputed from accepted trajectories only. At most 160 extra cases and three rounds are allowed.

The actual run accepted all 160 challenge trajectories. One coverage round accepted 50 of 51 candidates; the remaining random feature values also filled the bin targeted by the failed case, so no second round was needed. Final minimum bin counts were 105 for absolute horizontal offset, 100 for altitude, 30 for horizontal speed, 91 for vertical speed, 34 for attitude, 31 for angular rate, and 76 for mass. All seven coverage checks passed.

The final train shard combines 797 Day 23 trajectories, 160 challenge trajectories, and 50 coverage trajectories for a total of 1,007 trajectories and 100,700 aligned nonterminal state-action pairs. The IID validation run accepted `100/100`. The hard OOD test accepted `191/200`, or `95.5%`. Each of the nine hard-test failures failed fine simulator replay on both its warm-start and PID attempts. There were no timeouts and no successful solver result was removed by the duplicate, nonfinite, structure, or constraint filter.

All three final shards independently pass exact field, dtype, offset, finite-value, accepted-only, time-grid, and initial-condition identity validation. Stable IDs show zero duplicates within a split and zero overlaps between train, validation, and test. Normalization statistics are fitted from the 100,700 final train pairs only; validation and test values never enter the fit.

~~~bash
python scripts/finalize_offline_dataset.py
~~~

The command refuses to overwrite any output and atomically writes `dataset-manifest.json`, `normalization.json`, `dataset-distribution.png`, and one packed NPZ for each split under `artifacts/day24-final-dataset/`. Canonical shard SHA-256 values are train `4161bc93b6753bdcc84cfb6eec6afe1c45bbf814b34525817a7ec24835691e0d`, validation `e77b561f63f5738ec6e09e24b9dbffe2e4712ff33e7d186d12b4945e2aaa618c`, and test `6b67c7866e202b6719f07c733cd2cd82fdd3e2f1638c5bb33ec10ea3f3f5ba17`. The completion gate passed every source, oversampling, difficulty, coverage, count, shard, split-integrity, and normalization check.

## 31. Day 25 Behavior Cloning training

`scripts/train_behavior_cloning.py` reads `configs/bc-day25.yaml` and the Day 24 final dataset. It first verifies the manifest completion and split-integrity flags, the dataset configuration and Teacher digests, the train-only normalization record, and the train and validation shard schemas, counts, and canonical SHA-256 values. It does not load the hard OOD test shard. For each trajectory, the terminal state is excluded and its preceding `T` states are paired in order with `T` interval actions. The resulting matrices contain 100,700 train pairs and 10,000 IID validation pairs.

The first policy has widths `7-64-64-2`, ReLU hidden layers, and a linear normalized-action output. State and action features use the Day 24 train-fitted standard scores. The objective is the mean of the two normalized squared action errors. Mini-batch Adam uses batch size 1,024, learning rate 0.001, seed `20260925`, at most 80 epochs, and patience 12 with minimum validation improvement `1e-5`. The checkpoint with the smallest IID validation MSE is restored for reporting and export. At inference, the seven-state input is normalized, the two predicted actions are returned to physical units, and throttle and gimbal are clipped to the configured actuator bounds.

Before full training, an independent model must fit 16 train pairs to normalized MSE at most `0.0025` within 2,000 optimizer steps. The actual check reached `0.000707` at step 125. The full run selected epoch 79 of 80. At that checkpoint, train MSE was `0.024989`, IID validation MSE `0.030977`, validation throttle MAE `0.032989`, and gimbal MAE `0.272693 deg`. The validation loss was still improving near the epoch limit, so early stopping did not trigger in this run; a separate test exercises that branch.

~~~bash
python scripts/train_behavior_cloning.py
~~~

The ignored `artifacts/day25-bc/` directory contains a pickle-free NPZ checkpoint and a JSON report with the complete epoch history, physical action errors, overfit gate, settings, data provenance, and timing. The checkpoint embeds normalization and actuator limits. `load_checkpoint` verifies parameter shapes, float32 dtypes, finite values, and the exact expected NPZ fields. This Day 25 result is open-loop only; the following section measures closed-loop landing and hard OOD success.

## 32. Days 26-35 closed-loop policy experiments

`scripts/advance_imitation_learning.py` runs the following stages with deterministic train/validation splits. `--through` accepts 26, 28, 32, 34, or 35. The command reads the hard OOD test shard only after model selection and checkpoint freeze in Day 35.

| Stage | Implementation and completion evidence |
|---|---|
| 26 | Compare 64-ReLU baseline, 96-tanh and 128-ReLU with action/terminal weighting; select minimum IID validation MSE. Result: 0.03098, 0.04185, 0.03692, so baseline retained. |
| 27-28 | Roll out 100 validation initial states with 0.1 s learned control updates and 0.02 s RK4 physics; compare BC 14/100 and PID 100/100, classify failures, quantify normalized nearest-neighbor drift, render MP4, and verify checkpoint metadata. |
| 29-32 | Roll out student policy from **train** initial states, select 48 visited states per iteration, re-solve each with the planar optimal-control Teacher, reject failed solver/replay cases, and retrain with about 5% replayed on-policy rows. Accepted labels: 43, 35, 47. The selected v4 checkpoint reached 5/32 on development validation versus BC 4/32. |
| 33 | Compare raw and rate-limited commands plus sensor-noise and combined wind/thrust-error stress on a separate 32-case validation slice. Train a separate 20,000-row train-only sensor-noise augmentation candidate; it achieved 5/32 clean and 2/32 noisy, and was not used in the frozen final comparison. Limit is 2 throttle/s and 60 deg/s. Peak-rate calculations exclude a final shortened contact/fuel-cutoff step. |
| 34 | Evaluate 25%, 50%, and 75% PID action blends as an optional residual-policy feasibility study. All achieved 0/32 and were rejected. No PPO training was run. |
| 35 | Copy and SHA-256-check selected BC/DAgger checkpoints before reading 191 hard OOD initial states. Compare BC 0/191, DAgger 3/191, PID 191/191; measure inference latency and save failure/improvement MP4 replays. |

The DAgger update uses the original 100,700 train pairs and repeats its accepted on-policy labels to about 5% of each training epoch; it does **not** modify the frozen Day 24 dataset. Its OCP Teacher uses the nominal model even when the student visited states under 0.8 m/s wind or 5% thrust loss. That model mismatch is explicit and can affect label quality. Neither the 32-case development slice nor the 100-case validation set is a hard OOD test. The Day 32 nominal-nondegradation target failed: the full validation success rate was BC 14/100 versus stabilized DAgger 11/100. The Day 35 OOD gain of three landings is insufficient for deployment. See the [model card](model-card.md) for failure analysis and use restrictions.

## 33. Days 36-38 residual landing environment and command filter

`configs/rl-evaluation-v1.yaml` fixes six condition groups and a root seed of `378`. Training, development, and final holdout use separate stream IDs `0`, `1`, and `2`. The final-holdout sampler is locked by default. The Day 36 PID pilot uses only development cases and does not filter initial states by Teacher success. The 180-row result and full condition/range/selection record are generated by `scripts/prepare_rl_protocol.py` under ignored `artifacts/day36-rl-protocol/`.

`ResidualLandingEnv` uses the same physical/sensor/actuator stepper as that PID pilot. One normalized residual action in `[-1,1]^2` is submitted every `0.1 s`, and the selected command is sent to the actuator path for five `0.02 s` physics steps; action delay and engine lag can change the command actually applied. Before filtering, the candidate is `PID + [0.08 * residual_throttle, radians(2) * residual_gimbal]`; the second component is in radians. Its 17 float32 observation values are, in order: seven delayed/noisy physical state values divided by `[30 m,160 m,10 m/s,40 m/s,20 deg,30 deg/s,250 kg]` (altitude is relative to ground and mass is relative to dry mass); two PID action values scaled by `[1, gimbal_limit]`; approach and terminal PID integrals divided by their limit; both previous vertical errors divided by `40 m/s`; terminal-phase flag; two pending command values scaled by `[1, gimbal_limit]`; and applied throttle. True state appears in `info` for diagnostics, not in the policy observation.

The command filter returns the current PID command for malformed, non-finite, or out-of-range residuals, actuator saturation, command slew beyond `2 throttle/s` or `60 deg/s`, candidate fuel consumption that breaches a `1 kg` reserve while exceeding PID throttle, or projected attitude/ground margins that are already stressed and worsen relative to PID. The projection checks use the nominal vehicle model, a `0.1 s` horizon, `20 deg` attitude limit, and `0.5 m` ground stopping buffer. They are heuristics, especially under model error and delayed/noisy sensing; they do not prove physical safety or even preserve the PID's earlier trajectory after a previously accepted residual. The base environment retains its sparse simulator reward; the Day 39 wrapper supplies the training reward separately.

Each control update records the baseline action, proposed residual, selected action, fallback reasons, cumulative interventions, and the physical outcome. Wall-clock filter time is kept in `env.safety_timing_us`, outside deterministic Gymnasium `info`. `scripts/verify_residual_landing.py` probes three training cases per group with independent uniform residual components in `[-0.25,0.25]`, writes a non-overwriting JSON report, and does not claim trained-policy performance. Zero residual matched every PID physics state and physical command in all six groups. The refactored 180-case PID CSV retained its previous canonical row digest `f4c2dc64c5bdcd4086bfacce14d82e98b5602552e92baae15048c49fa46a3b52`.

## 34. Days 39-40 reward, curriculum, and SAC pilot

`LandingRewardWrapper` changes only the training reward, not the physical transition or outcome. At each control step, reward is `potential(next) - potential(previous) - 0.1 * elapsed_seconds + terminal_bonus`. Potential lies in `[-11, 0]` and sums clipped penalties for horizontal error, altitude above ground, horizontal and vertical speed, attitude, and angular rate. It has no fuel term. Terminal bonuses are `+100` for success, `-120` for hard landing, `-150` for crash, and `-130` for fuel depletion or timeout. Potential differences telescope over an episode, and the simulator's finite horizon bounds the time term. Hovering incurs time cost, and every failure remains below any successful terminal return. The reward does not make a safety guarantee or imply that the learned policy will optimize the intended objective in a short run.

`CurriculumResidualEnv` samples **train-split IDs only** using a seed-378 generator. Its named stages are nominal, initial OOD, wind, single disturbance, and combined disturbance. Each stage has fixed weights over train-group/difficulty pairs; a stage change takes effect at the next reset and the active episode retains its original stage label. The generator state and episode index are checkpointed. The development cases are sampled from a separate protocol stream, and the final holdout is never materialized by the pilot.

`configs/sac-pilot-v1.yaml` fixes three 2,000-step stages (nominal, initial OOD, wind), total budget 6,000, CPU SAC, a `64x64` network, replay capacity 20,000, 200 initial collection steps, batch 64, learning rate `3e-4`, discount `0.99`, target update `0.005`, and one gradient update per environment step. The actor mean layer is initialized to zero, making the initial deterministic residual exactly zero. The initial 36 development cases reproduce PID outcomes. The run saves model/replay checkpoints with SHA-256 manifests, per-episode training records, paired per-case development evaluations, and PNG learning curves at steps 0, 2,000, 4,000, and 6,000. `--until N` and `--resume` support budget-bounded continuation; resumes restore the model, replay buffer, and curriculum sampler. This is functional checkpoint continuation, not a claim of bitwise-identical training across interrupted runs.

| Environment steps | Nominal | Initial OOD | Wind | New failures on PID successes, in group order |
|---:|---:|---:|---:|---|
| PID / 0 | 12/12 | 12/12 | 6/12 | 0, 0, 0 |
| 2,000 | 12/12 | 12/12 | 7/12 | 0, 0, 0 |
| 4,000 | 11/12 | 10/12 | 7/12 | 1, 2, 3 |
| 6,000 | 9/12 | 10/12 | 6/12 | 3, 2, 2 |

All policy failures in this pilot were classified as crashes. The 2,000-step checkpoint rescued one wind case without losing a PID success among these 36 cases, but this small development result is exploratory. The 6,000-step checkpoint regressed and is not selected over PID. No final-holdout claim, independent-seed robustness claim, or six-group completion claim follows from this pilot.

## 35. Days 41-42 multi-seed robustness and controlled filter experiment

`scripts/advance_residual_sac.py` uses the protocol's root seed `378` and `numpy.SeedSequence(378).spawn(3)` to derive independent training seeds `3024638723`, `2326541261`, and `1907174973`. Day 41 initializes a separate zero-residual actor for each seed, rather than copying the Day 40 pilot. The fixed CPU `64x64` SAC uses learning rate `1e-4`, replay capacity 20,000, 200 initial collection steps, batch 64, discount `0.99`, and one update per environment step. Its 8,000-step curriculum boundaries are nominal 1,000, initial OOD 3,000, wind 5,000, and model mismatch 8,000. The training sampler uses only train-split IDs.

Every checkpoint is evaluated on the same 12 development cases in each of the six condition groups. The evaluation records each case's PID and policy outcomes, paired rescues and new failures, crashes, fuel use as a descriptive field, total control updates, fallback rate and reasons, and which throttle/gimbal bound was exceeded. Fuel is not a model-selection criterion. Each seed's model/replay file, SHA-256 manifest, sampler state, episode records, and learning curve are saved separately. `--resume` continues from the latest complete checkpoint; the final holdout remains sealed. Repeating the same 12 cases under three training seeds does not create 36 independent test cases.

At step 8,000 the safe-landing counts were, in group order nominal / initial OOD / wind / model mismatch / sensor-delay / combined:

| Policy | Nominal | Initial OOD | Wind | Model mismatch | Sensor/delay | Combined |
|---|---:|---:|---:|---:|---:|---:|
| PID | 12 | 12 | 6 | 8 | 6 | 0 |
| Seed 0 | 12 | 12 | 7 | 11 | 8 | 0 |
| Seed 1 | 12 | 12 | 8 | 8 | 7 | 0 |
| Seed 2 | 12 | 12 | 9 | 6 | 8 | 0 |

All cells have denominator 12. Seed 2's model-mismatch result includes two rescued PID failures **and four new failures** on PID successes, so its net count is lower. Combined-condition policies landed 0/12 for all seeds; simulator `crash` outcomes were 12, 12, and 11 respectively. The `severe_crashes` report field counts this outcome label, not measured structural damage. In sensor/delay rollouts, fallback occurred on 10.1%, 16.1%, and 14.9% of control updates. Re-evaluating the frozen checkpoints with axis diagnostics found throttle-slew rejection counts of 121, 132, and 176 versus gimbal-slew counts of 10, 18, and 34. This motivated a single-parameter Day 42 comparison, not a claim that the filter caused every crash.

Day 42 starts each branch from the **same** seed-matched Day 41 step-8,000 model and replay buffer. Both branches train 2,000 steps in a sensor/delay-focused mixture and 2,000 in a broader mixture including combined disturbances, using the same development cases. The control retains the Day 41 filter; `reduced_throttle` changes only the normalized throttle residual scale from `0.08` to `0.04`. Configuration validation checks that the control matches the source filter and that exactly one parameter differs. Results at step 12,000 were:

| Branch / seed | Nominal | Initial OOD | Wind | Model mismatch | Sensor/delay | Combined |
|---|---:|---:|---:|---:|---:|---:|
| Control 0 | 9 | 11 | 8 | 9 | 5 | 1 |
| Control 1 | 9 | 11 | 5 | 6 | 6 | 1 |
| Control 2 | 10 | 12 | 7 | 8 | 6 | 1 |
| Reduced throttle 0 | 11 | 7 | 5 | 6 | 9 | 0 |
| Reduced throttle 1 | 9 | 7 | 5 | 3 | 3 | 0 |
| Reduced throttle 2 | 12 | 11 | 6 | 9 | 8 | 1 |

Every cell again has denominator 12. One combined landing in some branches came with new failures in nominal or other previously successful conditions. Neither continuation meets the protocol's per-group non-regression rule; the one-parameter reduction is rejected, and the PID remains the recommended baseline. These 12-case development observations are diagnostic, not a final robustness estimate or hardware safety claim. No final-holdout case was sampled.

~~~bash
python -m scripts.advance_residual_sac
python -m scripts.advance_residual_sac --settings configs/sac-day42-v1.yaml --output-dir artifacts/day42-sac
~~~
