# 모델과 환경 명세

[English specification](spec.md)

이 문서는 시뮬레이터, 제어기, 데이터셋, 학습 정책이 공유하는 상태·제어·좌표계·단위 규격을 정의합니다. 부호와 단위를 변경할 때에는 코드와 이 명세를 함께 수정합니다.

## 1. 모델 범위

- 2D planar rigid body
- 병진 자유도: `x`, `z`
- 회전 자유도: `theta`
- 가변 질량: `mass`
- 제어 입력: throttle과 추력 벡터 gimbal
- 중력, 추력, 연료 소모, 점 위치 기준 지면 접촉과 선택적 바람·이차 공기저항을 구현
- 평가 전용으로 Gaussian 센서 잡음, 실제 추력 scale 오차와 1차 throttle 지연을 구현
- 관성모멘트와 레버암은 고정

## 2. 관성 좌표계

```text
                       +z (위)
                        ^
                        |       body axis
                        |      /   theta > 0
                        |     /
                        |    /
                  COM   o----------------> +x (오른쪽)
                        |
                        |
                   ground: z = 0
```

- 원점: 착륙 목표 지점의 지면
- `+x`: 화면 오른쪽
- `+z`: 위쪽
- 중력 가속도 벡터: `(0, -g)`
- 지면: `z = ground_z_m`, 기본값은 `0 m`

## 3. 각도와 부호

- `theta = 0`: 로켓의 길이 방향 축이 `+z`와 일치한 수직 자세
- `theta > 0`: 로켓 기수가 `+x` 쪽으로 기울어지는 시계방향 회전
- `omega = d(theta)/dt`: `theta`와 같은 방향이 양수
- `gimbal_angle = 0`: 추력 벡터가 로켓 길이 방향 축과 일치
- `gimbal_angle > 0`: **추력 벡터**가 로켓 축에서 `+theta` 방향으로 회전
- `gimbal_angle`은 노즐 하드웨어의 기계적 편향각이 아니라 힘 벡터의 편향각

추력 벡터의 관성계 방향각은 `theta + gimbal_angle`입니다. 따라서 병진 동역학에서 사용할 추력 성분은 다음 규약을 따릅니다.

```text
thrust_x = thrust * sin(theta + gimbal_angle)
thrust_z = thrust * cos(theta + gimbal_angle)
```

운동방정식은 다음과 같습니다.

```text
x_dot     = vx
z_dot     = vz
vx_dot    = thrust_x / mass
vz_dot    = thrust_z / mass - g
theta_dot = omega
omega_dot = torque / moment_of_inertia
mass_dot  = -thrust / (specific_impulse * standard_gravity)
```

질량이 건조 질량에 도달하면 적용 추력, 토크와 질량 유량을 모두 0으로 설정합니다.

```text
if mass <= dry_mass:
    applied_thrust = 0
    torque = 0
    mass_dot = 0
```

`standard_gravity`는 비추력 `s`를 유효 배기속도 `m/s`로 변환하기 위한 표준중력이며, 시뮬레이션 환경에 적용하는 `gravity`와 구분합니다.

엔진 작용점은 질량중심 아래의 로켓 길이 방향 축 위에 있다고 가정합니다. 엔진 레버암의 길이를 `l`, 추력을 `T`, 짐벌각을 `delta`라고 하면 현재 부호 규약에서 토크는 다음과 같습니다.

```text
torque    = -l * T * sin(delta)
omega_dot = torque / moment_of_inertia
```

따라서 양의 짐벌은 음의 자세 토크와 각가속도를 만들고, 음의 짐벌은 양의 자세 토크와 각가속도를 만듭니다.

## 4. 상태 벡터

상태 배열의 순서는 변경하지 않습니다.

| 인덱스 | 이름 | 의미 | 내부 단위 |
|---:|---|---|---|
| 0 | `x` | 수평 위치 | m |
| 1 | `z` | 수직 위치 | m |
| 2 | `vx` | 수평 속도 | m/s |
| 3 | `vz` | 수직 속도 | m/s |
| 4 | `theta` | 수직축 기준 자세각 | rad |
| 5 | `omega` | 각속도 | rad/s |
| 6 | `mass` | 현재 질량 | kg |

```text
state = [x, z, vx, vz, theta, omega, mass]
```

## 5. 제어 벡터

제어 배열의 순서는 변경하지 않습니다.

| 인덱스 | 이름 | 의미 | 내부 단위/범위 |
|---:|---|---|---|
| 0 | `throttle` | 최대 추력 대비 명령 비율 | dimensionless, `[0, 1]` |
| 1 | `gimbal_angle` | 로켓 축 기준 추력 벡터 편향 | rad |

```text
control = [throttle, gimbal_angle]
```

설정 파일에서 사람이 읽기 쉬운 각도 제한과 초기값은 `_deg` 접미사를 사용합니다. 시뮬레이터 경계에서 한 번만 radian으로 변환하고, 내부 계산·로그·학습 데이터에는 radian을 사용합니다.

제어기가 생성하는 원시 명령은 물리 모델에 들어가기 전에 다음과 같이 제한합니다.

```text
applied_throttle = clip(commanded_throttle, throttle_min, throttle_max)
applied_gimbal   = clip(commanded_gimbal, -gimbal_limit, +gimbal_limit)
```

## 6. 기본 제한과 불변조건

- `mass >= dry_mass`
- `0 <= throttle <= 1`
- `abs(gimbal_angle) <= gimbal_limit`
- 모든 상태·제어 값은 유한한 실수여야 함
- 설정의 `dt_s`, `max_time_s`, `gravity_m_s2`, 질량, 추력은 양수여야 함
- 초기 질량은 건조 질량보다 크거나 같아야 함

## 7. 착륙 성공 기준

지면 접촉 직전의 상태가 아래 조건을 모두 만족할 때 성공으로 판정합니다.

| 기준 | 초기 한계값 |
|---|---:|
| 수평 위치 오차 | `abs(x) <= 1.0 m` |
| 수평 속도 | `abs(vx) <= 1.0 m/s` |
| 수직 속도 | `abs(vz) <= 2.0 m/s` |
| 자세각 | `abs(theta) <= 5 deg` |
| 각속도 | `abs(omega) <= 5 deg/s` |
| 질량 | `mass >= dry_mass` |

이 값들은 초기 권장값이며 `configs/default.yaml`에서 관리합니다. 이후 평가 시 모든 제어기에 동일한 seed와 초기조건을 적용합니다.


## API

`RocketLandingEnv(config)`에 `load_config`로 읽은 dict를 전달합니다. 생성자는 설정을 복사하고 초기 상태, sampling 범위, 단위, 적분기, 착륙 한계값을 검증합니다. 내부 동역학은 `simulate_planar`를 사용합니다.

- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `episode_log`: JSON 직렬화 가능한 전체 기록의 복사본
- `save_episode(path)`: 기록 저장, 기존 파일 덮어쓰기 거부; 부모 폴더는 호출자가 준비

reset 이전 또는 종료 이후의 step은 오류입니다. reset은 새 에피소드를 시작하므로 보관하려는 로그는 그 전에 저장해야 합니다. NaN/Inf, 잘못된 행동 형상, 지면 아래 초기조건, 건조 질량 미만 초기조건은 거부합니다.

관측은 float64 7상태, 행동은 float64 `[throttle, gimbal_angle]`이고 내부 단위는 SI/radian입니다. position/velocity에 임의의 유한 상한을 두지 않아 Gymnasium checker가 관측 공간의 infinity에 관한 권고 경고를 표시할 수 있습니다. 정상 범위에서의 NaN/Inf 관측은 허용하지 않습니다.

## 초기조건

기본 reset은 `initial_state_sampling`의 각 범위에서 독립 균등분포로 값을 추출합니다. YAML 단위는 `initial_state`와 같으며 theta와 omega만 degree에서 radian으로 변환합니다. 범위가 생략된 필드는 기본 초기값을 사용합니다. sampling 섹션 전체가 없으면 기본 초기값을 사용합니다.

- `options={"randomize": False}`: 명목 초기조건
- `options={"initial_state": [...]}`: 정확한 SI/radian 7상태; randomize보다 우선
- `reset(seed=n)`: 난수 발생기를 n으로 초기화
- `reset(seed=None)`: 난수 발생기의 기존 상태를 이어 사용

같은 seed와 동일 행동열은 같은 기록을 생성합니다. action_space의 sample을 사용하는 호출자는 action_space.seed도 별도로 지정합니다. 기록된 초기 상태와 행동열을 사용하면 seed 없이도 같은 환경 설정에서 재계산할 수 있습니다.

## 종료 판정

지면은 현재 모델의 점 위치 z 기준입니다. 첫 접촉의 상태를 판정에 사용하고 지면 이하 관측을 반환하지 않습니다. theta는 판정 시에만 `atan2(sin(theta), cos(theta))`로 한 바퀴 주기를 반영합니다. 내부 상태와 기록은 unwrapped angle을 유지합니다.

| outcome | 판정 |
|---|---|
| running | 진행 중 |
| success | 접촉 위치, vx, vz, theta, omega의 절댓값이 YAML 성공 한계값 이하 |
| hard_landing | 위치, theta, omega는 한계 이내이나 vx 또는 vz 초과 |
| crash | 접촉 시 위치, theta 또는 omega 한계 초과 |
| fuel_depletion | 공중에서 건조 질량 도달 |
| timeout | 최대 시뮬레이션 시간 도달 |

시작 상태가 지면 또는 건조 질량이면 첫 step에서 t=0 종료 판정합니다. 접촉 > 연료 고갈 > 시간초과 순서로 동시 사건을 판정합니다. 순차적으로 발생하는 사건은 먼저 발생한 시각에서 멈춥니다. 연료 고갈 시 환경은 즉시 종료하므로 이후 탄도 착륙 가능성을 성공으로 간주하지 않습니다. 물리 함수의 연료 소진 후 탄도 동작은 그대로 유지됩니다.

성공 및 세 물리적 실패는 terminated=True이고 시간초과만 truncated=True입니다. 성공 보상은 +1, 물리적 실패는 -1, 진행 중과 시간초과는 0입니다. 미래 학습의 최종 reward는 아직 설계하지 않았습니다.

## 스텝 내 사건 계산

한 action은 simulation.dt_s 동안 일정하게 유지됩니다. 내부 적분 간격은 최대 0.02초이며 더 작은 설정 dt는 그대로 사용합니다. 마지막 스텝은 최대 시간에 맞춰 단축합니다. 질량 유량으로 계산한 연료 소진 시점에서도 내부 구간을 단축합니다.

각 내부 구간에서 지면 교차를 검사하고 Brent root search로 접촉 시각을 찾습니다. 양 끝 고도가 모두 양수여도 구간 중간에 하강 후 상승할 수 있으므로 가속도 상한으로 가능한 구간을 선별하고 고도의 최소점을 검색합니다. root search 허용시간 오차는 1e-12초이나 최종 물리 정확도는 선택한 Euler/RK4와 적분 간격에 의해 제한됩니다.

하강 중인 구간 끝점의 고도가 지면보다 부동소수점 정밀도 수준으로 높게 남는 경우에는 접촉으로 취급합니다. 고도 허용오차는 `64 * eps * max(1, |ground_z|, |z_start|, |z_end|)` m이며, 정확히 시간제한과 겹치는 접촉이 반올림 오차로 timeout으로 바뀌는 것을 방지합니다.

## 에피소드 JSON schema_version=1

| 필드 | 내용 |
|---|---|
| config | 실행에 사용한 설정 전체 |
| seed_argument | 해당 reset에 전달한 seed, 생략 시 null |
| rng_state_after_reset | 초기조건 생성 직후 난수 상태 |
| state_names / state_units | 상태 순서 및 단위 |
| action_names / action_units | 행동 순서 및 단위 |
| initial_state | t=0의 SI 7상태 |
| steps | 각 step의 결과 목록 |
| outcome | 최신 종료 사유, 미종료 시 running |

각 steps 원소에는 index, time_s, state, commanded_action, clipped_action, thrust_start_n, thrust_end_n, fuel_used_kg, reward, terminated, truncated, outcome, is_success가 들어갑니다. fuel_used_kg는 초기 질량과 현재 질량의 차이입니다. clipped_action은 액추에이터 제한 명령이며 실제 연료 차단 여부는 thrust_start_n/end_n으로 구분합니다. 상태·반환 info·로그를 외부에서 변경해도 환경 내부 기록이 바뀌지 않도록 복사합니다.

API 참고: [Gymnasium Env](https://gymnasium.farama.org/api/env/), [환경 검사기](https://gymnasium.farama.org/api/utils/).

## 기록 재생과 시각화

`save_episode_data`는 같은 에피소드 기록을 JSON 또는 압축 NPZ로 저장합니다. NPZ에는 원본 JSON과 검증용 `times_s`, `states`, `actions`, `thrust_n` 배열을 함께 저장하며 객체 pickle은 사용하지 않습니다. `actions`는 각 step에 적용한 명령이고 나머지 배열은 초기 프레임을 포함합니다. `load_episode_data`는 배열과 원본 기록의 일치 여부를 확인합니다. 두 형식 모두 저장된 상태를 직접 읽으므로 물리 모델을 다시 실행하지 않습니다.

시계열 그래프는 위치, 속도, 자세·각속도, 질량, throttle, gimbal을 표시합니다. 2D 애니메이션은 x-z 궤적, 로켓 자세, 추력 벡터, 지면과 x=0 착륙 목표를 표시합니다. 로켓 도형은 부호 확인용이며 실제 크기나 착륙 다리 형상을 나타내지 않습니다. GIF는 기본 지원하고 MP4는 시스템 FFmpeg 또는 `video` 추가 의존성에 포함된 실행 파일을 사용합니다. 출력 파일은 기존 파일을 덮어쓰지 않습니다.

## 수직 Suicide burn 기준선

SuicideBurnController는 수평 위치와 자세 오차가 없는 수직 하강만 다룹니다. 첫 번째 점화 고도 계산은 연소 중 질량을 현재 질량으로 고정합니다.

~~~text
net_deceleration = max_thrust / mass - gravity
stopping_distance = (downward_speed^2 - target_speed^2) / (2 * net_deceleration)
~~~

estimate_suicide_burn은 이 상수 질량 계산을 그대로 제공합니다. 실제 SuicideBurnController는 연료 소모에 따라 질량과 가속도가 변하는 수직 운동 해석해로 점화 고도를 보정합니다. 필요한 속도 감소에 비해 연료가 부족하거나 최대 추력이 중력을 이기지 못하면 infeasible로 판정하고 즉시 최대 추력을 명령합니다.

이상적인 점화가 0.02초 제어 주기 안에 있으면 첫 연소 스텝의 스로틀 비율을 조정해 이산 시간 오차를 줄입니다. 점화 이후에는 최대 추력을 유지합니다. 실제 점화 고도가 계산값보다 높으면 early_burn, 낮으면 late_burn, 허용오차 안이면 on_time으로 분류합니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller suicide-burn
~~~

이 기준선은 바람, 수평 오차, 기체 기울기와 센서 또는 엔진 지연을 보정하지 않습니다. 접지 속도 2 m/s는 현재 환경의 성공 판정 한계이며 실제 구조 안전 속도가 아닙니다.

## 수직속도 PID 기준선

VerticalVelocityPIDController는 고도에 따른 목표 수직속도를 만들고 throttle PID로 추종합니다. 수평 위치, 수평 속도, 자세와 각속도가 0인 수직 하강 조건을 대상으로 합니다.

~~~text
speed_limit = sqrt(touchdown_speed^2 + 2 * profile_deceleration * altitude)
target_vz = -min(max_descent_speed, speed_limit)
error = target_vz - current_vz
~~~

기본 profile은 접지 목표속도 1 m/s, 최대 하강속도 30 m/s, 감속도 3 m/s²입니다. Profile의 감속도와 중력을 보상하는 feed-forward throttle에 PID 보정값을 더합니다.

~~~text
feed_forward = mass * (gravity + target_acceleration) / max_thrust
raw_throttle = feed_forward + Kp * error + Ki * integral + Kd * error_rate
throttle = clip(raw_throttle, throttle_min, throttle_max)
~~~

기본 이득은 Kp=0.08, Ki=0.001, Kd=0.01입니다. 적분 오차는 ±10 m로 제한합니다. 출력이 상한에서 포화되고 오차가 양수이거나, 하한에서 포화되고 오차가 음수이면 해당 스텝의 적분을 취소합니다. reset은 적분값, 이전 오차와 마지막 명령 정보를 모두 초기화합니다.

평가 초기조건은 고도 80-120 m, 수직속도 -25~-15 m/s, 질량 950-1000 kg의 독립 균등분포입니다. 20260910 seed로 생성한 100개 조건에서 100회 모두 성공했으며 평균 접지 속도는 1.1010 m/s, 95백분위는 1.1938 m/s, 평균 연료 사용량은 37.4923 kg입니다.

~~~powershell
python scripts/run_episode.py --initial-state 0 100 0 -20 0 0 1000 --controller velocity-pid
python scripts/sweep_velocity_gains.py --episodes 100 --seed 20260910
~~~

이 결과는 현재 설정 범위의 결정론적 수직·무풍 시뮬레이션에 한정됩니다. 수평 위치와 자세 제어는 현재 범위 밖이며 바람, 모델 오차, 센서 잡음과 구동기 지연은 이후 강건성 단계에서 평가합니다.

## 수평 위치·자세 제어기

`HorizontalAttitudeController`는 학습을 사용하지 않는 직렬 제어기입니다. 현재 상태와 별도 수직 제어기가 계산한 기본 throttle을 입력받습니다. 이 인터페이스로 수평·자세 로직을 통합 제어기 단계 전까지 독립적으로 유지합니다.

외부 루프는 위치와 속도 오차로 수평 가속도를 계산합니다.

~~~text
raw_ax = Kp_x * (target_x - x) - Kd_x * vx
target_ax = clip(raw_ax, -max_ax, max_ax)
target_theta = clip(atan2(target_ax, gravity), -max_tilt, max_tilt)
~~~

기본 외부 루프 이득은 `Kp_x=0.05 s^-2`, `Kd_x=0.45 s^-1`입니다. 수평 가속도는 2.5 m/s², 목표 기울기는 15도로 제한합니다.

내부 루프는 목표 각가속도를 계산하고 문서의 토크 부호 규약을 역으로 풉니다. 양의 gimbal은 음의 각가속도를 만듭니다.

~~~text
target_alpha = Kp_theta * (target_theta - theta) - Kd_theta * omega
sin(gimbal) = -target_alpha * inertia / (lever_arm * thrust)
~~~

기본 내부 루프 이득은 `Kp_theta=4.0 s^-2`, `Kd_theta=3.0 s^-1`입니다. 역계산 값은 삼각함수 정의역과 기체의 15도 gimbal 제한에 맞춰 자릅니다. 기본 throttle이 0이면 추력 편향으로 토크를 만들 수 없으므로 gimbal은 0입니다.

기울어진 추력의 수직 성분은 `cos(theta + gimbal)`만큼 줄어듭니다. Coupling 보상을 켜면 출력 throttle의 수직 성분이 요청된 기본 throttle과 같아지도록 반복 계산하며, 최종 출력에는 0-1 구동기 제한을 적용합니다.

재현 실험은 고도 100 m의 무풍 조건에서 100개 초기조건을 각각 8초 동안 실행합니다. 초기 수평 오차는 -20~20 m 범위에서 최소 5 m이며 수평 속도, 자세, 각속도와 질량은 설정 범위에서 표본화합니다. Seed 20260914에서 100개 모두 정상 완료했고 절대 수평 오차가 감소했습니다. 평균 오차는 12.4712 m에서 5.4894 m로 줄었고 최대 목표 기울기는 9.7647도, 최대 gimbal은 7.0393도, 최대 고도 편차는 0.0149 m였습니다.

~~~powershell
python scripts/evaluate_horizontal_control.py --episodes 100 --seed 20260914
~~~

이 분리 결과는 무풍에서 착륙장 방향으로 수렴함을 검증하지만 자체적으로 접지 성공을 의미하지는 않습니다. 아래 통합 제어기는 두 명령을 결합하고 전체 착륙을 평가합니다.

## 통합 착륙 제어기

`IntegratedLandingController`는 수직속도 PID throttle과 수평 위치·자세 gimbal 제어를 결합합니다. 시뮬레이터는 0.02초 간격으로 진행하고 제어기는 0.1초마다 갱신합니다. 중간의 시뮬레이션 4개 step에서는 가장 최근 명령을 유지합니다. 제어 주기가 시뮬레이션 주기보다 짧거나 정수배가 아니면 설정을 거부합니다.

고도에 따라 독립적인 상태를 가진 두 제어 phase 중 하나를 선택합니다. 30 m보다 높으면 접근 phase가 접지 목표속도 1.0 m/s, profile 감속도 1.5 m/s², 수직 PID gain `(0.08, 0.001, 0.01)`, 수평 gain `(0.5, 1.7)`, 목표 기울기 제한 15도를 사용합니다. 30 m 이하에서는 말기 phase가 접지 목표속도 0.8 m/s, profile 감속도 1.0 m/s², 수직 PID gain `(0.1, 0.001, 0.015)`, 수평 gain `(0.3, 1.35)`, 목표 기울기 제한 5도를 사용합니다. 말기 제한은 접지 자세와 각속도를 낮추는 데 우선순위를 둡니다.

수직 제어기가 기본 throttle을 먼저 계산합니다. 선택된 수평·자세 제어기는 gimbal을 계산하고 기울어진 추력의 수직 성분을 보상합니다. 마지막으로 이전 제어 갱신값을 기준으로 통합 명령의 변화량을 제한합니다.

~~~text
abs(throttle[k] - throttle[k-1]) <= 2.0 * control_interval
abs(gimbal[k] - gimbal[k-1]) <= deg2rad(60) * control_interval
~~~

비교할 이전 출력이 없으므로 첫 계산 명령은 held action의 초기값이 됩니다. 이후 갱신에서는 throttle과 gimbal 변화율 제한 여부를 따로 제공합니다. 시간은 유한한 0 이상의 값이며 감소할 수 없습니다. `reset`은 두 phase, held action, 갱신 횟수와 scheduling 상태를 모두 초기화합니다.

중간 난이도 평가는 무풍 초기조건 100개를 사용합니다. 수평 오차는 착륙장 양쪽에서 3-15 m, 고도 80-120 m, 수평 속도 -2~2 m/s, 수직 속도 -25~-15 m/s, 자세 -5~5도, 각속도 -2~2 deg/s, 질량 950-1000 kg 범위입니다. Seed 20260914에서 100개 모두 모든 접지 기준을 만족했습니다.

평균·최대 절대 접지 위치는 0.4516 m와 0.9440 m, 수평 속도는 0.1476 m/s와 0.3526 m/s, 수직 속도는 0.7819 m/s와 0.7827 m/s였습니다. 평균·최대 자세는 0.7559도와 1.4386도, 각속도는 1.1004 deg/s와 2.4399 deg/s였습니다. 평균 연료 사용량은 53.0159 kg입니다. 관측된 최대 변화율은 throttle 초당 1.7784, gimbal 초당 60.0000도였습니다.

~~~powershell
python scripts/evaluate_integrated_control.py --episodes 100 --seed 20260914
python scripts/run_episode.py --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid
~~~

이 제어기는 결정론적 무풍 기준선입니다. 외란 대응, 불확실성, 센서 잡음, 구동기 지연과 하드웨어 안전은 이번 결과의 범위 밖입니다.

## 기준선 튜닝 자동화

`tune_integrated_controller.py`는 통합 제어기의 두 phase에 함께 적용할 세 배율을 탐색합니다. 탐색 대상은 수평 위치 이득, 수평 속도 이득과 하강 profile 감속도입니다. Cartesian `grid` search와 seed를 고정한 균등분포 `random` search를 지원하며, 두 방식 모두 배율을 적용하지 않은 `(1, 1, 1)` 기준 후보를 포함합니다. 후보 배율은 복사된 설정에 적용하므로 입력 설정은 변경되지 않습니다.

모든 후보는 동일한 학습 조건 묶음에서 평가합니다. 다음 단일 목적함수를 최소화해 후보를 선택합니다.

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

실패 항이 두 보조 항보다 우선하며, 연료와 접지 오차를 정규화해 각 척도를 명시적으로 맞춥니다. 목적함수 가중치, 탐색 범위, 후보 수, 에피소드 수와 seed는 모두 설정에서 바꿀 수 있습니다. 잘못된 범위, 0 이하 실행 횟수, 같은 학습·검증 seed, 두 보조 항을 모두 사용하지 않는 목적함수는 설정을 읽을 때 거부합니다.

기본 Cartesian grid는 배율 조합 12개에 기준 후보를 더해 총 13개를 평가합니다. 학습은 seed 20260912에서 표본화한 초기조건 16개만 사용합니다. 후보 선택은 학습 목적함수로만 수행하며, 선택된 후보는 seed 20260913에서 독립적으로 표본화한 초기조건 100개에서 한 번만 검증합니다.

선택된 배율은 `(1.10, 0.95, 1.10)`입니다. 학습 조건 16개는 모두 안전하게 착륙했고 목적함수는 3.716630, 평균 연료 사용량은 51.1532 kg이었습니다. 기준 후보도 16개 모두 성공했지만 목적함수는 4.179125, 평균 연료 사용량은 52.9862 kg이었습니다. 독립 검증에서는 100회 중 100회 안전하게 착륙했고 평균 연료 사용량 51.9394 kg, 평균 절대 접지 위치 0.3942 m, 평균 절대 수직 접지 속도 0.7802 m/s를 기록했습니다.

~~~powershell
python scripts/tune_integrated_controller.py --report artifacts/tuning.json --best-config artifacts/best-integrated.yaml
~~~

JSON 보고서는 모든 후보의 평가 결과, 목적함수 항, seed와 독립 검증 결과를 포함합니다. YAML 출력은 선택 정보가 추가된 전체 설정이며 Git에 포함된 기본 설정과 같은 검증을 통과합니다. 기존 출력 파일은 덮어쓰지 않습니다.

## PID 분리 외란 평가

`evaluate_pid_disturbances.py`는 12일차에 선택한 gain 배율 `(1.10, 0.95, 1.10)`을 적용한 뒤 한 번에 하나의 외란 모델만 바꿉니다. 한 곡선 안의 모든 강도는 seed 20260915에서 생성한 같은 초기조건 30개를 사용합니다. 바람 방향, 돌풍 시작 시각과 센서 잡음열도 강도 단계 사이에서 episode별로 짝을 맞춥니다. 다섯 곡선의 외란 강도 0 결과가 서로 다르면 평가를 중단합니다.

### 바람과 공기저항

바람은 관성좌표계의 2차원 공기 속도 벡터입니다. 기체 속도를 `v`, 공기 속도를 `w`라고 할 때 상대속도와 이차 항력은 다음과 같습니다.

~~~text
v_relative = v - w
F_drag = -0.5 * air_density * drag_coefficient * reference_area
         * norm(v_relative) * v_relative
~~~

힘은 모델의 질량중심에 작용하므로 공력 토크를 만들지 않습니다. 각 RK4 단계에서 다시 계산합니다. 바람 모델을 전달하지 않으면 기존 이상 동역학을 정확히 보존하고 공기력을 적용하지 않습니다.

일정 바람은 각 episode에서 표본화한 양 또는 음의 수평 방향을 비행 내내 유지합니다. 돌풍은 지속시간 2초인 수평 half-sine pulse이며 시작 시각은 2-6초 균등분포, 방향은 episode마다 독립적으로 표본화합니다. 진폭을 비교할 때는 같은 방향과 시작 시각을 다시 사용합니다.

### 센서 잡음

센서 잡음은 제어기가 보는 상태에만 더하며 접촉 판정과 물리 진행에는 실제 상태를 사용합니다. 서로 독립인 영평균 Gaussian 표본의 1배 표준편차는 x와 z `0.5 m`, vx와 vz `0.1 m/s`, 자세와 각속도 `0.25 deg`와 `0.25 deg/s`, 질량 `0.5 kg`입니다. 하나의 무차원 강도 배율을 일곱 값에 모두 곱합니다. 잡음이 포함된 고도와 질량은 제어기의 물리 영역을 벗어나지 않도록 지면과 건조 질량에서 제한합니다.

### 추력 오차와 엔진 지연

추력 배율은 시뮬레이터의 실제 최대 추력만 바꾸고 제어기는 명목 모델을 그대로 사용합니다. 연료 유량은 실제 추력과 기존 비추력을 사용합니다. 따라서 배율 `0.95`는 모든 throttle에서 제어기 예상 추력의 95%만 발생한다는 뜻입니다.

엔진 지연은 각 0.02초 시뮬레이션 간격에서 정확히 적분한 1차 throttle 응답입니다.

~~~text
response = 1 - exp(-dt / time_constant)
actual_throttle += response * (commanded_throttle - actual_throttle)
~~~

실제 throttle 초기값은 0입니다. Gimbal servo 동역학은 이번 외란에 포함하지 않으므로 gimbal은 즉시 반응합니다. 시정수가 0이면 명령을 그대로 재현합니다.

### 성능 곡선과 붕괴 기준

안전 착륙 성공률이 95%보다 낮아지는 첫 외란 강도를 붕괴점으로 기록합니다. 강도별 episode가 30개이므로 한 사례는 성공률 3.33%p에 해당합니다. 측정된 붕괴점 결과는 다음과 같습니다.

| 외란 | 마지막 통과 강도 | 붕괴 강도 | 성공률 | 붕괴점 결과 |
|---|---:|---:|---:|---|
| 일정 바람 | 10 m/s | 20 m/s | 86.7% | 성공 26, crash 4 |
| 돌풍 진폭 | 40 m/s | 60 m/s | 86.7% | 성공 26, crash 4 |
| 센서 잡음 배율 | 1배 | 2배 | 86.7% | 성공 26, crash 4 |
| 실제 추력 배율 | 1.00배 | 0.95배 | 93.3% | 성공 28, crash 2 |
| Throttle 지연 시정수 | 0.2 s | 0.5 s | 90.0% | 성공 27, crash 3 |

공통 무외란 결과는 안전 착륙 29회와 crash 1회였습니다. 별도 12일차 검증 묶음에서는 나타나지 않았던 경계 조건이 새 seed에서 드러난 것입니다. 위 수치는 실험 grid에서 측정한 경계이며 정확한 물리 한계나 하드웨어 인증값이 아닙니다.

~~~powershell
python scripts/evaluate_pid_disturbances.py --report artifacts/disturbances.json --plot artifacts/disturbances.png
~~~

JSON에는 모든 강도의 결과 분포, 최종 상태 오차, 정규화 오차와 연료 사용량이 들어갑니다. PNG에는 안전 착륙 성공률 곡선 다섯 개, 95% 기준과 측정한 붕괴 강도가 표시됩니다. 기존 파일은 덮어쓰지 않습니다.

## 동결된 2주차 기준선 protocol

`configs/pid-baseline-v1.yaml`은 이후 모든 방법이 사용할 비교 기준입니다. 12일차에서 선택한 배율을 통합 제어기 값에 직접 반영했습니다. 접근 phase의 위치 gain, 속도 gain과 profile 감속도는 `0.55`, `1.615`, `1.65 m/s^2`이고, 말기 phase 값은 `0.33`, `1.2825`, `1.1 m/s^2`입니다. 이 제어기 mapping의 동결 digest는 `80d4cbedbc545457072dc08f308ee408fd25996e3df500520135ec0452818a8a`입니다.

고정 평가 집합은 sampler `integrated_uniform_v1`, 1,000 episode와 seed 20260914를 사용합니다. 수평 위치의 부호와 크기를 따로 추출해 모든 초기상태가 목표에서 3 m 이상 떨어지도록 하고, 나머지 여섯 상태 성분은 통합 제어기 평가 설정의 범위에서 서로 독립적으로 추출합니다. 각도는 hash 계산과 simulation 전에 radian으로 변환합니다.

`initial_condition_sha256`은 schema prefix, little-endian unsigned 64-bit 정수로 표현한 행렬 크기 두 개, 연속된 little-endian IEEE 754 float64 상태 행렬을 순서대로 hash합니다. 기대 digest는 `a9593a383acc0017738ed383a123ca743a32bce46c64d249b08ca577a448a952`입니다. `frozen_baseline_initial_states`는 행렬을 다시 생성한 뒤 제어기와 상태 digest를 모두 검증합니다. 반환 행렬은 읽기 전용이며 제어기 수정, seed·범위 변경, sampler 변경 또는 episode 수 변경을 거부합니다.

명목 평가 결과는 다음과 같습니다.

| 평가 항목 | 결과 |
|---|---:|
| 안전 착륙 | 999 / 1,000 (99.9%) |
| Crash | 1 / 1,000 |
| 평균 / 최대 절대 접지 x | 0.4098 / 1.3494 m |
| 평균 / 최대 절대 접지 vx | 0.1656 / 0.5915 m/s |
| 평균 / 95백분위 / 최대 절대 접지 vz | 0.7802 / 0.7810 / 0.7814 m/s |
| 평균 / 최대 절대 접지 자세 | 0.5991 / 3.2498 deg |
| 평균 / 최대 절대 접지 각속도 | 1.3178 / 3.6500 deg/s |
| 평균 연료 사용량 | 51.4460 kg |

성공률 99.9%로 설정된 2주차 통과 기준 70%를 만족했습니다. 한 번의 crash는 수평 위치 제한 1 m를 초과해 발생했으며, 수평 속도·수직 속도·자세·각속도의 최대값은 각 접지 제한 안에 있었습니다. 이 유한 simulation 결과는 모든 상황의 강건성을 보장하지 않습니다.

~~~powershell
python scripts/evaluate_frozen_baseline.py --output-dir artifacts/week2-baseline
~~~

이 명령은 JSON 평가 보고서, 재생 가능한 episode log 두 개와 GIF 두 개를 생성합니다. 대표 성공은 초기조건 0번의 무풍 결과입니다. 대표 실패는 같은 0번 초기조건에 40 m/s 일정 수평 바람을 적용하며, 접지 위치 오차 2.5911 m로 crash가 발생합니다. 이 스트레스 사례는 알려진 실패 양상을 보여주기 위한 것이며 명목 성공률에는 포함하지 않습니다. 실행 전 모든 대상 파일을 확인하고 기존 출력은 덮어쓰지 않습니다.

## 15일차 최적제어 Teacher 문제 정식화

`LandingOptimalControlProblem`에 7상태 Teacher 문제를 정의했습니다. 16일차에는 수직 부분, 17일차에는 이상 추력 방향 병진 부분을 먼저 풀었고 18일차에는 전체 평면 문제를 풉니다. 모델은 무풍, 이상적인 센서와 즉시 반응하는 구동기를 가정합니다. 평면 상태 순서와 단위, 추력 방향, 토크 부호, 가변 질량 식 및 착륙 판정 제한은 현재 시뮬레이터와 같습니다. 명목 실행에서는 바람 모델을 전달하지 않으므로 선택적 공기저항은 적용되지 않습니다. 최적화 식은 action clipping이나 건조질량 연료 차단을 포함하지 않습니다. 대신 구동기 경계와 1 kg 연료 여유를 강제해 가능한 궤적을 매끄러운 영역에 둡니다.

상태는 `X = [x, z, vx, vz, theta, omega, m]`, 제어는 `U = [q, delta]`입니다. `q`는 무차원 throttle, `delta`는 radian gimbal 각도입니다. 제어는 `N = 100`개 구간에서 각 구간 동안 일정합니다. 종료시간 `T`는 `[5, 25] s` 범위의 결정변수이며 mesh 간격은 `h = T/N`입니다. 초기상태는 주어진 값으로 고정합니다. 모든 물리 단위는 SI이고, 각도 양의 방향은 프로젝트 규약에 따라 `+x`로 향하는 시계방향입니다.

| 수학적 항목 | 식 또는 제한 | 코드 입력과 출력 |
|---|---|---|
| 추력 | `F = T_max q` | `dynamics(X, U)`가 throttle `q`를 받고 7성분 미분 벡터를 반환 |
| 위치 | `dx/dt = vx`, `dz/dt = vz` | 미분 벡터 0-1번 성분 |
| 병진 | `dvx/dt = F sin(theta + delta)/m`; `dvz/dt = F cos(theta + delta)/m - g` | 2-3번 성분, 매끄러운 허용 영역에서 기존 `state_derivative`와 일치 |
| 회전 | `dtheta/dt = omega`; `domega/dt = -L F sin(delta)/I` | 4-5번 성분, 음의 토크 부호가 시뮬레이터와 일치 |
| 연료 | `dm/dt = -F/(Isp g0)` | 6번 성분 |
| 다중 사격 결함 | `X[k+1] - RK4(X[k], U[k], T/N) = 0` | `rk4_defects(states, controls, T)`가 `N x 7` 행렬 반환 |
| 경로 여유 | `z-ground >= 0`; `m-dry_mass-1 kg >= 0`; `q_min <= q <= q_max`; `|delta| <= delta_max`; `|theta| <= 20 deg`; `|omega| <= 30 deg/s` | `path_margins(X, U)`가 0 이상이어야 할 열 개 값을 반환하며 상태 제한은 최종 node에도 적용 |
| 종단 착륙 | `z(T)=ground`; `|x(T)-target_x|<=1 m`; `|vx(T)|<=1 m/s`; `-2<=vz(T)<=0 m/s`; `|theta(T)|<=5 deg`; `|omega(T)|<=5 deg/s` | `terminal_violations(X_T)`가 정규화된 양의 위반량 여섯 개를 반환; 모두 0이면 조건 충족 |

종단 고도 등식의 A단계 위반량은 고정된 1 m 척도로 정규화합니다. 종단 수직속도 제약은 위로 올라가며 지면에 닿는 상태를 제외합니다. 시뮬레이터는 점 접촉 모델이며 착륙 다리와 충격 하중을 다루지 않습니다. 경로 제약은 사격 node에서 적용되므로, 최적화 결과는 node 사이 지면 관통이나 전사 오차가 없는지 사건 처리 시뮬레이터에서 다시 rollout해야 합니다.

목적함수는 하나의 큰 벌점으로 합치지 않고 두 단계로 나눕니다. **A단계**에서는 초기상태, 동역학, 시간, 구동기, 고도와 연료 제약을 hard constraint로 유지하고 종단 제약만 임시로 완화합니다. 여섯 종단 위반량의 제곱합을 최소화하며, 최대 정규화 종단 위반량이 `0.001` 이하일 때만 통과합니다. 이렇게 하면 연료 절약이 착륙 실패를 상쇄하지 못합니다. **B단계**에서는 A단계 결과를 초기해로 사용하고 종단 제약까지 hard constraint로 적용한 뒤 아래 비용을 최소화합니다.

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

목표 종단 수직속도 `-0.8 m/s`는 시뮬레이터의 안전 하강 구간 `[-2, 0] m/s` 안에 있습니다. 비용 항은 모두 무차원입니다. B단계 연료 비율은 명목 용량 250 kg이 아니라 해당 초기상태에서 사용할 수 있는 연료를 분모로 사용합니다. 두 제어 변화율 척도는 통합 제어기의 설정된 slew 기준을 사용합니다. Mesh 시간 간격으로 나누므로 같은 물리 제어 ramp는 node 개수가 달라도 같은 평활성 비용을 갖습니다. 이 기준은 soft objective 척도이며 hard 변화율 제한이 아닙니다. Solver 수렴, 제약 잔차 확인과 독립 시뮬레이터 rollout까지 끝나야 Teacher 궤적으로 인정합니다.

## 16일차 수직 CasADi/IPOPT Teacher

16일차는 15일차 문제 중 수직 부분만 풉니다. 상태는 `Y=[z,vz,m]`, 제어는 throttle `q`이며 동역학은 `dz/dt=vz`, `dvz/dt=T_max*q/m-g`, `dm/dt=-T_max*q/(Isp*g0)`입니다. 초기 수평·자세 상태는 0이어야 합니다. 제어 구간 100개, `[5,25] s` 범위의 자유 종료시간, 건조질량 위 1 kg 연료 여유, 구동기 경계, 착륙 수직속도 제한은 평면 문제와 같은 설정을 사용합니다. 각 구간에 CasADi RK4 연속 등식이 있으므로 단일 사격이 아닌 직접 다중 사격법입니다.

초기 추정치는 등가속 하강으로 종료시간과 throttle을 계산한 뒤 수치 적분해 상태 node를 만듭니다. A단계는 초기상태, RK4 동역학, 시간, 고도, 연료, throttle 제한을 hard constraint로 유지합니다. 두 개의 비음수 종단 slack으로 고도·수직속도 위반을 일시 허용하고 정규화 제곱합을 최소화합니다. B단계는 A단계 해를 초기값으로 사용해 종단 고도를 지면에 고정하고 최종 속도를 `[-2,0] m/s`로 제한한 뒤 15일차 목적함수의 수직 부분인 연료 비율·접지 속도 오차·throttle 변화량을 최소화합니다. 설정된 정규화 가능해 허용치 `0.001`을 충족해야 해를 채택합니다.

JSON 보고서는 단계별 IPOPT 종료 상태, 반복 수, 목적값, 정규화된 최대 hard·종단 위반량과 각 상태의 RK4 결함을 물리 단위로 기록합니다. 최적 throttle은 기존 평면 시뮬레이터에서 `0.02 s` 간격으로 별도 적분합니다. 보고서는 지면 고도, 최종 하강속도, 연료 여유, 최소 고도와 모든 최적화 node에서의 불일치를 확인하고, PNG는 재적분 궤적과 최적화 node를 겹쳐 보여줍니다. 솔버 또는 가능해 실패 시 그래프 없이 진단 JSON을 저장합니다. 이 명목 재적분은 바람·센서 오차·엔진 지연 또는 다양한 초기조건에 대한 성능 검증이 아닙니다.

~~~bash
python -m pip install -e ".[optimization]"
python scripts/solve_vertical_landing.py --output-dir artifacts/day16-vertical
python scripts/solve_vertical_landing.py --initial-z 110 --initial-vz -18 --initial-mass 980 --output-dir artifacts/day16-custom
~~~

기존 결과 파일은 덮어쓰지 않습니다. 기본 초기조건에서는 종료시간 `5.000 s`, 시뮬레이터 최종 속도 약 `-0.832 m/s`, 연료 사용량 약 `27.435 kg`으로 수렴했습니다. 한 건의 수치 결과이며 전체 성공률은 아닙니다. 17일차에는 수평 병진을 추가했고, 18일차에는 회전과 짐벌 토크를 추가합니다.

## 17일차 평면 병진 Teacher

17일차는 풀이가 가능한 수직 문제를 `Y=[x,z,vx,vz,m]`, 제어 `U=[q,alpha]`로 확장합니다. `alpha`는 수직축 기준 **절대 추력 벡터 각도**이며 실제 짐벌 각도가 아닙니다. 몸체 자세와 각속도는 의도적으로 18일차까지 제외합니다. 동역학은 `dx/dt=vx`, `dz/dt=vz`, `dvx/dt=T_max*q*sin(alpha)/m`, `dvz/dt=T_max*q*cos(alpha)/m-g`, `dm/dt=-T_max*q/(Isp*g0)`입니다. 한 구간에서 몸체 각도를 `alpha`, 짐벌을 0으로 둔 기존 시뮬레이터의 병진 상태·부호와 일치합니다.

직접 다중 사격은 16일차와 같은 제어 구간 100개와 `[5,25] s` 자유 종료시간을 사용합니다. 초기상태, RK4 연속 등식, `z>=ground`, `m>=dry_mass+1 kg`, `q_min<=q<=q_max`, `|alpha|<=15 deg`는 사격 node의 hard constraint입니다. 추력 방향 15도 제한은 축소 모델의 명시적 가정이며 몸체 회전이나 짐벌이 그 명령을 실현할 수 있다는 증거가 아닙니다. A단계는 종단 위치, 고도, 수평속도와 하강 수직속도 제한만 네 개의 비음수 정규화 slack으로 일시 완화합니다. B단계는 종단 고도를 지면으로 고정하고 `|x-target_x|<=1 m`, `|vx|<=1 m/s`, `-2<=vz<=0 m/s`를 적용한 뒤 15일차 연료·접지·제어 변화 비용 중 병진 부분을 최소화합니다. 관련 정규화 위반량이 `0.001` 이하일 때만 각 단계의 해를 채택합니다.

기본 해석식 초기 추정치는 고도·수직속도로 하강 시간을 추정하고, 목표 위치와 수평속도 0으로 끝나는 수평 위치 3차 곡선을 만든 뒤 필요한 가속도를 제한된 throttle과 추력 각도로 바꿉니다. 이렇게 정한 제어를 적분해 사격 상태 node를 만듭니다. `--guess pid`를 선택하면 기존 통합 PID를 같은 초기상태에서 실행하고 적용 throttle과 순간 순추력 방향(`theta+gimbal`)을 최적화 격자에 표본화합니다. 표본화된 제어는 축소 모델의 경계로 제한한 뒤 5상태 모델로 다시 적분합니다. PID 궤적은 초기 추정치일 뿐이며, 최적해가 PID의 착륙 성공을 그대로 이어받는 것은 아닙니다.

IPOPT 수렴 후 저장된 제어를 설정된 더 작은 시간 간격으로 `simulate_planar`에 재생합니다. 각 구간마다 몸체 각도를 최적 `alpha`, 짐벌을 0으로 설정하고 다섯 병진 상태를 모든 사격 node에서 비교합니다. 이는 이상적인 추력 방향 병진 동역학과 node 사이 고도·연료 동작을 확인하지만, **물리적 회전·짐벌 변화속도·접지 자세는 검증하지 않습니다**. JSON에는 두 단계의 솔버 상태, 목적값, 반복 수, hard·종단 위반량, 다섯 상태의 물리 단위 RK4 결함과 재생 비교가 담깁니다. 솔버나 가능해 탐색 실패 시 그래프 없이 진단 JSON을 저장하며, 기존 결과 파일은 덮어쓰지 않습니다.

~~~bash
python scripts/solve_translation_landing.py --output-dir artifacts/day17-translation
python scripts/solve_translation_landing.py --guess pid --output-dir artifacts/day17-pid
python scripts/solve_translation_landing.py --initial-x -12 --initial-vx 1 --output-dir artifacts/day17-custom
~~~

기본 단일 조건은 `x=10 m`, `z=100 m`, `vx=0`, `vz=-20 m/s`, `m=1000 kg`입니다. 해석식과 PID 초기 추정치 모두 수렴했습니다. 해석식 초기 추정치의 이상 추력 방향 재생 결과는 약 `5.000 s`에 `x=0.003 m`, `vx=-0.005 m/s`, `vz=-0.902 m/s`로 끝나며 연료 `27.605 kg`을 사용했습니다. 이는 초기조건 한 건에서의 수평 오차 수정 결과이지 여러 조건의 성공률이 아닙니다. 18일차에는 자세를 즉시 바꾸는 가정을 없애고 실제 자세·gimbal 궤적을 풉니다.

## 18일차 전체 평면 3자유도 Teacher

18일차에는 전체 상태 `X=[x,z,vx,vz,theta,omega,m]`와 물리 제어 `U=[q,delta]`를 풉니다. 17일차와 달리 `delta`는 몸체 축 기준 gimbal 각도입니다. 병진 운동은 관성계 추력 방향 `theta+delta`를 사용하고 회전 운동은 `dtheta/dt=omega`, `domega/dt=-L*T_max*q*sin(delta)/I`를 사용합니다. CasADi 기호 미분식은 동일한 상태와 제어에서 기존 시뮬레이터 미분식과 일치함을 검사했습니다.

직접 다중 사격 문제는 구간별 일정한 제어 100개와 `[5,25] s` 자유 종료시간을 유지합니다. 모든 사격 node에 고도, 건조질량 위 1 kg 연료 여유, 몸체 기울기 `|theta|<=20 deg`, 각속도 `|omega|<=30 deg/s` 제한을 적용합니다. 모든 제어에는 물리 throttle 범위와 `|delta|<=15 deg`를 적용합니다. A단계는 여섯 착륙 조건만 정규화된 비음수 slack으로 일시 완화합니다. B단계는 최종 고도를 지면으로 고정하고 위치·속도·자세·각속도 착륙 제한을 hard constraint로 적용한 뒤 기존 연료·접지·제어 변화 목적함수를 최소화합니다.

통합 PID 제어기는 요청한 초기상태에서 한 번 실행되어 실용적인 초기 제어열을 제공합니다. 이 제어를 최적화 격자에 표본화하고 7상태 RK4 모델로 다시 적분하므로 초기 node는 동역학적으로 일관됩니다. PID 결과는 초기 추정치의 출처로만 저장됩니다. IPOPT 수렴, 독립 계산한 hard·종단 위반량 `0.001` 이하, 별도의 세밀한 시뮬레이터 재적분을 모두 통과해야 Teacher 궤적으로 채택합니다.

실패 보고서는 실패 단계, IPOPT 상태, 가장 큰 제약의 이름과 크기, 전체 hard constraint 분해값, 여섯 종단 위반량과 상태별 RK4 결함을 기록합니다. 따라서 단순한 infeasible 상태만 남기지 않고 고도, 연료, 구동기, 기울기, 각속도, 시간, 동역학과 종단 조건 중 무엇이 문제인지 구분할 수 있습니다.

독립 재적분은 구간 사이 `theta`와 `omega`를 연속으로 유지하고 최적 gimbal을 `simulate_planar`에 직접 적용합니다. 자세를 강제로 재설정하지 않습니다. 종단 조건 전체, node 사이 고도, 연료 여유, 기울기, 각속도, gimbal 크기와 모든 사격 node의 불일치를 검사합니다. 보고서와 그래프는 기존 파일을 덮어쓰지 않습니다.

~~~bash
python scripts/solve_planar_landing.py --output-dir artifacts/day18-planar-3dof
python scripts/solve_planar_landing.py --initial-x -8 --initial-theta-deg -2 --initial-omega-deg-s 1 --output-dir artifacts/day18-custom
~~~

기본 조건은 `x=10 m`, `z=100 m`, `vx=0`, `vz=-20 m/s`, `theta=3 deg`, `omega=-1 deg/s`, `m=1000 kg`입니다. 두 단계 모두 `Solve_Succeeded`를 반환했습니다. 현재 19일차 목적함수의 전체 모델 재적분은 `5.000 s` 후 약 `x=0.012 m`, `vx=-0.023 m/s`, `vz=-0.974 m/s`, `theta=0.101 deg`, `omega=-0.028 deg/s`로 끝나며 연료 `27.907 kg`을 사용했습니다. 최대 몸체 기울기는 `15.749 deg`, 최대 각속도는 `21.680 deg/s`, 최대 gimbal은 `14.671 deg`이고 상태 node 최대 불일치는 물리 단위 기준 `9e-7` 미만입니다. 이는 단일 명목 궤적이며 성공률, 외란 강건성, 구동기 대역폭, 구조 하중이나 하드웨어 안전 결과가 아닙니다.

## 19일차 목적함수와 수치 안정성 연구

19일차에는 전체 평면 B단계 목적함수를 mesh 사이에서 비교할 수 있게 만들었습니다. `ObjectiveScales`는 사용 가능한 연료, 다섯 종단 착륙 제한과 두 제어 변화율 기준을 명시적으로 기록합니다. Solver와 사후 평가기는 같은 정규화 연료·접지·변화율 기반 평활성 정의를 사용합니다. 회귀 테스트에서는 같은 선형 제어 ramp를 두 mesh에 적용했을 때 평활성 비용이 같음을 확인합니다.

설정된 sweep은 연료와 접지 가중치를 `1.0`, `0.1`로 고정하고 평활성 가중치 `0.001`, `0.01`, `0.1`을 비교합니다. 초기조건은 18일차 기본 사례와 같습니다. 측정된 trade-off는 다음과 같습니다.

| Profile | 연료 사용량 | 평활성 항 | 최대 throttle 변화율 | 최대 gimbal 변화율 |
|---|---:|---:|---:|---:|
| 연료 우선 | 27.858 kg | 0.11080 | 1.479/s | 50.013 deg/s |
| Baseline | 27.907 kg | 0.04708 | 0.697/s | 23.824 deg/s |
| 평활 제어 | 28.003 kg | 0.02661 | 0.478/s | 11.685 deg/s |

Baseline 가중치는 `N=25, 50, 100, 200`에서 각각 풉니다. 최적 제어열은 전사 node 결과만으로 채택하지 않고 기존 사건 처리 simulator에서 `0.005 s` 간격으로 다시 적분합니다. 최적화와 재적분의 최종 상태 차이는 `x,z`에서 `0.001 m`, `vx,vz`에서 `0.001 m/s`, `theta`에서 `0.01 deg`, `omega`에서 `0.01 deg/s`, 질량에서 `0.001 kg` 이하여야 합니다. 또한 재적분은 착륙 제약, 고도와 추진제 여유를 만족해야 합니다.

| 제어 구간 | 측정 풀이 시간 | 최대 최종 오차 / 허용오차 | 재적분 결과 |
|---:|---:|---:|---|
| 25 | 0.613 s | 0.231 | 통과 |
| 50 | 0.872 s | 0.0140 | 통과 |
| 100 | 1.476 s | 0.000870 | 통과 |
| 200 | 2.618 s | 0.00101 | 통과 |

설정된 모든 사례가 19일차 완료 기준을 통과했습니다. Solver 허용오차, 적응형 재적분과 보간도 오차에 영향을 주므로 mesh를 늘릴 때마다 오차가 반드시 단조 감소할 필요는 없습니다. 100구간과 200구간 모두 허용오차의 약 1천분의 1 수준입니다. 풀이 시간은 한 번의 개발 환경 실행에서 측정한 값이며 성능 보장이 아닙니다. 이 연구는 초기조건 한 건에서 명목 전사 일치를 확인한 결과입니다. Throttle·gimbal 변화율을 hard constraint로 강제하지 않으며, 20일차에는 solver를 재현 가능한 초기조건 batch로 확장합니다.

~~~bash
python scripts/analyze_optimal_control.py --output-dir artifacts/day19-study
~~~

명령은 모든 설정, 목적함수 항, solver 반복 수, 물리 단위 최종·node 오차와 완료 기준을 담은 JSON을 저장합니다. 또한 trade-off, 정규화 최대 변화율, mesh 풀이 시간과 정규화 재적분 오차를 보여주는 4패널 PNG를 생성합니다. 기존 출력 파일은 덮어쓰지 않습니다.

## 20일차 다중 초기조건 Teacher pipeline

20일차에는 단일 case 전체 평면 solver를 결정적인 batch pipeline으로 확장했습니다. `integrated_uniform_v1`은 통합 제어기 평가와 같은 7상태 중간 난이도 범위를 사용합니다. 수평 위치는 양방향 절대 오차 3-15 m, 고도 80-120 m, 수평속도 -2-2 m/s, 수직속도 -25--15 m/s, 자세 -5-5 deg, 각속도 -2-2 deg/s, 질량 950-1000 kg입니다. 기본 batch는 seed `20260920`의 100개 case이며 canonical 행렬 hash는 `7a58dc505c4535e8f081544cea22de17743bff0d80f34debc9454b2029a9f9b9`입니다.

Warm-start 출처를 결정적으로 유지하기 위해 runner는 순차 방식입니다. 최적화는 부모 process가 아니라 재사용 가능한 spawn worker에서 실행됩니다. 부모는 각 시도를 최대 30초만 기다리고 timeout이면 worker를 종료하므로, 풀이가 끝난 뒤 시간만 측정하는 것이 아니라 실제 제한을 강제합니다. 장시간 solver 메모리를 제한하기 위해 대기 상태 worker를 25회마다 교체합니다. Worker는 NumPy 기반 궤적과 구조화된 진단만 반환하고 solver console 출력은 숨깁니다.

첫 case는 통합 PID 궤적을 초기 추정치로 사용합니다. 한 case가 성공하면 throttle·gimbal 제어열과 종료시간을 복사하고, 다음 초기상태에서 제어열을 다시 적분해 동역학적으로 일관된 warm-start node를 만듭니다. 해당 시도가 solver 또는 재적분 검증에 실패하거나 timeout이면 새 PID 추정치로 한 번 재시도합니다. 성공 case는 solver 내부 A·B단계 검증과 `0.01 s` 독립 재적분을 모두 통과해야 채택합니다. 재적분은 19일차 상태별 일치 허용오차와 종단 제약, 고도, 추진제 여유, 몸체 기울기, 각속도와 gimbal 제한을 함께 확인합니다.

모든 시도에는 명시적인 상태가 있습니다. Solver 실패는 실패 단계, IPOPT 반환 상태, 이름이 붙은 최악 제약, 가능한 경우 전체 hard·종단 잔차와 RK4 결함을 저장합니다. Timeout 기록에는 설정 제한과 경과시간이 들어갑니다. Worker crash, protocol 불일치, 예상하지 못한 예외 종류와 재적분 실패는 서로 구분합니다. 최상위 `failures` 목록은 실패가 0건인 실행에서도 존재하며, 실패가 있으면 해당 case의 모든 시도를 포함합니다.

~~~bash
python scripts/generate_teacher_pipeline.py --output-dir artifacts/day20-teacher-pipeline
~~~

명령은 두 결과 중 하나라도 이미 있으면 덮어쓰지 않으며 최종 파일 두 개만 생성합니다.

| 파일 | 내용 |
|---|---|
| `teacher-pipeline.json` | 설정, batch hash, 완료 기준, case별 시도, solver 요약 지표, 재적분 검사, 전체 요약과 실패 metadata |
| `teacher-trajectories.npz` | 채택 case index, 초기상태, `101 x 7` 상태 node, `100 x 2` 제어, 101개 시간 node와 종료시간 |

기본 실행은 100개 case를 `126.246 s`에 처리했습니다. 100개 모두 첫 시도에 수렴하고 재적분을 통과했습니다. 0번 case는 PID, 1-99번 case는 직전 성공 궤적을 사용했으므로 재시도, timeout과 실패는 모두 0건입니다. A·B단계 평균 반복 수는 `37.54`, `29.09`였습니다. 평균 연료 사용량은 `27.349 kg`, 범위는 `24.669-30.571 kg`, 최적 종료시간 범위는 `5.000-5.762 s`였습니다. 최대 최종 재적분 오차는 설정 허용오차의 `0.00331`배였습니다. 압축 dataset의 상태 node 배열은 `(100,101,7)`, 제어 배열은 `(100,100,2)`입니다.

로드맵 완료 기준은 최소 100개 초기조건 처리 여부이므로 solver 성공과 실행 완료 기준을 별도로 보고합니다. 이번 실행에서는 100개 명목 궤적이 모두 채택됐지만, 바람, 센서 오차, 엔진 지연, 모델 불일치 또는 설정 범위 밖 초기조건의 강건성을 증명하지는 않습니다. 21일차에는 PID 기준선과 Teacher 품질을 비교하고 대표 궤적을 선정합니다.

## 21일차 Teacher 검증과 동결

21일차에는 `planar-teacher-v1`을 변경 불가능한 3주차 비교 protocol로 정의합니다. Canonical 설정 digest는 좌표 규약, 시뮬레이션·기체 parameter, 명목 환경, 착륙 제한, 초기 추정치에 사용하는 통합 PID 설정, 최적제어 정식화와 Teacher pipeline 정책을 모두 포함합니다. `pid-baseline-v1.yaml`의 digest는 `dfea2326159997cc0fccc8977e13e316856c6eb2ec5e26f97cbb07196f648793`입니다. 명목 시험 집합은 seed `20260920`의 100-case 행렬이며 SHA-256은 `7a58dc505c4535e8f081544cea22de17743bff0d80f34debc9454b2029a9f9b9`입니다. 원본 보고서와 궤적 archive도 각각 `11ef6d9f823686483dcf1d4106415097cb57db43a6899973d757fe1aa7dfe414`, `c19f897d63c51e6a8b5f79f5dbe8594e7bbd2578221ea4873f4652149905a9dd`로 동결합니다. 어느 digest든 바뀌거나 시험 sampler, case 수 또는 seed가 생성 pipeline과 다르면 검증을 중단합니다.

수치 평가 전 20일차 bundle loader는 보고서 problem·schema version, 정확한 batch 식별값, 순서와 누락이 없는 case 기록, 실패 case와 최상위 실패 목록의 일대일 대응, 요약 개수, NPZ field 집합, 배열 크기·dtype, 성공 case index, 초기상태, 엄격히 증가하는 시간 node, 종료시간, 구동기 경계와 JSON 배열 metadata를 확인합니다. NPZ는 pickled object를 허용하지 않고 엽니다. 따라서 오래됐거나 순서가 바뀌고 일부가 누락됐거나 수동으로 수정된 dataset을 PID와 조용히 비교할 수 없습니다.

채택된 모든 제어열은 `0.01 s` 간격으로 `simulate_planar`에서 다시 적분합니다. 이는 20일차 보고서에 저장된 Boolean 재적분 결과를 재사용하는 것이 아니라 저장된 초기상태, 제어와 종료시간으로 새로 수행한 21일차 계산입니다. 검증기는 최종·node 불일치, 정규화 종단 잔차, node 사이 고도, 추진제 여유, 몸체 기울기, 각속도, throttle과 gimbal 검사를 다시 계산합니다. 이름이 붙은 검사별 실패 수와 전체 batch의 최소 물리 margin도 보고합니다.

Teacher 100개 해가 모두 새 재적분을 통과했습니다. 상태별 허용오차 대비 최대 최종·node 오차는 `0.00331085`, 최대 정규화 종단 위반은 `2.63555e-6`으로 가능해 한계 `0.001`보다 작았습니다. Solver에 저장된 최대 B단계 hard constraint 잔차는 `8.89050e-9`였고, 이름이 붙은 모든 재적분 검사의 실패 수는 0입니다. 원시 재적분의 최소 기울기 margin은 `-0.00938 deg`였지만 판정에 명시된 자세 허용오차 `0.01 deg` 안이므로 통과했습니다. 최소 각속도 margin은 `1.84e-6 deg/s`, 최소 gimbal margin은 `2.50e-6 deg`, 최소 throttle 상한 margin은 `4.62e-8`이었습니다. 여러 해가 경계에 매우 가깝기 때문에 이 수치를 구동기나 모델 강건성 margin으로 해석하면 안 됩니다.

통합 PID는 같은 초기조건 100개와 무풍 조건에서 평가했습니다. 99개는 착륙했고 1개는 crash였습니다. 실패 case는 위치 제한을 `1.582 m`, 수평속도 제한을 `0.277 m/s`, 각속도 제한을 `1.851 deg/s` 넘었습니다. 나머지 case에는 종단 제한 위반이 없었습니다. Teacher의 전체 채택 case 평균 연료는 `27.349 kg`입니다. PID도 착륙한 99개의 짝지은 case에서 Teacher 평균은 `27.347 kg`, PID 평균은 `51.907 kg`으로, Teacher가 평균 `24.560 kg` 덜 사용했습니다.

Teacher pipeline 전체 측정시간은 `126.246 s`, 채택 시도 합계는 `123.566 s`, 채택 시도 평균은 `1.236 s`였습니다. PID 평가는 물리 시뮬레이션을 포함해 `20.661 s`가 걸렸습니다. PID 명령 계산 자체는 합계 `5.609 s`, controller update당 약 `0.483 ms`였습니다. Offline 비선형계획 풀이 시간과 online 제어 명령 시간은 서로 다른 측정값이므로 보고서는 이를 분리하며 두 값의 비율을 실시간 속도 향상으로 주장하지 않습니다.

대표 선정 규칙은 채택 집합의 연료 중앙값에 가장 가까운 성공 case이며, 차이가 같으면 case index가 작은 쪽을 선택합니다. 이에 따라 22번 case가 선정됐습니다. 연료 중앙값은 `27.3636 kg`, 선택 case는 `27.3641 kg`입니다. 제어열을 새로 재적분해 표준 episode log schema로 만들고 240-frame GIF로 렌더링했습니다. 마지막 재생 frame은 `5.000 s` 후 `x=-0.0091 m`, `vx=0.0194 m/s`, `vz=-0.9645 m/s`, `theta=-0.0878 deg`, `omega=0.0243 deg/s`에 도달했고 연료 `27.3641 kg`을 사용했습니다.

~~~bash
python scripts/validate_teacher.py
~~~

명령은 결과를 덮어쓰지 않으며 정확히 세 파일을 생성합니다. `teacher-validation.json`에는 protocol 검증, Teacher·PID 지표, 제약 분석, PID case별 결과, 독립 재적분 결과, 실패 집계, 비교와 통과 판정이 들어갑니다. `representative-teacher.json`은 선택된 표준 재생 episode이고 `representative-teacher.gif`는 해당 애니메이션입니다. 3주차 통과 기준은 solver 성공률 90% 이상, 모든 채택 해의 기존·신규 재적분 통과, 모든 최적화 실패의 별도 집계입니다. Protocol과 완료 기준 검사 6개가 모두 통과했습니다. 이 결과는 동결된 무풍 명목 표본에 한정되며 외란 강건성, 구동기 대역폭, 모델 불일치, hard 제어 변화율 제한과 하드웨어 안전은 아직 검증하지 않았습니다.

## 22일차 데이터셋 규약과 split 설계

22일차에는 대규모 궤적을 만들기 전에 offline Behavior Cloning 입력을 정의합니다. `planar-bc-v1` 상태 순서는 `[x,z,vx,vz,theta,omega,mass]`, 단위는 `m,m,m/s,m/s,rad,rad/s,kg`입니다. Action 순서는 `[throttle,gimbal_angle]`, 단위는 `1,rad`입니다. 상태가 `T+1`개인 채택 trajectory에는 구간 action `T`개가 있습니다. 앞의 상태 `T`개만 지도학습 입력으로 사용하고 종단 상태는 검증에만 사용합니다. 물리값은 float64로 저장하며 schema를 검사한 뒤 학습 loader가 batch를 float32로 바꿀 수 있습니다.

길이가 다른 trajectory는 split별 `packed_npz_v1` shard 하나에 이어 붙이고 별도의 상태·action offset으로 복원합니다. Pickle은 허용하지 않습니다. 각 기록에는 안정적인 trajectory·initial-condition ID, 원본 index, 종료시간, 시간·상태·action 값, solver 두 단계 반복 수, hard·종단 잔차, 재적분 검사와 정규화 오차비, 연료 사용량, 시도 수가 들어갑니다. 실패한 시도는 manifest에 명시하고 shard에서는 제외합니다. NaN·무한값, 잘못된 offset, 증가하지 않는 시간, 상태·action 개수 불일치, 재적분 실패와 initial-condition ID 중복은 모두 거부 조건입니다.

Split 단위는 완전한 trajectory입니다. Initial-condition ID는 split label 없이 7상태 vector의 canonical little-endian float64 byte에서 SHA-256으로 계산합니다. 한 split 안의 중복 또는 train, validation, test 사이의 겹침이 하나라도 있으면 생성을 실패 처리합니다. 각 split은 서로 다른 고정 seed를 사용합니다. Train과 validation은 같은 쉬운 범위라서 validation은 IID model 선택을 측정하지만, gradient와 normalization 통계에는 참여하지 않습니다.

| Field | Train·validation | Hard OOD test |
|---|---:|---:|
| 수평 절대 오차 | 2-10 m | 12-20 m |
| 고도 | 70-105 m | 110-140 m |
| 수평속도 | -1.5-1.5 m/s | -4-4 m/s |
| 수직속도 | -22에서 -15 m/s | -32에서 -24 m/s |
| 자세 | -3-3 deg | -10-10 deg |
| 각속도 | -1.5-1.5 deg/s | -5-5 deg/s |
| 질량 | 970-1000 kg | 900-960 kg |

Train, validation, test 계획은 각각 800, 100, 200개이며 seed는 `20260922`, `20261022`, `20261122`입니다. Test는 수평 절대 오차, 고도, 하강속도와 질량이 train과 엄격히 분리됩니다. 따라서 계획된 모든 hard-test case는 모든 train case보다 멀고, 높고, 더 빠르게 하강하며, 사용할 수 있는 추진제가 적습니다. 나머지 운동 범위도 더 넓습니다. 이 관계가 하나라도 없어지면 설정 검증이 실패합니다.

Normalization은 채택 train trajectory의 서로 정렬된 비종단 상태·action row만 사용해 standard score를 계산합니다. Feature별 scale은 `max(모집단 표준편차, 1e-6)`입니다. Count, 평균, 모집단 표준편차, 실제 scale, 최솟값과 최댓값을 float64로 누적·저장하고 각도는 rad를 유지합니다. 추론에서는 상태를 normalize하고 예측 action을 denormalize한 다음 물리 구동기 제한을 적용합니다. 실제 수치 통계는 23-24일차에 train trajectory를 생성한 뒤에만 계산할 수 있습니다.

~~~bash
python scripts/design_offline_dataset.py
~~~

명령은 JSON 규약 보고서와 결정적인 초기상태·ID 배열 6개를 담은 압축 NPZ를 생성하며 기존 결과를 덮어쓰지 않습니다. 22일차 완료 gate는 상태, action, trajectory, initial condition과 solver 품질 schema, 세 split 범위, trajectory leakage 부재, train 전용 normalization과 엄격히 더 어려운 test 범위를 검사합니다. 생성 파일은 Git에서 제외되는 `artifacts/day22-dataset-design/`에 둡니다. 사용 목적과 한계는 [한국어 데이터셋 카드](dataset-card.ko.md)와 [영문 데이터셋 카드](dataset-card.md)에 정리합니다.

## 23일차 대규모 쉬운 Teacher 궤적 생성

23일차에는 22일차의 쉬운 train 초기조건 800개 전체를 풉니다. 기존 timeout 격리 순차 worker에 호출자가 초기조건을 전달할 수 있는 재사용 batch interface를 추가하되, 동결된 20일차 실행 명령의 sampler와 정책은 유지합니다. 23일차 정책은 시도당 30초 제한, 25회마다 worker 교체, 직전 채택 제어열 warm start와 새 PID trajectory 재시도 1회를 사용합니다. 모든 제어 warm start는 현재 초기상태에서 다시 적분한 뒤 최적화에 전달합니다.

Generator는 첫 case와 이후 10개마다 경과시간, 평균 처리속도와 전체 실행 ETA를 출력합니다. 시간값은 개발 장비에서 측정한 설명용 값이며 성능 보장이 아닙니다. 측정 sample은 manifest에 저장합니다. 출력은 임시 파일을 만든 뒤 최종 경로로 원자적으로 교체하며, 두 결과 중 하나라도 이미 있으면 실행을 거부합니다.

Solver 성공만으로 train shard에 들어가지는 않습니다. 각 후보는 split 독립 initial-condition·trajectory ID 중복, 상태·action·시간·종료시간·품질값의 NaN 또는 무한값, fixed-mesh shape와 상태-action 개수 오류, 증가하지 않거나 종료시간과 맞지 않는 시간축, 초기상태 불일치, 구동기 경계, B단계 hard·terminal 잔차와 독립 계산한 종단·고도·추진제·기울기·각속도 제한을 검사합니다. Packed shard도 정확한 field와 dtype, offset, 유한값, ID 고유성, 상태 `T+1`개와 action `T`개의 관계, trajectory별 시간축, 재구성한 initial-condition ID와 accepted-only solver·replay flag를 다시 검사합니다.

Shard는 22일차 `packed_npz_v1` 규약을 사용합니다. Trajectory·initial-condition ID 797개, 원본 index, 상태·action offset 798개, 상태·시간 row 80,497개, action row 79,700개, 종료시간, solver 반복 수, hard·terminal 잔차, 재적분 오차비, 연료 사용량과 시도 수를 저장합니다. Canonical 배열 digest는 정렬한 field 이름, little-endian dtype, shape와 원시 byte를 사용하므로 NPZ ZIP timestamp에 의존하지 않습니다.

~~~bash
python scripts/generate_offline_dataset.py
~~~

계획된 800개를 모두 실행했습니다. 797개를 채택했고 3개는 previous-success와 PID 시도 모두 재적분 기울기 제한을 넘어 제외됐습니다. 637번 case는 첫 최적화가 실패했지만 PID 재시도에서 통과해 전체 재시도 case는 4개입니다. Timeout과 solver 성공 이후 filter 탈락은 0건입니다. 전체 시간은 `947.487 s`, 실행 case당 `1.184 s`, 채택률은 `99.625%`입니다.

채택 궤적의 평균 연료 사용량은 `26.594 kg`, 범위는 `24.785-28.704 kg`입니다. 평균 종료시간은 `5.003 s`, 범위는 `5.000-5.230 s`입니다. 저장된 B단계 최대 hard 잔차는 `6.758e-9`, 최대 terminal 잔차는 `1.770e-6`, 세밀한 재적분의 상태별 허용오차 대비 최대 오차는 `0.002351`입니다. Shard canonical SHA-256은 `04c0f2c0ff3c2e77a28276b37cd8005d2274dad041fbca2e145cd0aec9fe0876`입니다.

완료 gate는 계획된 split 전체 실행, 모든 filter를 통과한 trajectory 최소 500개와 최종 packed shard의 모든 구조·품질 검사 통과를 요구합니다. 모든 검사가 통과했습니다. `train-manifest.json`과 `train-trajectories.npz`는 Git에서 제외되는 `artifacts/day23-easy-dataset/`에 보관합니다. 이는 쉬운 train shard이며 최종 동결 offline dataset은 아닙니다. 24일차에는 더 넓고 어려운 조건을 추가하고 coverage를 시각화한 뒤 부족한 구간을 보강합니다.

## 24일차 최종 offline dataset

24일차에는 검증된 23일차 원본 shard를 변경하지 않고 `planar-bc-v1-final`을 생성합니다. 생성 전에 23일차 manifest와 NPZ를 `allow_pickle=False`로 읽고 dataset 설정 digest, 완료 gate, packed schema, 재구성한 initial-condition ID와 canonical shard SHA-256을 확인합니다. 하나라도 다르면 새 solver process를 시작하기 전에 중단합니다.

Challenge train 범위는 수평 절대거리 `8-11 m`, 고도 `90-108 m`, 수평속도 `-3-3 m/s`, 수직속도 `-23.5`에서 `-20 m/s`, 자세 `-7-7 deg`, 각속도 `-3-3 deg/s`, 질량 `962-980 kg`으로 easy 범위를 확장합니다. Hard OOD test는 수평거리, 고도, 하강속도와 질량이 확장 train과도 엄격히 분리되며 수평속도, 자세와 각속도 범위도 더 넓게 유지합니다.

어려운 case 과표집은 seed `20260924`로 challenge 후보 640개를 먼저 만듭니다. 각 후보는 수평 절대거리, 고도, 수평속도 절대값, 하강속도, 자세 절대값, 각속도 절대값과 낮은 질량의 정규화 난이도 7개 평균을 받습니다. 점수가 높은 120개와 나머지 중 무작위 40개를 선택합니다. 후보 전체 평균 점수 `0.5010`이 선택 집합에서는 `0.6079`로 높아지며, 일부 무작위 표본을 남겨 분포 폭이 지나치게 좁아지는 것을 막습니다.

Coverage는 easy와 challenge train 범위의 합집합에서 계산합니다. 수평 위치는 목표까지의 절대거리로 바꾸고 자세와 각속도는 degree로 바꾼 뒤 7개 상태 feature를 각각 고정 8구간으로 나눕니다. 모든 marginal 구간에 채택 train trajectory가 최소 30개 있어야 합니다. Challenge 생성 뒤 전체 deficit은 51개였습니다. 보강 sampler는 부족 feature를 번갈아 선택하고 하나의 목표 feature를 부족 구간 안에 고정하며 나머지는 최종 train 범위에서 표본화합니다. 중복 initial-condition ID는 제거하고 같은 Teacher 정책으로 풉니다. 각 round가 끝날 때 채택 trajectory만으로 coverage를 다시 계산하며 최대 추가 case는 160개, 최대 round는 3회입니다.

실제 실행에서는 challenge 160개가 모두 채택됐습니다. Coverage 1차 보강은 51개 중 50개를 채택했습니다. 다른 feature의 무작위 배치가 실패 case가 목표로 한 구간도 채워 2차 보강은 필요하지 않았습니다. 최종 구간별 최소 count는 수평 절대거리 105, 고도 100, 수평속도 30, 수직속도 91, 자세 34, 각속도 31, 질량 76입니다. Coverage 검사 7개가 모두 통과했습니다.

최종 train shard는 23일차 797개, challenge 160개와 coverage 50개를 합친 trajectory 1,007개이며 정렬된 비종단 상태-action pair 100,700개를 담습니다. IID validation은 `100/100`, hard OOD test는 `191/200`, 즉 `95.5%`를 채택했습니다. Hard test 실패 9개는 warm-start와 PID 시도 모두 세밀한 simulator 재적분에 실패했습니다. Timeout은 없고 solver 성공 결과가 중복, 비유한 값, 구조 또는 제약 filter에서 추가 제외된 사례도 없습니다.

최종 shard 3개는 정확한 field, dtype, offset, 유한값, accepted-only, 시간축과 initial-condition identity 검사를 독립적으로 통과했습니다. 안정적인 ID 검사 결과 split 내부 중복과 train, validation, test 사이 겹침은 모두 0개입니다. Normalization 통계는 최종 train pair 100,700개만 사용하며 validation과 test 값은 fit에 들어가지 않습니다.

~~~bash
python scripts/finalize_offline_dataset.py
~~~

명령은 기존 결과를 덮어쓰지 않고 `artifacts/day24-final-dataset/`에 `dataset-manifest.json`, `normalization.json`, `dataset-distribution.png`와 split별 packed NPZ를 원자적으로 저장합니다. Canonical shard SHA-256은 train `4161bc93b6753bdcc84cfb6eec6afe1c45bbf814b34525817a7ec24835691e0d`, validation `e77b561f63f5738ec6e09e24b9dbffe2e4712ff33e7d186d12b4945e2aaa618c`, test `6b67c7866e202b6719f07c733cd2cd82fdd3e2f1638c5bb33ec10ea3f3f5ba17`입니다. 원본, 과표집, 난이도, coverage, 개수, shard, split integrity와 normalization 완료 검사가 모두 통과했습니다.

## 25일차 Behavior Cloning 학습

`scripts/train_behavior_cloning.py`는 `configs/bc-day25.yaml`과 24일차 최종 데이터셋을 읽습니다. 먼저 manifest 완료·split 무결성, dataset 설정과 Teacher digest, train 전용 normalization, train·validation shard의 schema·개수·canonical SHA-256을 검사합니다. Hard OOD test shard는 읽지 않습니다. 각 trajectory에서 마지막 종단 상태를 제외한 상태 `T`개와 구간 action `T`개를 순서대로 짝지어 train 100,700쌍과 IID validation 10,000쌍을 만듭니다.

첫 정책은 `7-64-64-2` 폭의 ReLU 은닉층 2개와 선형 정규화 action 출력층을 사용합니다. 상태·action은 24일차 train에서만 계산한 표준점수로 변환합니다. 손실은 두 정규화 action의 제곱오차 평균입니다. 미니배치 Adam은 batch 1,024개, 학습률 0.001, seed `20260925`, 최대 80 epoch, patience 12, validation 최소 개선량 `1e-5`를 사용합니다. IID validation MSE가 가장 낮은 checkpoint를 최종 보고·저장합니다. 추론에서는 7상태를 정규화하고 예측 action을 물리 단위로 되돌린 뒤 설정된 throttle·gimbal 구동기 한계로 제한합니다.

전체 학습 전 독립 모델이 train 16쌍에서 2,000단계 안에 정규화 MSE `0.0025` 이하로 내려가야 합니다. 실제 검사는 125단계에서 `0.000707`로 통과했습니다. 전체 학습에서는 80 epoch 중 79 epoch checkpoint가 선택됐습니다. 해당 모델의 train MSE는 `0.024989`, IID validation MSE는 `0.030977`, validation throttle MAE는 `0.032989`, gimbal MAE는 `0.272693도`입니다. 이번 실행은 epoch 상한에 가까울 때도 validation이 개선되어 조기 종료가 발생하지 않았으며, 별도 테스트에서 조기 종료 분기를 확인했습니다.

~~~bash
python scripts/train_behavior_cloning.py
~~~

Git에서 제외한 `artifacts/day25-bc/`에는 pickle을 사용하지 않는 NPZ checkpoint와 전체 epoch 기록, 물리 단위 action 오차, 부분집합 과적합 gate, 설정, 데이터 출처와 실행시간이 담긴 JSON 보고서가 있습니다. Checkpoint에는 정규화와 구동기 제한도 포함됩니다. `load_checkpoint`는 parameter 모양, float32 dtype, 유한값과 정확한 NPZ field를 검사합니다. 25일차 수치는 열린고리 지도학습 오차만 보여주며 폐루프와 hard OOD 결과는 아래 절에서 측정합니다.

## 26~35일차 폐루프 정책 실험

`scripts/advance_imitation_learning.py`는 결정적인 train/validation 분리로 다음 단계를 실행합니다. `--through`에는 26, 28, 32, 34, 35를 지정할 수 있습니다. Hard OOD test shard는 35일차 모델 선택과 checkpoint 동결을 마친 다음에만 읽습니다.

| 단계 | 구현과 확인 결과 |
|---|---|
| 26일차 | 64-ReLU 기준선, 96-tanh, 128-ReLU와 행동·말기 표본 가중치를 비교하고 IID 검증 MSE 최솟값을 선택했습니다. 각각 0.03098, 0.04185, 0.03692로 기준선을 유지했습니다. |
| 27~28일차 | 검증 초기조건 100개에서 학습 명령은 0.1초마다, RK4 물리는 0.02초마다 갱신했습니다. BC 14/100, PID 100/100을 비교하고 실패 유형, 정규화 최근접거리 분포 이탈, MP4와 checkpoint metadata를 확인했습니다. |
| 29~32일차 | **train** 초기조건에서 학생을 실행하고 반복마다 방문 상태 48개를 뽑아 평면 최적제어 Teacher로 다시 풀었습니다. Solver·재적분 실패는 제외하고 새 라벨을 약 5% 비율로 재생 학습했습니다. 채택 라벨은 43, 35, 47개입니다. 선택된 v4 모델은 개발 검증 32건에서 BC 4/32 대비 5/32였습니다. |
| 33일차 | 다른 검증 32건에서 원래 명령과 변화율 제한, 센서 오차, 바람·추력 오차 결합을 비교했습니다. Train 20,000행에만 센서 오차를 더한 별도 증강 후보는 원래 상태 5/32, 센서 오차 2/32이며 최종 동결 비교에는 넣지 않았습니다. 제한은 throttle `2/s`, gimbal `60 deg/s`입니다. 최대 변화율 계산에서는 지면 접촉·연료 차단으로 짧아진 마지막 스텝을 제외합니다. |
| 34일차 | 선택 사항인 residual 정책 실험으로 PID 행동 25%, 50%, 75% 혼합을 평가했습니다. 모두 0/32여서 채택하지 않았고 PPO 학습은 실행하지 않았습니다. |
| 35일차 | Hard OOD 초기조건 191개를 읽기 전에 BC·DAgger checkpoint를 복사하고 SHA-256을 확인했습니다. BC 0/191, DAgger 3/191, PID 191/191을 비교하고 추론 시간, 실패·개선 MP4를 저장했습니다. |

DAgger는 기존 train 100,700쌍에 채택된 방문 상태 라벨을 반복해 epoch당 약 5%가 되도록 학습합니다. 동결된 24일차 데이터셋 자체는 바꾸지 않습니다. 학생이 바람 0.8 m/s 또는 추력 5% 저하 조건에서 방문한 상태에도 OCP Teacher는 명목 동역학으로 풉니다. 이 모델 불일치는 라벨 품질에 영향을 줄 수 있습니다. 개발 검증 32건과 전체 IID 검증 100건은 hard OOD test가 아닙니다. 32일차의 명목 성능 비하락 목표는 달성하지 못했습니다. 전체 검증 성공은 BC 14/100, 변화율 제한 DAgger 11/100입니다. 35일차 OOD 착륙 3건 증가만으로는 실사용을 주장할 수 없습니다. 실패 원인과 사용 제한은 [모델 카드](model-card.ko.md)에 있습니다.

## 36~38일차 잔차 착륙 환경과 명령 필터

`configs/rl-evaluation-v1.yaml`은 여섯 조건군과 기준 seed `378`을 고정합니다. 학습, 개발, 최종 보류 스트림 ID는 각각 `0`, `1`, `2`이며 최종 보류 사례 생성은 기본적으로 잠겨 있습니다. 36일차 PID 조사는 개발 사례만 사용하고 Teacher 성공 여부로 초기상태를 거르지 않습니다. `scripts/prepare_rl_protocol.py`는 180행 결과와 조건 범위·선택 규칙을 Git에서 제외되는 `artifacts/day36-rl-protocol/`에 저장합니다.

`ResidualLandingEnv`는 PID 조사와 같은 물리·센서·구동기 실행기를 사용합니다. 정규화 잔차 행동 `[-1,1]^2` 하나를 `0.1 s`마다 받고 선택된 명령을 `0.02 s` 물리 스텝 다섯 개 동안 구동기 경로에 보냅니다. 행동 지연과 엔진 지연이 있으면 실제 적용 명령은 달라질 수 있습니다. 필터 전 후보는 `PID + [0.08 * residual_throttle, radians(2) * residual_gimbal]`이며 두 번째 성분의 물리 단위는 radian입니다. 17개 float32 관측값의 순서는 다음과 같습니다. 지연·잡음이 적용된 물리 상태 7개를 `[30 m,160 m,10 m/s,40 m/s,20 deg,30 deg/s,250 kg]`로 나눈 값(고도는 지면 기준, 질량은 건조질량 기준), `[1,gimbal_limit]`로 나눈 PID 행동 2개, 한계값으로 나눈 접근·말기 PID 적분항 2개, `40 m/s`로 나눈 두 단계의 이전 수직 오차 2개, 말기 phase 표시 1개, 같은 행동 스케일로 나눈 대기 명령 2개, 실제 적용 throttle 1개입니다. 참 상태는 진단용 `info`에만 두고 정책 관측에는 넣지 않습니다.

명령 필터는 잘못된 모양, 비유한 값, 잔차 범위 초과, 구동기 한계, `2 throttle/s` 또는 `60 deg/s` 변화율 초과를 PID 복귀 이유로 기록합니다. 후보의 연료 소비가 `1 kg` 예비량을 침범하면서 PID보다 throttle을 높이는 경우와, 예측 자세·지면 정지 여유가 이미 부족한데 PID보다 악화되는 경우에도 복귀합니다. 예측은 명목 모델의 `0.1 s` 구간, `20 deg` 자세 한계, `0.5 m` 지면 정지 여유를 사용합니다. 모델 오차와 센서 지연에서는 특히 정확한 안전 판정이 아니며 물리적 안전을 증명하지 않습니다. 이전에 잔차를 적용한 뒤 PID로 복귀해도 과거 PID 궤적으로 되돌아가는 것은 아닙니다. 기본 환경은 희소 종료 보상을 유지하고 39일차 래퍼가 학습용 보상을 따로 제공합니다.

제어 갱신마다 PID 기준 명령, 제안 잔차, 선택 명령, 복귀 이유, 누적 개입 수와 물리 결과를 기록합니다. 필터 실행시간은 결정적 Gymnasium `info`가 아닌 `env.safety_timing_us`에 보관합니다. `scripts/verify_residual_landing.py`는 조건군마다 학습 사례 3건에 `[-0.25,0.25]` 균등 잔차를 넣어 덮어쓰기 없는 JSON 보고서를 만들며 학습 정책 성능을 주장하지 않습니다. 잔차 0은 여섯 조건군의 모든 물리 상태·명령에서 PID와 같았고, 재구성한 PID 180건 CSV의 이전 canonical row digest `f4c2dc64c5bdcd4086bfacce14d82e98b5602552e92baae15048c49fa46a3b52`도 유지했습니다.

## 39~40일차 보상·커리큘럼·SAC 예비 학습

`LandingRewardWrapper`는 학습 보상만 바꾸고 물리 상태 변화와 착륙 결과는 그대로 둡니다. 제어 스텝의 보상은 `potential(next) - potential(previous) - 0.1 * 경과초 + 종료 보너스`입니다. Potential은 `[-11,0]`이며 수평 오차, 지면 위 고도, 수평·수직 속도, 자세, 각속도의 제한된 벌점 합입니다. 연료 항은 없습니다. 종료 보너스는 성공 `+100`, 과속 착륙 `-120`, 충돌 `-150`, 연료 고갈·시간 초과 `-130`입니다. Potential의 차이는 한 에피소드에서 망원합으로 소거되고 시뮬레이터 시간도 유한합니다. 공중 정지는 시간 비용이 들며 실패 누적 보상은 성공 누적 보상보다 낮습니다. 이 설계가 안전을 보장하거나 짧은 학습에서 의도한 정책을 얻는다는 뜻은 아닙니다.

`CurriculumResidualEnv`는 seed `378` 생성기로 **train ID만** 뽑습니다. 단계는 명목, 초기상태 OOD, 바람, 단일 외란, 복합 외란이며 단계마다 학습 조건·난도 쌍의 가중치가 고정돼 있습니다. 단계 변경은 다음 리셋부터 적용하고 진행 중인 에피소드에는 원래 단계 이름을 남깁니다. 생성기 상태와 에피소드 번호를 체크포인트에 보관합니다. 개발 사례는 별도 규약 스트림에서 뽑고 최종 보류 집합은 예비 학습에서 생성하지 않았습니다.

`configs/sac-pilot-v1.yaml`은 명목·초기 OOD·바람을 각각 2,000스텝씩, 총 6,000스텝으로 고정합니다. CPU SAC, `64x64` 신경망, 리플레이 용량 20,000, 초기 수집 200스텝, 배치 64, 학습률 `3e-4`, 할인율 `0.99`, 목표망 갱신 `0.005`, 환경 스텝당 경사 갱신 1회를 사용합니다. Actor 평균 출력층을 0으로 초기화해 처음 결정론적 잔차가 정확히 0입니다. 초기 개발 사례 36건도 PID 결과와 같았습니다. 0·2,000·4,000·6,000스텝마다 모델·리플레이 체크포인트와 SHA-256 manifest, 학습 에피소드, PID와 짝지은 사례별 개발 평가, PNG 학습 곡선을 저장합니다. `--until N`, `--resume`으로 예산 내 재개할 수 있으며 모델·리플레이·커리큘럼 표본 생성기를 복구합니다. 실행 재개가 가능하다는 뜻이지 중단하지 않은 실행과 비트 단위로 같은 학습을 보장하지는 않습니다.

| 환경 스텝 | 명목 | 초기 OOD | 바람 | PID 성공 사례의 새 실패(조건군 순서) |
|---:|---:|---:|---:|---|
| PID / 0 | 12/12 | 12/12 | 6/12 | 0, 0, 0 |
| 2,000 | 12/12 | 12/12 | 7/12 | 0, 0, 0 |
| 4,000 | 11/12 | 10/12 | 7/12 | 1, 2, 3 |
| 6,000 | 9/12 | 10/12 | 6/12 | 3, 2, 2 |

이 예비 학습에서 정책 실패는 모두 충돌로 판정됐습니다. 2,000스텝은 이 36건에서 PID 성공을 잃지 않고 바람 1건을 구조했지만 작은 개발용 탐색 결과입니다. 6,000스텝 모델은 악화됐으므로 PID 대신 선택하지 않습니다. 최종 보류 성능, 독립 seed 강건성 또는 6조건군 완료를 주장할 근거는 아닙니다.

## 41~42일차 다중 seed 강건성·필터 통제 실험

`scripts/advance_residual_sac.py`는 규약의 기준 seed `378`에 `numpy.SeedSequence(378).spawn(3)`을 적용해 학습 seed `3024638723`, `2326541261`, `1907174973`을 얻습니다. 41일차에는 40일차 예비 모델을 복사하지 않고 seed마다 잔차 0 actor를 별도로 초기화했습니다. 고정 CPU `64x64` SAC의 학습률은 `1e-4`, 리플레이 용량 20,000, 초기 수집 200스텝, 배치 64, 할인율 `0.99`, 환경 스텝당 갱신 1회입니다. 8,000스텝 커리큘럼 경계는 명목 1,000, 초기 OOD 3,000, 바람 5,000, 모델 불일치 8,000입니다. 학습 표본 생성기는 train ID만 사용합니다.

모든 체크포인트는 여섯 조건군 각각의 동일한 개발 사례 12건에서 평가합니다. 사례별 PID·정책 결과, 짝지은 구조·신규 실패, 충돌, 설명용 연료 사용량, 제어 갱신 수, PID 복귀율·이유, throttle/gimbal 한계 초과 축을 기록합니다. 연료는 모델 선택 기준이 아닙니다. seed마다 모델·리플레이 파일, SHA-256 manifest, 표본 생성기 상태, 에피소드 기록과 학습 곡선을 분리해 저장합니다. `--resume`은 마지막 완료 체크포인트에서 이어 실행하며 최종 보류 집합은 잠겨 있습니다. 같은 개발 사례 12건을 세 학습 seed에 반복했다고 독립 시험 사례가 36건이 되지는 않습니다.

8,000스텝 안전 착륙 수는 명목 / 초기 OOD / 바람 / 모델 불일치 / 센서·지연 / 복합 외란 순서로 다음과 같습니다.

| 정책 | 명목 | 초기 OOD | 바람 | 모델 불일치 | 센서·지연 | 복합 외란 |
|---|---:|---:|---:|---:|---:|---:|
| PID | 12 | 12 | 6 | 8 | 6 | 0 |
| seed 0 | 12 | 12 | 7 | 11 | 8 | 0 |
| seed 1 | 12 | 12 | 8 | 8 | 7 | 0 |
| seed 2 | 12 | 12 | 9 | 6 | 8 | 0 |

모든 칸의 분모는 12입니다. seed 2의 모델 불일치는 PID 실패 2건을 구조했지만 **PID 성공 4건을 새로 실패**해 순감소했습니다. 복합 외란은 세 seed 모두 0/12이고 시뮬레이터 `crash` 판정은 각각 12, 12, 11건이었습니다. 보고서의 `severe_crashes` 필드는 이 결과 라벨을 세며 실제 구조 손상 측정값은 아닙니다. 센서·지연의 제어 갱신 대비 PID 복귀율은 `10.1%`, `16.1%`, `14.9%`였습니다. 동결 체크포인트를 명령 축 진단 코드로 다시 평가했을 때 throttle 변화율 거절은 `121`, `132`, `176`회이고 gimbal은 `10`, `18`, `34`회였습니다. 이를 근거로 42일차에 매개변수 하나를 비교했으며 필터가 모든 충돌의 원인이라고 단정하지 않습니다.

42일차의 두 분기는 seed별 **동일한** 41일차 8,000스텝 모델과 리플레이 버퍼에서 시작합니다. 둘 다 센서·지연 중심 혼합 단계 2,000스텝과 복합 외란을 포함한 더 넓은 혼합 단계 2,000스텝을 학습하고 같은 개발 사례로 평가합니다. Control은 기존 필터를 유지하고 `reduced_throttle`은 정규화 throttle 잔차 한도 **하나만** `0.08`에서 `0.04`로 바꿉니다. 설정 검사는 control이 출발 필터와 같고 두 분기의 차이가 정확히 한 항목인지 확인합니다. 12,000스텝 결과는 다음과 같습니다.

| 분기 / seed | 명목 | 초기 OOD | 바람 | 모델 불일치 | 센서·지연 | 복합 외란 |
|---|---:|---:|---:|---:|---:|---:|
| Control 0 | 9 | 11 | 8 | 9 | 5 | 1 |
| Control 1 | 9 | 11 | 5 | 6 | 6 | 1 |
| Control 2 | 10 | 12 | 7 | 8 | 6 | 1 |
| 잔차 축소 0 | 11 | 7 | 5 | 6 | 9 | 0 |
| 잔차 축소 1 | 9 | 7 | 5 | 3 | 3 | 0 |
| 잔차 축소 2 | 12 | 11 | 6 | 9 | 8 | 1 |

모든 칸의 분모는 12입니다. 몇 분기에서 복합 외란 1건이 착륙했지만 명목 등 기존 성공 조건의 신규 실패가 함께 발생했습니다. 어느 분기도 조건군별 비하락 규칙을 통과하지 못하므로 한도 축소안을 기각하고 PID를 권장 기준선으로 유지합니다. 이 개발용 12건 관측을 최종 강건성 추정치나 하드웨어 안전 증거로 사용하지 않습니다. 최종 보류 사례는 생성하지 않았습니다.

~~~bash
python -m scripts.advance_residual_sac
python -m scripts.advance_residual_sac --settings configs/sac-day42-v1.yaml --output-dir artifacts/day42-sac
~~~
