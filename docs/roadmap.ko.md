# DM100 개발 로드맵 (2D 700 km 궤도 RTLS 완성 → 3D 6자유도 MuJoCo + Unreal Engine 5)

[README](../README.ko.md) | [모델 명세](spec.ko.md) | [검증 설계](validation.ko.md) | [검증 결과](results.ko.md)

이 문서는 **100톤 메탈록스 700 km 원궤도(1,800 kg) 투입 + 1단 재사용 RTLS 시뮬레이터 DM100**의 2D 완성 아키텍처와, 이를 계승해 별도의 신규 저장소에서 진행할 **3차원 6자유도(6DOF) 물리엔진(MuJoCo) + 시네마틱 시각화(Unreal Engine 5)** 확장 계획을 정리합니다.

---

## 1. 전체 아키텍처 전략

```text
[Stage 1: 본 저장소 · 2D Orbital RTLS] → [Stage 2: 신규 3D 저장소 · MuJoCo 6DOF] → [Stage 3: UE5 시네마틱 영상]
  · 100 t 7× 메탈록스 · 700 km 원궤도       · MJCF 1단 + 독립 4핀 + 4다리              · 50 Hz ENU+Quaternion 리플레이
  · Plant ↔ Forces ↔ GNC 완전 분리         · 2D Forces/GNC → 6DOF 3D 확장            · UE5 Python Sequencer 임포트
  · 25~43 km 무동력 격자핀 활강            · MEKF 쿼터니언 + 3D SOCP + 롤/요 배분     · Niagara 메탈록스 플룸·재진입
  · 3→1 호버슬램 · 487/487 전 제약 ≤ 80%   · 3D 횡풍·크로스레인지 Headless 검증        · 지상 / 온보드 / 착륙장 멀티캠
  · 접지 후 5초 착륙다리 접촉·직립 동역학
```

### 역할 분리 원칙
- **물리·검증 엔진 (2D RK4 → 3D MuJoCo)**: `float64` 배정밀도, RK4 적분, 착륙다리 접촉·전복 동역학, 고속 Headless 배치 실행 및 결정론적 재현성 검증을 담당합니다.
- **시각화 렌더러 (Unreal Engine 5)**: 물리 계산에 개입하지 않고, 검증을 통과한 <strong>표준 50 Hz 리플레이 JSON(<code>dm100_replay_3d_v1</code>, 오른손 ENU 미터 + 단위 쿼터니언 <code>wxyz</code>)</strong>만 불러와 오프라인 시네마틱 영상을 생성합니다.

---

## 2. Stage 1 (완료): 2D DM100 700 km Orbital Reusable RTLS

- **인터페이스 분리 (`rtls/interfaces.py`, `rtls/forces.py`)**: `TelemetryPacket → ActuatorCommand` 경계로 제어기와 물리엔진을 격리하고, 순수 외력 모듈을 분리해 추후 MuJoCo의 `xfrc_applied` 콜백으로 그대로 이식할 수 있습니다.
- **700 km 원궤도 + 8단계 비행 시퀀스 (`rtls/guidance.py`, `rtls/orbit.py`)**:
  1. `ascent` + `rtls/orbit.py`: 7기 메탈록스 클러스터 상승, 고고도(`53~71 km`) 콜드 분리 및 2단 구면 지구 2회 연소 700 ± 15 km 원궤도(1,800 kg) 투입 판정.
  2. `flip` → `boostback` → `coast`: 저동압 RCS 반전, 3엔진 부스트백, 정점 `120~170 km` 탄도 귀환 및 격자핀 전개.
  3. `entry_burn`: `0.85 s` 점화 스로틀 램프와 동압·열유속 이중 트리거로 재진입 동압 `≤ 33.31 kPa`, 정체 열유속 `≤ 21.05 kW/m²` 유지.
  4. `aero_glide`: **주추진제 명령 0% 무동력 격자핀 활강(`25.0~43.0 km`)**, 초음속 고고도 `5°` 받음각 상한 및 인과적 온라인 항력·밀도 배율 추정 기반 IIP 유도.
  5. `landing_burn` → `touchdown_settle`: 3→1엔진 최소 스로틀(36%) 호버슬램(해석적 자세 복귀 예측기 `_predict_zero_tilt_state` + 필터링된 `bias_x` + 볼록 LP 참조 + 우선순위 제어 할당) 및 **접지 후 5초 무동력 착륙다리 직립 안정성** 판정.
- **전수 검증 (`scripts/evaluate_validation.py`)**: 100% 동시 최악 코너를 포함한 487케이스 **487 / 487 (100.0%) 통과**, 16개 전 제약 최대 사용률 **`79.3%`**(80% 여유 달성).
- **오프라인 로컬 브라우저 UI (`local_app.py`, `ui/index.html`)**: 옥스포드 블루(`#002147`) 기반 2D 실시간 Canvas 리플레이, 활강·제어 그래프, 8단계 연료 및 2단 700 km 궤도 분석표, 50 Hz 3D 리플레이 JSON 내보내기, 선택적 MP4 렌더링 지원.

---

## 3. Stage 2 & Stage 3 계획 (별도 신규 3D 저장소에서 착수)

본 저장소(`DM100-2D-Rocket-Recovery`)는 2D DM100 궤도 RTLS 플랫폼을 완성·유지하며, Stage 2(3D MuJoCo 6DOF)와 Stage 3(Unreal Engine 5)은 별도의 **신규 3D 저장소**에서 시작합니다.

| 단계 | 핵심 구성 | 2D → 3D 매핑 |
|---|---|---|
| **Stage 2: 3D MuJoCo 6DOF** (신규 3D 저장소) | MJCF 1단 강체 + 독립 그리드핀 4개 + 착륙다리 4개 스트럿, `float64` RK4, `xfrc_applied` 외력 콜백 | `theta → quaternion`, 1D 핀 → 4핀(롤·피치·요), 1축 짐벌 → 2축 TVC 짐벌, 2D LP → 3D SOCP, 3D 횡풍·크로스레인지(Y축) |
| **Stage 3: UE5 시네마틱 영상** (신규 3D 저장소) | 50 Hz 리플레이 JSON → UE5 Python Level Sequence 자동 임포트 (`ENU m → UE5 cm`), Niagara 메탈록스 플룸·RCS·재진입 가열, Movie Render Queue 렌더 | 시뮬레이션과 완전 분리된 오프라인 리플레이 파이프라인 |
