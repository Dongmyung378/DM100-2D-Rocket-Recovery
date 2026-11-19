# DM100 개발 로드맵 (2D RTLS 완성 → 신규 3D MuJoCo + Unreal Engine 5 저장소)

[README](../README.ko.md) | [모델 명세](spec.ko.md) | [검증 설계](validation.ko.md) | [검증 결과](results.ko.md)

이 문서는 **2차원 SpaceX(Falcon 9) 방식 1단 재사용 귀환(RTLS) 시뮬레이터 DM100**의 아키텍처와, 이를 계승해 별도의 신규 저장소에서 진행할 **3차원 6-DOF 물리엔진(MuJoCo) + 시네마틱 시각화(Unreal Engine 5)** 확장 계획을 정리합니다.

---

## 1. 전체 아키텍처 전략

```text
[Stage 1: 본 저장소 — 2D DM100 RTLS] ──▶ [Stage 2: 신규 3D 저장소 — MuJoCo 6-DOF] ──▶ [Stage 3: UE5 시네마틱 영상]
  · Plant ↔ Forces ↔ GNC 완전 분리          · MJCF 1단 + 독립 4핀 + 4다리                · 50 Hz ENU+Quaternion 리플레이
  · SpaceX형 7단계 RTLS + 무동력 핀 활강     · 2D Forces/GNC → 3D 확장                    · UE5 Python Sequencer 임포트
  · 3→1엔진 최소 스로틀(44%) 호버슬램        · MEKF 쿼터니언 + 3D SOCP                    · Niagara 플룸 / RCS / 재진입 효과
  · 해석적 자세 복귀 예측 + 인과적 항력 추정   · Headless 3D 몬테카를로                     · 지상 / 온보드 / 착륙장 멀티캠
  · 접지 후 5초 착륙다리 접촉·직립 동역학
```

### 역할 분리 원칙
- **물리·검증 엔진 (2D RK4 → 3D MuJoCo)**: `float64` 배정밀도, RK4 적분, 착륙다리 접촉·전복 동역학, 고속 Headless 배치 실행 및 결정론적 재현성 검증을 담당합니다.
- **시각화 렌더러 (Unreal Engine 5)**: 물리 계산에 개입하지 않고, 검증을 통과한 **표준 50 Hz 리플레이 JSON(`dm100_replay_3d_v1`, 오른손 ENU 미터 + 단위 쿼터니언 `wxyz`)**만 불러와 오프라인 시네마틱 영상을 생성합니다.

---

## 2. Stage 1 (완료): 2D DM100 SpaceX RTLS

- **인터페이스 분리 (`rtls/interfaces.py`, `rtls/forces.py`)**: `TelemetryPacket → ActuatorCommand` 경계로 제어기와 물리엔진을 격리하고, 순수 외력 모듈을 분리해 추후 MuJoCo의 `xfrc_applied` 콜백으로 그대로 이식할 수 있습니다.
- **SpaceX형 7단계 비행 시퀀스 (`rtls/guidance.py`)**:
  1. `ascent`: 37 ± 1 km 콜드 분리 (최대 동압 ≤ 30 kPa, Q-alpha ≤ 150 kPa·deg 준수)
  2. `flip`: 분리 직후 저동압 RCS 신속 자세 반전
  3. `boostback`: 3엔진 유한 연소 예측-보정으로 발사장 활강 진입창으로 궤적 전환
  4. `coast`: 탄도 귀환 및 36 km 이하 격자핀 전개
  5. `entry_burn`: 무추력 예측 동압이 안전 여유를 넘을 구간에서만 짧게 역추진 후 MECO-2 완전 소화
  6. `aero_glide`: **주추진제 0 kg 무동력 격자핀 활강**(9.6–14.3 km), 인과적 온라인 항력 편차 추정 기반 IIP 유도
  7. `landing_burn` → `touchdown_settle`: 3→1엔진 최소 스로틀(44%) 호버슬램(해석적 자세 복귀 예측기 `_predict_zero_tilt_state` + 볼록 LP 참조 + 우선순위 제어 할당 `그리드핀 → TVC → RCS`) 및 **접지 후 5초 무동력 착륙다리(스프링-댐퍼 + 쿨롱 마찰) 직립 안정성** 판정.
- **극한 전수 검증 (`scripts/evaluate_validation.py`)**: 100% 끝단 초극한 악조건을 포함한 극한 엣지 32개 + 검증 연구 169개 + 회귀 286개 = **총 487케이스** 검증.
- **오프라인 로컬 브라우저 UI (`local_app.py`, `ui/index.html`)**: 2D 실시간 Canvas 비행 리플레이, 활강·제어 그래프, 단계별 연료 분석표, 50 Hz 3D 리플레이 JSON 내보내기, 선택적 MP4 렌더링 지원.

---

## 3. Stage 2 & Stage 3 계획 (별도 신규 3D 저장소에서 착수)

본 저장소(`Powered-Landing-Guidance`)는 2D DM100 플랫폼을 완성·유지하며, Stage 2(3D MuJoCo 6-DOF)와 Stage 3(Unreal Engine 5, 로컬 `D:` 드라이브)는 별도의 **신규 3D 저장소**에서 시작합니다.

| 단계 | 핵심 구성 | 2D → 3D 매핑 |
|---|---|---|
| **Stage 2: 3D MuJoCo 6-DOF** (신규 3D 저장소) | MJCF 1단 강체 + 독립 그리드핀 4개 + 착륙다리 4개 스트럿, `float64` RK4, `xfrc_applied` 외력 콜백 | `theta → quaternion`, 1D 핀 → 4핀(롤·피치·요), 1축 짐벌 → 2축 짐벌, 2D LP → 3D SOCP |
| **Stage 3: UE5 시네마틱 영상** (신규 3D 저장소) | 50 Hz 리플레이 JSON → UE5 Python Level Sequence 자동 임포트 (`ENU m → UE5 cm`), Niagara 플룸·RCS·재진입 효과, Movie Render Queue 렌더 | 시뮬레이션과 완전 분리된 오프라인 리플레이 파이프라인 |
