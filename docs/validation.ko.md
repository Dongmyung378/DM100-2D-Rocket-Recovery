# DM100 검증 설계 및 재현

[English](validation.md) | [모델 명세](spec.ko.md) | [검증 결과](results.ko.md) | [README](../README.ko.md)

이 문서는 DM100 100톤 메탈록스 700 km 궤도 발사 및 1단 재사용 RTLS 시뮬레이터의 **487케이스 전수 검증 설계**와 재현 방법을 설명합니다.

---

## 1. 인과성 및 경계 격리 원칙

- 제어기(`RtlsController`)는 지연·잡음·편향·패킷 누락이 포함된 `TelemetryPacket`과 예보 대기/예보 바람만 받습니다.
- 숨겨진 실제 물리 외란(`density_scale`, `drag_coefficient_scale`, `thrust_scale`, `initial_mass_scale`, `inertia_scale`, 실제 돌풍, 실제 IMU 편향)은 `Plant2D` 내부에만 존재합니다.
- 모든 제어기 보정(저역통과 필터링된 항법 가속도 편향 `bias_x/z`, 활강 중 온라인 항력 배율 추정, 각가속도 외란 토크 편향 추정, 자이로 편향 추정)은 과거 관측만으로 수행하는 **인과적 추정**입니다.

---

## 2. 487케이스 평가 스위트 구성

| 평가 집단 | 케이스 수 | 입력 출처 및 목적 |
|---|---:|---|
| 극한 엣지 케이스 (`edge-cases`) | **32** | [evaluate_validation.py](../scripts/evaluate_validation.py) 내장: 최대 동/서풍, 극한 전단풍, 상승/재진입/활강/착륙 5 m/s 돌풍, 최대 동압·중량저추력 코너, 남/북극(자전 보너스 0)·적도 코리올리, 최대 항법·IMU 편향·패킷 누락(2/20), <strong>허용 상하한 100% 끝단을 동시 결합한 초극한 복합 악조건(<code>edge-23</code>~<code>edge-32</code>)</strong> |
| 개발 조건 (`development`) | **16** | `configs/validation-study-v1.json` 개발 분할 |
| 복합 홀드아웃 (`holdout`) | **96** | `configs/validation-study-v1.json` 복합 외란 홀드아웃 |
| 확장 센서 홀드아웃 (`sensor-holdout`) | **57** | `configs/validation-study-v1.json` IMU 잡음·편향·드롭아웃 홀드아웃 |
| 전체 운용 회귀 셋 (`regression-*`) | **286** | `configs/apr-regression-v1.json` (`development` 205 + `previous-holdout` 72 + `supplemental` 9) |
| **전체 합계** | **487** | **2단 700 km 궤도·비행 안전(동압·열유속·하중·각가속도)·최초 접지·추진제 보존·5초 무동력 직립 동시 판정 (16개 전 제약 ≤ 79.3% 인증 · 80% 여유 달성)** |

---

## 3. 재현 명령

```bash
# 단일 비행 실행 및 50 Hz 3D 리플레이 JSON(ENU + 단위 쿼터니언) 내보내기
python scripts/run_flight.py --scenario calm \
  --export-replay artifacts/calm_replay_50hz.json
python scripts/run_flight.py --scenario example-weather \
  --export-replay artifacts/weather_replay_50hz.json

# 극한 엣지 32케이스 병렬 검증
python scripts/evaluate_validation.py --suite edge-cases --workers 12

# 전체 487케이스 전수 병렬 검증
python scripts/evaluate_validation.py --suite full-487 --workers 14 \
  --out-json artifacts/validation-487.json

# 소프트웨어·물리·궤도·제어·로컬 UI 자동화 테스트 및 린트
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```

> **해석 한계**: 487케이스 전수 통과 및 전 제약 ≤ 80% 여유 달성은 선언한 2D 합성 운용 범위 내의 소프트웨어 검증 결과이며, 실물 로켓의 비행 안전성 인증을 의미하지 않습니다.
