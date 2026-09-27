# 2D 재사용 로켓 자동착륙 유도

[English README](README.md)

재사용 로켓의 평면상 최종 하강을 다루는 재현 가능한 시뮬레이션·제어 프로젝트입니다. 가변 질량 동역학, Gymnasium 환경, 고전제어 착륙기, 최적제어 궤적, 학습 정책, 동일 조건의 강건성 평가를 포함합니다.

이 프로젝트의 범위는 **착륙**입니다. 발사부터 착륙까지의 전체 비행이나 3D 비행을 구현한 것은 아닙니다. 시뮬레이션 성공은 실제 하드웨어 안전성을 입증하지 않습니다.

## 시스템 개요

| 구성 요소 | 구현 내용 |
|---|---|
| 상태 | `x, z, vx, vz, theta, omega, mass` |
| 제어 | 스로틀과 짐벌 각도 |
| 물리 모델 | 평면 강체, 가변 질량, 0.02초 간격 RK4 적분 |
| 제어 주기 | 0.1초 |
| 기준선 | Suicide-burn, 수직 PID, 통합 착륙 PID |
| 학습 제어기 | 행동 복제, DAgger, PID 잔차 SAC |
| 실행 환경 | Python 3.12.7 |

환경에는 바람과 돌풍, 추력·관성 오차, 센서 잡음, 관측·명령 지연, 엔진 응답 지연을 줄 수 있습니다. 안전 착륙과 실패는 같은 착륙 기준으로 판정합니다.

## 현재 검증 결과

현재 권장 제어기는 통합 PID입니다. 오프라인 최적제어 데이터셋은 채택된 학습 궤적 1,007개, 동일 분포 검증 궤적 100개, 채택된 어려운 분포 밖 테스트 궤적 191개로 구성됩니다. **Teacher 재시뮬레이션을 통과한** 어려운 테스트에서 PID는 191/191, 행동 복제는 0/191, DAgger는 3/191 착륙했습니다. 계획된 테스트 궤적 중 9개는 Teacher 재시뮬레이션에서 탈락했으므로, 이 결과를 모든 어려운 초기 조건에서의 PID 성공으로 일반화할 수는 없습니다. 집합 정의와 한계는 [데이터셋 카드](docs/dataset-card.ko.md)와 [모델 카드](docs/model-card.ko.md)에 정리했습니다.

별도 외란 평가 규약은 기준 시드 `378`과 분리된 학습·개발·최종 보류 집합을 사용합니다. 조건별 개발 집합 12개에서 PID의 안전 착륙 수는 다음과 같습니다.

| 정상 조건 | 초기 상태 분포 밖 | 바람 | 모델 오차 | 센서·지연 | 복합 외란 |
|---:|---:|---:|---:|---:|---:|
| 12/12 | 12/12 | 6/12 | 8/12 | 6/12 | 0/12 |

잔차 SAC가 복합 외란에서 한 사례를 착륙시킨 적은 있지만, 시험한 정책들은 다른 조건에서 기존 PID의 성공 사례를 잃었습니다. 조건별 성능 비퇴보 기준을 통과한 학습 제어기는 없으므로 PID를 교체하지 않았습니다. 개발 집합의 소수 사례만 본 결과이며 최종 강건성 추정치는 아닙니다. 최종 보류 집합은 아직 열지 않았습니다. 정확한 조건, 평가 규약, 실패 분석은 [기술 명세](docs/spec.ko.md)를 참조하세요.

## 착륙 실행

저장소 루트에서 Python 3.12.7을 사용합니다.

```powershell
python -m pip install -e ".[dev,rl,video]"
python scripts/run_episode.py --seed 378 --initial-state 10 100 0 -20 0 0 1000 --controller integrated-pid --output artifacts/landing.json --plot artifacts/landing.png --animation artifacts/landing.mp4
```

초기 상태의 순서는 `x z vx vz theta omega mass`이며 SI 단위를 사용합니다. 각도는 라디안입니다. 실행 결과로 착륙 판정과 최종 상태를 출력하고, 재생 가능한 기록·진단 그래프·MP4 애니메이션을 저장합니다. 기존 출력 파일은 덮어쓰지 않습니다. 저장된 기록만 다시 애니메이션으로 만들려면 다음 명령을 사용합니다.

```powershell
python scripts/run_episode.py --replay artifacts/landing.json --animation artifacts/landing-replay.mp4
```

SAC 또는 MP4가 필요 없다면 `.[dev]`만 설치하고 `--animation` 옵션을 제외해도 됩니다.

## 검증 재현

```powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python scripts/prepare_rl_protocol.py --output-dir artifacts/rl-protocol
```

마지막 명령은 개발 사례에 대해서만 고정된 PID를 평가합니다. Git에서 제외한 `artifacts/` 아래 CSV와 규약 보고서를 작성하며 기존 파일은 덮어쓰지 않습니다. 조건 범위·기준 시드·정책 선택 기준은 [`configs/rl-evaluation-v1.yaml`](configs/rl-evaluation-v1.yaml)에 있습니다. 학습·평가 실행 코드는 [`scripts/`](scripts/)에, 구현 코드는 [`src/powered_landing_guidance/`](src/powered_landing_guidance/)에 있습니다.

## 범위와 한계

- 최적제어 Teacher는 명목 모델을 풉니다. 복합 외란이나 구동기 지연 아래의 성능을 보증하지 않습니다.
- 현재 잔차 명령 필터는 경험적 제한 장치이며 형식적인 안전 보증이 아닙니다. 필터를 거친 정책도 PID가 성공하는 사례를 실패하게 만들 수 있습니다.
- 시뮬레이터는 발사, 3D 동역학, 구조 충격 손상, 비행 인증 수준의 하드웨어 동작을 모델링하지 않습니다.
- 생성 결과와 로컬 전용 기록은 Git에서 제외합니다.

자세한 내용: [기술 명세](docs/spec.ko.md) · [데이터셋 카드](docs/dataset-card.ko.md) · [모델 카드](docs/model-card.ko.md).
