# 100톤 2차원 로켓 자동 회수

[English](README.md) · [모델 명세](docs/spec.ko.md)

**발사·콜드 분리·원래 발사장으로의 1단 귀환과 착륙**을 계산하는 평면
시뮬레이터입니다. 가변 질량 강체 운동, 모델 기반 유도, 피드백 자세 제어,
엔진 짐벌·RCS·공력핀을 결합합니다.

현재 기체의 발사 질량은 **100,000 kg**, 1단 주 추진제는 **70,200 kg**,
RCS 추진제는 **300 kg**입니다. 상단에는 **500 kg 탑재체**를 포함하며
이론상 **700 km 원궤도**를 목표로 합니다. 2단 궤도 진입은 성공을 가정하며
비행을 계산하거나 검증하지 않습니다.

## 기체와 모델

| 항목 | 현재 제원 |
|---|---:|
| 총 발사 / 1단 초기 질량 | 100,000 / 82,500 kg |
| 1단 건조질량 | 12,000 kg, 핀과 착륙장치 포함 |
| 1단 주 추진제 / RCS 추진제 | 70,200 / 300 kg |
| 2단 건조질량 / 추진제 / 탑재체 | 1,500 / 15,500 / 500 kg |
| 1단 해면 / 진공 추력 | 1,600 / 약 1,760 kN |
| 1단 해면 / 진공 비추력 | 300 / 약 330초 |
| 초기 추력 대 중량비 | 약 1.63 |
| 스로틀 / 짐벌 | 이상적인 연속 0-100% / ±6° |
| 분리 허용 조건 | 37 ±1 km, 주 추진제 20,000 kg 이상 |
| 회수 목표 | 원래 발사장 |

19개 상태 모델은 고도별 중력·층별 대기·바람과 돌풍·공력·변하는 질량
특성·구동기 응답·지연 관측·액체 수집과 공급·폐쇄형 헬륨 가압·비선형
진자 출렁임·재점화 준비·분리 충격량을 포함합니다. 설치한 핀 4개는 평면
등가 면 2개로 계산합니다.

착륙 평가는 첫 발 접지에서 끝내며 접지 후 안정 기립은 계산하지 않습니다.
발 수직속도 ≤1.5 m/s이면 다리가 견딘다고 가정하며 접촉 하중·재료 강도는
평가하지 않습니다. 부품과 계수는 실측 보정한 하드웨어 자료가 아닌 이론적
설계 가정입니다. 입력·물리·한계는 [모델 명세](docs/spec.ko.md)에 있습니다.

## 설치와 단일 비행

Python 3.12.x를 사용하며 작업 환경은 Python 3.12.7입니다.

~~~powershell
python -m pip install -e ".[dev,video]"

# 비행 없이 운용 입력 검증
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --check-inputs

# 무풍·표준 대기·기체와 센서 편차 없음
python -m scripts.run_nominal_flight --output-dir artifacts/apr100t/nominal-current --progress

# 지정한 날씨와 정상 기체 편차를 반영한 전체 비행
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --output-dir artifacts/apr100t/configured-current --progress
~~~

비행마다 새 출력 폴더를 사용합니다. `result.json`과 `trajectory.png`에
실제 상태·사건·성공 판정·연료 수지·소스와 입력 해시를 저장합니다.
성공 기준을 통과하지 못하면 0이 아닌 종료 코드를 반환합니다.
기준 시드는 **378**이며 입력 허용이 비행 성공을 뜻하지는 않습니다.

## 검증 결과

현재 제원은 임무 완료·비행 안전·착륙·연료 수지를 함께 요구한 다음
고정 입력 전체 비행 검사를 통과했습니다.

| 시험군 | 통과 / 시험 |
|---|---:|
| 알려진 개발 조건 | 205 / 205 |
| 이미 관측한 평가 조건의 회귀 검사 | 72 / 72 |
| 실제 돌풍 노출·적분 간격 절반의 보조 검사 | 9 / 9 |

본 검사 277개와 보조 검사 9개는 별도 분모로 기록합니다. **전체 286개가
통과**했으며 통과를 위해 제어기나 성공 기준을 바꾸지 않았습니다.
이미 알려진 조건의 회귀 결과이지 새로운 독립 평가나 연속 입력 범위의
모든 조합에 대한 증명이 아닙니다.

| 전체 검사 286회의 항목 | 관측 최악값 | 기준 |
|---|---:|---:|
| 접지 질량중심 위치 오차 | 0.839 m | 2 m |
| 발의 수직 / 수평속도 | 1.2551 / 0.202 m/s | 1.5 / 0.5 m/s |
| 접지 자세 / 각속도 | 0.0212° / 0.0226°/s | 3° / 0.5°/s |
| 최대 동압 | 28.949 kPa | 30 kPa |
| 접지 전 최대 각가속도 | 3.780°/s² | 5°/s² |
| 최소 잔여 주 추진제 / RCS 추진제 | 2,958 / 95.875 kg | >0 / ≥0 kg |

최대값은 올림, 잔량은 내림으로 표시했습니다. 최대 RCS 사용량은
204.124806 kg이며 95.875194 kg, 초기 재고의 31.96%를 남겼습니다.
무풍 검사도 발 수직속도 약 1.005 m/s, 주 추진제 5,796 kg·RCS 262.546 kg을
남기며 통과했습니다. NumPy/SciPy float64 계산을 CPU 프로세스 8개로 실행했으며
GPU 가속은 구현하거나 속도를 측정하지 않았습니다.

현재 원본 결과는 로컬의 `artifacts/apr100t/rcs300-v1`에 보관합니다.
보고서·영상·개인 기록은 Git에서 제외하며 새로 복제한 저장소에 포함하지
않습니다. 저장소의 고정 회귀 입력 목록으로 과거 로컬 기록 없이 같은
선언 조건을 다시 실행할 수 있습니다.

## 회귀 검사 재실행

저장소의 `configs/apr-regression-v1.json`은 같은 286개 알려진 입력·관측 시드
방식·적분 간격을 선언합니다. 준비 단계는 비행 없이 입력을 검증하고 해시를
남깁니다. 실행은 모든 실패를 보존하며 기존 결과를 덮어쓰지 않습니다.

~~~powershell
python -m scripts.evaluate_regression_suite --prepare --output-dir artifacts/apr100t/regression-v1
python -m scripts.evaluate_regression_suite --run all --workers 8 --output-dir artifacts/apr100t/regression-v1
~~~

개인 근거 묶음 없이 실행할 수 있습니다. 다시 시험하려면 새 출력 폴더를
사용합니다. 전체 비행 검사는 입력 검증보다 오래 걸리며 `--workers`는
독립적인 CPU 비행 프로세스 수입니다. 알려진 조건을 다시 실행해도 새로운
독립 평가가 되지는 않습니다.

## 결과 보고서

보고서 도구는 완료한 현재 제원 결과를 읽으며 비행을 추가 실행하지 않습니다.
영문이 기본이며 `--language ko`는 한글, `--language both`는 서로 연결한
영문·한글 버전을 생성합니다.

~~~powershell
# 로컬에 완료한 현재 제원 결과가 있어야 함
python -m scripts.present_validation_results --language both

# 새로 복제한 저장소에서도 위 단일 비행·회귀 명령을 완료한 뒤 실행
python -m scripts.present_validation_results --campaign-dir artifacts/apr100t/regression-v1 --nominal-result artifacts/apr100t/nominal-current/result.json --out-dir artifacts/apr100t/presentation-current --language both
~~~

기본 출력 폴더는 `artifacts/apr100t/presentation-current`입니다.
`validation-report.html`·`validation-report.ko.html`, 요약·계산 그래프와
`presentation-data.json`을 저장합니다. 새로 복제한 저장소에는 실행 결과가
없으므로 먼저 비행을 실행하고 `--campaign-dir`·`--nominal-result`에 그 출력을
지정합니다. 새 보고서 폴더는 `--out-dir`로 지정합니다. 보고서와 그래프는
새로운 독립 평가나 비행 결과를 만들지 않습니다.

## 애니메이션

현재 제원으로 완료한 비행을 추가 시뮬레이션 없이 재생합니다.

~~~powershell
python -m powered_landing_guidance.visualization --replay artifacts/apr100t/nominal-current/result.json --animation artifacts/apr100t/nominal-current/flight.mp4 --preview artifacts/apr100t/nominal-current/preview.png --fps 24 --width 1920 --end-hold 4
~~~

고도별 배경·로켓·지상 시설·엔진과 RCS 불꽃·계산값 대시보드를 표시합니다.
Full HD·24 fps의 정상 배속입니다. 첫 접지 이후에는 불꽃을 숨기고 기록한
자세를 4초 유지합니다. 실제 엔진 종료나 안정 기립을 계산한 것이 아닌
화면 연출입니다. MP4 생성은 별도 명령이며 비행 실행만으로 영상이 생기지는
않습니다.

## 구성과 검사

~~~text
configs/                         운용 입력과 고정 회귀 입력 목록
docs/                            영문·한글 모델 명세
scripts/                         비행·회귀 검사·결과 보고 도구
src/powered_landing_guidance/     운동·제어·계통·시각화
tests/                           물리·보존 법칙·항법·평가 규약 검사
~~~

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

소프트웨어 검사와 유한한 합성 비행이 실물 기체의 안전 인증을 대신하지는 않습니다.
