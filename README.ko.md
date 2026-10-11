# DM100 - 2차원 로켓 발사와 회수

[English](README.md) | [검증 결과](docs/results.ko.md) | [모델 명세](docs/spec.ko.md)

총질량 100톤 로켓의 발사, 약 37 km에서의 단 분리, 1단의 발사장 귀환을
계산하는 **2차원 개념 시뮬레이터**입니다. 직접 구현한 19개 상태 변수의
강체 운동과 추진 시스템에 모델 기반 귀환 유도, PID 계열 자세 제어,
인과적 상태 추정과 최초 접지 판정을 연결했습니다. 로컬 실험 화면은
영어를 기본으로 제공하며 한국어로 전환할 수 있습니다.

![로컬 실험 화면](docs/assets/ui.png)

## 구현한 기능

- 발사, 콜드 단 분리, 귀환 연소, 대기권 재진입과 최초 접지 계산
- 질량과 관성 변화, 엔진 짐벌, RCS와 핀, 단순화한 추진제 공급과 가압,
  액체 추진제 흔들림과 분리 충격량 모델
- 잡음, 편향, 지연과 예정된 센서 패킷 누락을 고려한 항법 및 자세 추정
- 평가 입력과 소스 동결, 동일 조건의 기능 제거 실험, 전체 비행의 적분
  간격 비교, 매개변수 민감도와 독립 적분기 비교
- 비행별 공개 기록, 표준 라이브러리 기반 결과 검증기, 측정값 그래프와
  저장한 궤적을 사용하는 선택적 MP4 재생

Python 3.12, NumPy/SciPy, FastAPI/Uvicorn, HTML/CSS/JavaScript와
Matplotlib/FFmpeg를 사용했습니다. 명시적인 모델로 유도와 상태 추정을
수행하며, **머신러닝 정책을 학습하는 프로젝트는 아닙니다.**

## 측정 결과

| 평가 집단 | 통과 / 실행 |
|---|---:|
| 기존 조건 회귀 검증 | 286 / 286 |
| 새로운 복합 조건 | 96 / 96 |
| 센서 오차와 자세 추정기 | 48 / 48 |
| 동일 센서 조건과 기존 처리 | 48 / 48 |

두 방식 모두 기준을 충족한 동일 조건 48쌍에서 RCS 소모량 중앙값은 120.977 kg에서 99.122 kg으로 18.07% 감소했습니다.

기존 286개 조건은 회귀 검증입니다. 새로운 복합 조건 96개와 센서 확장
조건 48개는 구현을 동결한 뒤 평가했습니다. 각 집단의 분모를 구분하며,
센서 처리 방식은 입력과 시드가 같은 조건끼리 비교합니다. 센서 오차의
범위는 실제 부품에서 측정한 사양이 아닌 합성 모델의 가정입니다.

[영어 결과 보고서](docs/results.md)와 [한국어 결과 보고서](docs/results.ko.md)에
접지 수치, 유한 표본의 구간, 기능 제거 실험, 수치 수렴과 실패 결과를
정리했습니다. 임무 완료, 비행 안전, 최초 접지와 추진제 보존 검사를
모두 충족해야 성공으로 판정합니다.

![기록된 검증 결과](docs/assets/validation.png)

의존성 설치나 비행 재실행 없이 공개 기록을 확인할 수 있습니다.

```bash
python scripts/validate_evidence.py docs/results/validation-study-v1
python -m scripts.validate_regression_evidence docs/results/regression-v1.1
python -m scripts.check_source_binding docs/results/validation-study-v1/protocol.json docs/results/regression-v1.1/protocol.json docs/results/numerical-reference.json
```

두 명령은 입력과 소스, 저장된 결과의 일관성을 검사하고 요약 수치를
다시 계산합니다. 실제 물리 데이터의 진위를 인증하는 검사는 아닙니다.

## 로컬 실행

같은 컴퓨터에서 **Windows의 Python 3.12.7 새 설치**와 **Linux 컨테이너의
Python 3.12.15 새 설치**를 검증했습니다. 각각 소프트웨어 검사 787개와
하위 검사 11개를 통과했고, 과거 로컬 기록을 필요로 하는 검사 1개는
건너뛰었습니다. 별도 환경에서의 재현이며 독립적인 제삼자 심사는 아닙니다.
버전과 대표 비행의 비교는 [reproduction.json](docs/results/reproduction.json)에 있습니다.

Windows에서 Python 3.12를 설치하고 소스 ZIP을 풀거나 저장소를 복제한 뒤
[launch-dm100.bat](launch-dm100.bat)를 더블클릭합니다. 실행기는 검증한
의존성 제약 조건으로 `.venv`를 준비합니다. 최초 설치에는 인터넷 연결이
필요하며, 정상 환경은 재설치 없이 사용합니다. 브라우저는 `127.0.0.1`에서
열립니다. 터미널을 유지하고 종료할 때 Ctrl+C를 누릅니다.

```powershell
# 앱을 열지 않고 실행 환경만 준비합니다.
.\launch-dm100.bat --setup-only
```

실행기는 기존 Python 3.12 설치를 사용합니다. 직접 지정한 `DM100_PYTHON`,
정상 로컬 환경, `py -3.12`, `PATH`의 Python과 저장소 내부 Python을 확인합니다.
명시한 인터프리터가 잘못되면 오류를 안내합니다. 폴더를 이동해 기존 환경을
사용할 수 없으면 `.local/environment-backups/`에 보관한 뒤 현재 경로로
재구성합니다.

저장소 루트에서 수동으로 설치할 수도 있습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements/constraints.txt -e ".[ui,video]"
.\.venv\Scripts\python.exe -m powered_landing_guidance.local_app
```

Linux에서는 Python 3.12 가상환경을 만들고 다음 명령을 실행합니다.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e ".[ui,video]"
python -m powered_landing_guidance.local_app
```

Linux에서 한국어 영상 자막을 표시하려면 Noto Sans CJK 등의 한글 글꼴이
필요합니다. 최소 검증 컨테이너에는 이 글꼴이 없어 누락 경고가 발생했으나
소프트웨어 검사와 수치 계산은 통과했습니다. 공개 검증 그래프는 영어
표기를 사용합니다. 독립 실행 파일이 아닌 Python 소스 앱이므로 다른
컴퓨터의 `.venv`를 복사해 배포하지 않습니다.

계산, 결과 저장과 영상 생성은 로컬에서 수행합니다. 실행 중 외부 API,
계정, 분석 도구나 CDN을 사용하지 않습니다.

1. 무풍 기본값을 유지하거나 허용된 날씨와 정상 기체 편차를 입력합니다.
2. 비행을 실행하고 임무, 안전, 접지 결과와 실패 항목을 확인합니다.
3. 성공한 비행은 저장한 궤적으로 MP4 영상을 생성할 수 있습니다.

입력이 유효하더라도 비행 성공을 보장하지 않습니다. 실행 오류와 정상적으로
완료된 실패 비행을 구분하며, 입력 화면을 바꿔도 저장된 입력은 유지됩니다.
로컬 화면에는 기존 운용 입력을 제공하고, 확장 센서와 기능 제거 설정은
실험 명세 또는 Python API에서 지정합니다.

전체 MP4 생성은 CPU 작업량이 커 비행 계산보다 오래 걸릴 수 있습니다.
빠르게 확인하려면 짧은 구간을 선택합니다. 영상 생성은 계산 결과를 변경하거나
비행을 재실행하지 않습니다. [포함된 영상](docs/assets/flight.mp4)은 짧은 단 분리 시연입니다.

## 기체와 판정 기준

| 항목 | 모델 값 |
|---|---:|
| 발사 질량 / 1단 초기 질량 | 100,000 / 82,500 kg |
| 1단 건조 / 주 추진제 / RCS 질량 | 12,000 / 70,200 / 300 kg |
| 2단 건조 / 추진제 / 탑재체 질량 | 1,500 / 15,500 / 500 kg |
| 해면 / 진공 추력 | 1,600 / 약 1,760 kN |
| 해면 / 진공 비추력 | 300 / 약 330 s |
| 추력 조절 / 짐벌 | 이상적 연속 0~100% / ±6도 |
| 콜드 분리 조건 | 37 ±1 km, 주 추진제 20,000 kg 이상 |
| 회수 목표 | 원래 발사장 |

착륙은 **다리 끝점이 처음 지면에 닿는 순간** 평가합니다. 질량중심 위치
오차는 2 m 이하, 다리 끝점 수직과 수평 속도는 각각 1.5와 0.5 m/s 이하,
자세는 3도 이하, 각속도는 0.5도/s 이하이며 주 추진제는 양수, RCS 추진제는
음수가 아니어야 합니다. 다리 전개, 발사대 내부 접지와 기체 지면 여유도
충족해야 합니다. 수직 속도 기준 안에서 다리가 충분하다고 가정하며 접촉
강도와 착륙 후 직립 안정성은 평가하지 않습니다.

[모델 명세](docs/spec.ko.md)는 좌표계, 상태 변수, 운용 범위와 보존 및
제한 조건을 설명합니다. [검증 설계](docs/validation.ko.md)는 확장 센서,
추정기 가정과 실험 구성을 설명합니다.

## 비행 및 전체 실험 재현

새 출력 파일이나 폴더를 사용합니다. 기존 기록은 덮어쓰지 않습니다.

```bash
# 원래 입력과 시드로 센서 확장 개발 조건 하나를 재현합니다.
python -m scripts.reproduce_case --job-id development/development-000/full/dt-0.01 --output artifacts/reproduced-case.json

# SciPy DOP853과 결합 물리 모델의 짧고 매끄러운 세 구간을 비교합니다.
python -m scripts.check_numerical_reference --output artifacts/numerical-reference.json

# 재현한 비행을 공개 기록과 비교합니다.
python -m scripts.verify_reproduction --evidence-dir docs/results/validation-study-v1 artifacts/reproduced-case.json

# 기존 286개 고정 조건의 회귀 검증을 실행합니다.
python -m scripts.evaluate_regression_suite --prepare --output-dir artifacts/regression
python -m scripts.evaluate_regression_suite --run all --workers 4 --output-dir artifacts/regression
```

새 실험은 개발, 센서 처리 비교, 기능 제거, 수렴과 민감도를 포함한
**305회 비행**입니다. [전체 실행 명령](docs/validation.ko.md#재현)으로
입력 준비, 개발 검증, 소스 동결, 각 단계의 실행과 공개 기록 추출을 진행합니다.
기존 회귀 검증까지 합하면 목적이 다른 591회 실행이며, 독립 최종 평가
조건이 591개라는 의미는 아닙니다.

CPU의 float64 연산을 사용하며 GPU 구현은 제공하지 않습니다. 전체 실험에는
상당한 시간이 필요합니다. 여러 프로세스를 사용하면 `OPENBLAS_NUM_THREADS=1`과
`OMP_NUM_THREADS=1`을 지정하고 컴퓨터에 맞는 작업자 수를 선택합니다.

## 소프트웨어 검사

```bash
python -m pip install -c requirements/constraints.txt -e ".[ui,video,dev]"
python -m pytest -q -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
```

물리 모델, 보존 법칙, 유도, 추정기의 인과성, 성공 판정, 기록 변조와
로컬 API 및 화면을 검사합니다. 전체 비행 평가와는 별개입니다.
[Windows와 Linux CI](.github/workflows/checks.yml)는 소프트웨어, 공개 기록과
패키지를 검사하며 모든 비행을 자동으로 반복하지 않습니다.

## 저장소 구성

```text
configs/                         운용, 회귀 검증과 새 실험의 고정 입력
docs/spec.md, spec.ko.md          영어와 한국어 물리 모델 명세
docs/validation.md, validation.ko.md  검증 설계와 재현 명령
docs/results.md, results.ko.md    기록에서 생성한 수치 보고서
docs/results/                    비행별 공개 기록과 환경 재현 결과
docs/assets/                     화면, 궤적, 검증 그래프와 시연 자료
requirements/constraints.txt     검증한 의존성 제약 조건
scripts/                         실행, 재현, 검증과 보고서 생성 명령
src/powered_landing_guidance/     물리 모델, 유도, 추정과 로컬 API 및 화면
tests/                           소프트웨어와 기록 일관성 검사
launch-dm100.bat                  Windows 환경 준비와 앱 실행기
```

`artifacts/`의 원본 실행 폴더, 환경, 캐시와 개인 작업 기록은 Git에서
제외합니다. 공개 설명 문서와 선별한 검증 기록은 저장소에 포함합니다.

## 적용 범위

이론적 소프트웨어 개념 모델로 수평과 수직 이동, 평면 내 회전을 다룹니다.
계수는 실물 측정으로 보정하지 않았습니다. 2단의 500 kg 탑재체와 700 km
원궤도 진입은 가정이며 계산하지 않습니다. 유연 구조, 상세 유체와 열 및
플룸, 다리와 발사대 재료 강도, 3차원 비행과 실제 비행 인증은 범위 밖입니다.
영상의 2단 이동과 접지 후 고정 자세는 시각적 연출입니다. 유한한 합성
조건의 성공은 실제 기체의 안전성을 입증하지 않습니다.
