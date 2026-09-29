# 2차원 로켓 자동 회수

[English](README.md)

**발사 질량 3,000 kg 로켓**의 상승, 10 km 분리, PID 기반 1단 귀환·착륙을
구현한 2차원 시뮬레이터입니다. Aero-PID Return(APR)은 질량이 변하는 강체
운동, 공력 제어핀, 착륙 다리의 실제 접촉을 함께 계산합니다.

현재 남겨둔 구현은 3톤 APR 모델입니다. **2단 비행과 가정한 700 km 궤도
진입은 계산하거나 검증하지 않습니다.** 소프트웨어의 잠정 모델이며 실제
기체의 제작·비행 안전성을 인증하는 결과물이 아닙니다.

## 기체와 임무

| 항목 | 현재 제원 |
|---|---|
| 총 발사 질량 | 3,000 kg |
| 1단 건조질량 | 520 kg, 제어핀 24 kg 포함 |
| 주 추진제 / RCS 추진제 | 1,450 kg / 15 kg |
| 2단·탑재체 결합 질량 | 1,015 kg, 탑재체 30 kg 포함 |
| 주엔진 | 해수면 추력 42 kN, 공칭 비추력 260 s |
| 초기 추력 대 중량비 | 약 1.43 |
| 스로틀 / 짐벌 | 연속 0-100% / ±6° |
| 분리 조건 | 10,000 ±100 m, 목표 기울기 9°, 관측된 최대 동압 통과 후 |
| 회수 목표 | 원래 발사 위치 |

제어기는 귀환 제동·착륙 점화를 함께 계획하고, 수평 정렬과 하강을
동기화합니다. 최종 하강 목표는 0.9 m/s이며, 착륙 속도뿐 아니라 위치,
수평속도, 자세, 각속도와 접촉 하중 조건도 통과해야 합니다.

## 설치

Python 3.12.x가 필요하며 프로젝트의 목표 버전은 3.12.7입니다.
구현 검사는 기존 대체 실행 환경인 3.12.14에서도 수행합니다.

~~~powershell
python -m pip install -e ".[dev,video]"
~~~

필수 라이브러리는 NumPy, SciPy, PyYAML, Matplotlib입니다.
`video` 선택 항목은 MP4용 FFmpeg를, `dev`는 pytest와 Ruff를 설치합니다.
현재 모델에는 Gymnasium, CasADi, 강화학습 라이브러리가 필요하지 않습니다.
이미 설치된 실행 환경은 그대로 사용할 수 있습니다.

## 비행 실행

프로젝트 루트에서 실행합니다. 출력 폴더는 매번 새 경로를 사용해야 합니다.

~~~powershell
# 비행 없이 입력 허용 여부만 확인
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --check-inputs

# 무풍·표준 대기·기체 편차 없음
python -m scripts.run_nominal_flight --output-dir artifacts/apr-calm

# 고도별 바람·돌풍·정상적인 기체 및 센서 편차 반영
python -m scripts.run_nominal_flight --conditions configs/apr-operating-v1.yaml --output-dir artifacts/apr-weather

# 저장된 1단 분리 상태부터 귀환·착륙 실행
python -m scripts.run_nominal_flight --separation-record artifacts/apr-calm/result.json --output-dir artifacts/apr-return
~~~

실행 결과는 `result.json`과 `trajectory.png`로 저장됩니다. 상태 이력,
사건·성공 판정, 분리된 연료 잔량, 소스·입력 해시를 기록하며 실패 결과를
성공으로 바꾸지 않습니다. 실패한 비행은 0이 아닌 종료 코드를 반환합니다.

분리 이후 실행에서는 제공된 위치, 속도, 연료, 실제 스로틀을 유지하지만
제어기 기억 상태는 새로 시작합니다. 이미 수행된 상승이나 바뀐 날씨에서의
분리 가능성까지 다시 검증한 것으로 취급하지 않습니다.

## 저장된 비행 애니메이션

저장된 상태를 재생하므로 **비행 실험을 다시 실행하지 않습니다.**
시간을 늘리지 않은 정상 배속으로 고도별 배경, 핀·다리, 추진 불꽃을
표시합니다. 착륙 접촉은 로켓 중심이 아니라 다리의 발을 기준으로 합니다.

~~~powershell
python -m powered_landing_guidance.visualization --replay artifacts/apr-calm/result.json --animation artifacts/apr-calm/flight.mp4 --preview artifacts/apr-calm/preview.png --fps 20
~~~

기존 MP4나 미리보기 파일은 덮어쓰지 않습니다.

## 입력 조건·성공 기준·검증 범위

기준 시드는 **378**, 기체 제원은 고정입니다. YAML 입력 검사에서 알려진
고장, 날씨·건전성 검토 미통과, 잠정 정상 운용 범위 밖의 입력을 거부합니다.
이 범위는 프로젝트 가정이며 **공인된 발사 허용 기준은 아닙니다.**
예제 YAML의 입력 허용은 해당 날씨에서 착륙에 성공했다는 뜻이 아닙니다.

접지 때 질량중심 위치 오차 ≤2 m, 발의 수직속력 ≤1 m/s,
수평속력 ≤0.5 m/s, 기울기 ≤3°, 각속도 ≤0.5°/s가 필요합니다.
다리가 잠기고 발이 패드 안에 있어야 합니다. 주 추진제는 0 kg 초과,
RCS 추진제는 0 kg 이상이며 별도의 잔여 연료 여유량은 요구하지 않습니다.
엔진 정지 후 서 있는 상태, 접촉·하중 한계, 해당 비행의 상승·회수 조건도
모두 만족해야 성공입니다.

보관한 비행 근거는 무풍 전체 비행 개발 사례입니다. 운용 범위 전체의
성공이나 독립적인 성공률은 **아직 확정하지 않았습니다.**
공급·가압, 연료 출렁임 연성, 점화·재점화, 탄성 구조, 실제 분리 충격,
2단 점화·플룸, 세부 열·지면 손상, 다리 재료·패드 강도 해석은 APR에
연결되지 않았습니다. 자세한 내용은 [기술 명세](docs/spec.ko.md)에 있습니다.

## 프로젝트 구성과 검사

~~~text
configs/apr-operating-v1.yaml             허용 범위 안의 합성 입력 예제
docs/spec.md, docs/spec.ko.md             영문·한글 물리 및 실행 명세
scripts/run_nominal_flight.py            비행 실행 및 진단 그래프
src/powered_landing_guidance/
  apr_recovery.py                        현재 기체·환경·PID 제어
  recovery_physics.py                    공용 물리식 및 다리 접촉
  integrators.py                        고정 시간 간격 적분
  visualization.py                      저장 결과 MP4 재생
  __init__.py                           패키지 공개 진입점
tests/test_recovery_physics.py           물리·입력·사건·재생 검사
~~~

~~~powershell
python -m pytest -p no:cacheprovider
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
~~~

생성 결과는 Git에서 제외한 `artifacts/`에 둡니다.
개인 기록과 일정은 `.local/experiment_logs/`에만 보관하며 공개
포트폴리오 문서에 넣지 않습니다. 이전 기체 모델과 학습 파이프라인은
현재 구현에 포함하지 않습니다.
