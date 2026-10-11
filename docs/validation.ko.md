# 검증과 실험 설계

[English](validation.md) | [모델 명세](spec.ko.md) | [README](../README.ko.md)

검증 자료는 소프트웨어의 재현성, 수치 계산의 정확성, 선언한 합성 조건에서의
성능, 물리 모델의 타당성을 구분합니다. 한 종류의 검증을 통과했다고 해서
다른 종류의 정확성까지 입증한 것은 아닙니다.

## 평가 집단

`configs/apr-regression-v1.json`의 기존 286개 입력은 그대로 보존했습니다.
이 자료는 이미 관측한 조건에서 수정 후 성능이 유지되는지 확인하는 데 사용합니다.
새 연구의 입력은 `configs/validation-study-v1.json`으로 별도 관리합니다.

| 평가 집단 | 선언한 조건 수 | 목적 |
|---|---:|---|
| 개발용 | 16개, 제어기 두 종류 | 구현의 문제를 분석하고 선택 |
| 새로운 조건 평가 | 96개 | 기존 관측 가정에서 새로운 복합 입력 평가 |
| 확장 센서 평가 | 48개, 제어기 두 종류 | 추가한 각도 계측 오차에서 같은 입력으로 비교 |
| 기능 제거 비교 | 개발용 12개, 제어기 다섯 종류 | 개별 기법의 기여도 확인 |
| 적분 간격 비교 | 4개, 적분 간격 세 종류 | 제어 주기를 유지한 전체 비행 계산 비교 |
| 민감도 분석 | 명목 조건과 개별 변수 변경 8개 | 추력과 항력, 관성, 초기 질량의 영향 확인 |

개발용 결과는 구현 변경에 활용할 수 있습니다. 두 최종 평가 집단의 실행 전에는
`freeze.json`에 소스와 입력 목록, 의존성 환경, 완료한 개발용 결과를 연결합니다.
최종 평가 결과를 보고 동결한 버전을 다시 조정하지 않습니다. 이후 버전을
개선한다면 새로운 평가 집단이 필요합니다.

실행 오류를 포함해 시도한 모든 비행을 분모에 남깁니다. 임무 완료와 비행 안전,
첫 접지 기준, 추진제 보존을 모두 통과해야 성공입니다. 기존 물리 판정 기준은
완화하지 않습니다.

## 새로운 조건의 생성

기존 세 지점의 바람 프로파일을 반복하는 대신, 고도 0, 4, 10, 22, 100 km의
지점에서 바람을 구성했습니다. 지상풍과 상층풍의 추출 범위는 각각 6 m/s와
18 m/s 이내입니다. 중간 지점은 선언한 보간식과 균등 분포 편차로 구성하며,
기본풍 예보 오차는 1.8 m/s 이내입니다. 수직풍은 2.5 m/s 이내, 돌풍은
5 m/s 이내, 밀도 배율은 0.97~1.03이며, 기존 정상 기체 편차도 함께 추출합니다.
돌풍 시간창은 상승 구간과 고고도 귀환, 늦은 하강 구간을 대상으로 미리 정합니다.
설정한 돌풍의 진폭만으로 실제 비행 중 돌풍을 만났다고 판단하지 않습니다.

생성기와 정확한 입력값, 난수 시드, 제어기 종류를 공개합니다. 실행 전에는
기존 입력 허용 모델을 통과해야 합니다. 이 분포는 전체 운용 범위의 일부를
다루며, 모든 연속 입력 조합이나 실제 비행의 성공을 입증하지 않습니다.

Wilson 방식의 95% 구간은 선언한 합성 조건 추출 방식에서의 경험적 성공률을
설명합니다. 실제 로켓의 신뢰도 구간으로 해석할 수 없습니다. 지표에는 해당
값을 계산할 수 있었던 모든 완료 비행을 포함하고, 실패 비행도 제외하지 않습니다.
접지 지표에는 접지까지 도달한 비행의 측정 수를 따로 표시합니다.

## 추가한 계측 모델

위치와 속도의 잡음, 관측 지연, 인과적인 항법 추정기는 기존에 구현되어 있었습니다.
이번 실험에서 추가한 합성 계측 가정은 다음과 같습니다.

| 항목 | 확장 센서 평가에서 추출하는 범위 |
|---|---:|
| 자세의 균등 잡음 반경 | 0.01~0.05도 |
| 자이로의 균등 잡음 반경 | 0.005~0.03도/초 |
| 일정한 자세 편향 | -0.02~+0.02도 |
| 일정한 자이로 편향 | -0.03~+0.03도/초 |
| 주기적인 패킷 누락 | 제어 표본 60~120개마다 1~2개 |

이 수치는 특정 센서의 사양이나 실측 보정 결과가 아닌 명시적인 모델 가정입니다.
누락 시에는 마지막 패킷과 원래 측정 시각을 유지하며, 같은 패킷을 다시 관측한
것으로 동화하지 않습니다. 각도 계측용 난수 흐름을 분리해 기존 위치와 속도의
잡음 흐름이 바뀌지 않도록 했습니다. 추진제와 구동 장치의 계측값은 여전히
이상적인 지연 계측값입니다.

자세 추정기는 각도와 각속도, 자이로 편향을 상태로 사용합니다. 각도 차이는
회전의 주기성을 반영하고, 공분산에는 Joseph 갱신식을 적용합니다. 설정한 실제
센서 편향은 추정기에 전달하지 않습니다. 자이로 편향을 식별하려면 독립적인
자세 관측이 필요하며, 절대 자세 편향은 남아 있는 한계입니다.

`raw-attitude` 비교에서는 같은 센서 설정과 선언한 난수 스트림을 사용하면서
자세 필터 없이 기존 제어 방식에 관측을 전달합니다. 제어 결과에 따라 궤적과
관측 시각은 서로 달라질 수 있습니다. 이전 계측 처리 방식을 새 센서 가정에
그대로 적용했을 때의 결과를
보여주는 비교입니다. 기존 286개 조건에 이러한 센서 오차가 있었다는 뜻은 아닙니다.

## 개별 기능의 비교

`ControllerOptions`로 병진 가속도 편향 보정과 토크 편향 보정, 귀환 제동의
재계획, 자세 필터를 하나씩 제거합니다. 물리 모델과 입력, 난수 흐름, 작동 제한,
판정 기준은 동일하게 유지합니다. 같은 조건에서 성공 여부와 지표 차이를 비교해
효과와 손해를 함께 기록합니다. 개발용 12개 조건은 원인 분석을 위한 표본이며,
독립적인 최종 평가 집단으로 표현하지 않습니다.

## 수치 계산 검증

전체 비행에서 적분 간격을 0.01, 0.005, 0.0025초로 바꾸되, 선언한 센서 시드와
0.02초의 제어 주기는 유지합니다. 접지와 추진제 소진에는 사건 검출을 적용합니다.
단 분리 시에는 검출된 사건 시각으로 제어 시계를 재설정하므로, 작은 사건 시각
차이가 이후 관측 시각을 이동시킬 수 있습니다. 가장 작은 적분 간격의 결과는
비교 기준이며 정확해는 아닙니다. 허용오차는 실행 전에 입력 목록에 선언하고,
초과한 결과도 그대로 기록합니다.

`scripts/check_numerical_reference.py`는 세 가지 매끄러운 구간에서 제어 입력을
1초 동안 고정하고 RK4와 SciPy DOP853을 별도로 비교합니다. 이 구간에서는 핀을
접어 전개 스위치와 포화 경계가 섞이지 않도록 합니다. 이러한 사건은 전체 비행
실험에서 다룹니다. 적분기의 일치는 수치 구현의 일관성을 확인하는 근거이며,
실제 물리와 모델의 일치를 입증하는 자료는 아닙니다.

기존 해석적 자유낙하와 질량 보존, 분리 운동량, 접지 사건 테스트도 유지합니다.
민감도 분석에서는 한 번에 하나의 매개변수를 바꿔 착륙과 추진제 지표의 변화를
기록합니다. 기체 계수와 센서 오차에는 실측 데이터에 기반한 보정이 없습니다.

## 재현

Python API에서 개별 실험의 센서를 지정할 수 있습니다.

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

공개된 고정 평가와는 별개의 개별 실험입니다. 자세 필터를 제거하려면
`ControllerOptions(attitude_filter=False)`를 지정합니다. 나머지 기능 제거
설정은 앞의 실험 설명과 소스에서 확인할 수 있습니다.

Python 3.12를 설치하고 저장소 루트에서 검증한 의존성 버전을 설치합니다.

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

각 실험에는 새로운 출력 폴더가 필요합니다. 중단한 단계는 소스가 같을 때
`--resume`으로 이어서 실행할 수 있습니다. 완료한 결과를 다시 계산하거나
교체하지 않습니다. 결과 검증기는 Python 표준 라이브러리만으로 요약 수치를
다시 계산하며, 비행을 실행하지 않습니다. 기록의 일관성을 확인하는 도구이며,
물리 데이터의 진위를 입증하는 도구는 아닙니다.

## 참고 자료

- [NASA 모델링 및 시뮬레이션 표준 7009](https://standards.nasa.gov/standard/nasa/nasa-std-7009)
- [NASA 6자유도 시뮬레이션 비교 사례](https://www.nasa.gov/centers-and-facilities/nesc/6dof-check-cases/)

검증 방법을 참고한 자료입니다. DM100의 표준 준수나 하드웨어 자격 검증,
3D 구현을 주장하지 않습니다.
