# 포트폴리오 시각 자료

[English](README.md) | [한국어](README.ko.md)

아래에는 한국어 그래프와 썸네일을 연결했습니다. [영어 안내](README.md)에서는 같은 데이터를 사용한 영문 이미지를 볼 수 있습니다. 수치는 `visual-data.json`에 기록한 실험 자료와 체크섬을 근거로 하며, 궤적과 측정값, 평가 집단 및 원래 실험 보고서는 그대로입니다.

생성기는 Python 기본 라이브러리만 사용합니다. 첫 명령은 보존한 한국어 표시 자료를 생성하고, 두 번째 명령은 글자 간격을 조정한 영어와 한국어 사본을 만듭니다. 그래프를 다시 생성해도 비행을 재계산하거나 과학적 평가를 검증하지는 않습니다.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
python docs/assets/portfolio/render_localized.py
```

`visual-receipt.json`은 보존한 SVG의 체크섬을 표시용 데이터에 연결합니다. 각 언어 폴더의 `localization-receipt.json`에는 원본과 출력 파일의 체크섬 및 그래프 도형의 보존 여부를 기록합니다. 다운로드 ZIP은 해당 버전의 자료로 유지하며, 이번 문서 변경으로 새로운 성능 측정값을 추가하지 않았습니다.

## 시각 자료

- [로켓 발사와 회수 개요](ko/dm100-thumbnail.svg)
- [평가 집단별 기준 충족 결과](ko/dm100-validation.svg)
- [동일 조건의 RCS 추진제 소모량 비교](ko/dm100-propellant.svg)
- [저장된 대표 비행의 궤적](ko/dm100-trajectory.svg)
