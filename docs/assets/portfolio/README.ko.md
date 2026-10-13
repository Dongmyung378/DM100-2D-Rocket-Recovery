# 포트폴리오 시각 자료

[English](README.md) | [한국어](README.ko.md)

사이트에서 사용하는 한국어 SVG 그래프와 썸네일입니다. 수치는 `visual-data.json`에 기록한 기존 실험 자료와 체크섬을 근거로 합니다. 학습 모델과 데이터 분할, 선택 판단 및 원래 실험 보고서는 변경하지 않습니다. 평가 범위와 한계는 프로젝트의 본문 문서를 확인하십시오.

생성기는 Python 기본 라이브러리와 포함된 표시용 데이터만 사용합니다. 그래프를 다시 생성하는 것은 모델 실행이나 과학적 평가 검증이 아닙니다.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json`은 SVG 파일의 바이트 수와 체크섬을 표시용 데이터에 연결합니다. 소스 다운로드 ZIP에도 사이트와 같은 시각 자료를 포함합니다. 처리 흐름도는 구현한 단계를 설명하며 새로운 성능 측정값이 아닙니다.

## 시각 자료

- [dm100-thumbnail.svg](dm100-thumbnail.svg)
- [dm100-validation.svg](dm100-validation.svg)
- [dm100-propellant.svg](dm100-propellant.svg)
- [dm100-trajectory.svg](dm100-trajectory.svg)
