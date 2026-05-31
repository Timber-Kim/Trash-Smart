# Trash-Smart — AI 기반 재활용 분리수거 분류기

쓰레기 이미지를 입력하면 AI가 종류를 분류하고, 한국 기준 배출 방법을 안내합니다.

---

## 주요 기능

- 이미지 업로드 시 쓰레기 종류 자동 분류
- 분류 결과에 따른 한국형 분리수거 배출 가이드 제공
- 다중 모델 비교 실험 (<!-- TODO: 모델 이름 -->)

---

## 데이터셋

<!-- TODO: 사용한 데이터셋 이름, 출처, 클래스 수, 이미지 수 기재 -->

---

## 모델

<!-- TODO: 사용한 모델 및 간단한 설명 기재 -->

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

```bash
# 학습
python src/train.py --config configs/config.yaml

# 평가
python src/evaluate.py --checkpoint models/checkpoints/best.pth

# 웹 데모 실행
uvicorn app.main:app --reload
```

---

## 결과

<!-- TODO: 학습 완료 후 정확도, Confusion Matrix 등 결과 기재 -->

---

## 팀 역할 분담

| 역할 | 담당 |
|------|------|
| 데이터 수집 및 전처리 | <!-- TODO --> |
| 모델 학습 및 실험 | <!-- TODO --> |
| 결과 분석 및 시각화 | <!-- TODO --> |
| 데모 UI 구현 | <!-- TODO --> |
| 보고서 및 발표 | <!-- TODO --> |

---

## 참고

- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [HuggingFace](https://huggingface.co/)
