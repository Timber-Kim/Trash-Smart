# Trash-Smart — AI 기반 재활용 분리수거 분류기

쓰레기 이미지를 입력하면 AI가 종류를 분류하고, 한국 기준 배출 방법을 안내합니다.

---

## 주요 기능

- 이미지 업로드 시 쓰레기 종류 자동 분류 (15개 클래스)
- ResNet-50 / EfficientNet-B0 두 모델 비교 실험
- 분류 결과에 따른 한국형 분리수거 배출 가이드 제공

---

## 프로젝트 구조

```
Trash-Smart/
├── configs/
│   └── config.yaml          # 학습 하이퍼파라미터 및 증강 설정
├── data/
│   ├── train/               # 학습 데이터 (80%, 32,160장)
│   ├── val/                 # 검증 데이터 (10%, 4,020장)
│   └── test/                # 테스트 데이터 (10%, 4,020장)
├── models/
│   └── checkpoints/         # 학습된 가중치 (.pth)
├── results/                 # 평가 결과 (Confusion Matrix 이미지 등)
├── scripts/
│   └── split_data.py        # 데이터 분할 스크립트 (8:1:1, seed=42)
├── src/
│   ├── dataset.py           # 데이터 로더 및 증강 파이프라인
│   ├── models.py            # ResNet-50 / EfficientNet-B0 모델 정의
│   ├── train.py             # 학습 스크립트
│   └── evaluate.py          # 평가 및 모델 비교 스크립트
├── requirements.txt
└── README.md
```

> `data/` 및 `models/checkpoints/`는 용량 문제로 `.gitignore`에 포함되어 Git에 업로드되지 않습니다.

---

## 데이터셋

**출처:** AI Hub — 재활용 쓰레기 데이터셋  
**총 이미지 수:** 40,200장 | **클래스 수:** 15개 | **분할:** Train 80% / Val 10% / Test 10% (seed=42)

| 카테고리 | 폴더 경로 | 전체 | Train | Val | Test |
|---------|-----------|-----:|------:|----:|-----:|
| 배터리 | `battery/battery/` | 2,400 | 1,920 | 240 | 240 |
| 철 캔 | `can/iron_can/` | 3,000 | 2,400 | 300 | 300 |
| 알루미늄 캔 | `can/aluminum_can/` | 3,000 | 2,400 | 300 | 300 |
| 형광등 | `fluorescent_lamp/fluorescent_lamp/` | 2,400 | 1,920 | 240 | 240 |
| 갈색 유리병 | `glass_bottle/glass bottle_brown/` | 2,100 | 1,680 | 210 | 210 |
| 무색 유리병 | `glass_bottle/glass bottle_colorless/` | 2,100 | 1,680 | 210 | 210 |
| 초록 유리병 | `glass_bottle/glass bottle_green/` | 2,100 | 1,680 | 210 | 210 |
| 종이 | `paper/paper/` | 3,000 | 2,400 | 300 | 300 |
| PE | `plastic/PE/` | 3,000 | 2,400 | 300 | 300 |
| PP | `plastic/PP/` | 3,000 | 2,400 | 300 | 300 |
| PS | `plastic/PS/` | 3,000 | 2,400 | 300 | 300 |
| 유색 패트병 | `plastic_bottle/colored/` | 3,000 | 2,400 | 300 | 300 |
| 무색 패트병 | `plastic_bottle/colorless/` | 3,000 | 2,400 | 300 | 300 |
| 스티로폼 | `styrofoam/styrofoam/` | 3,000 | 2,400 | 300 | 300 |
| 비닐 | `vinyl/vinyl/` | 2,100 | 1,680 | 210 | 210 |

---

## 데이터 증강 (Data Augmentation)

과적합 방지 및 일반화 성능 향상을 위해 Train 셋에만 증강을 적용합니다.  
각 항목은 `configs/config.yaml`에서 사용자가 직접 켜고 끌 수 있습니다.

| 증강 기법 | 설명 | 기본값 |
|-----------|------|:------:|
| `random_horizontal_flip` | 좌우 반전 | `true` |
| `random_vertical_flip` | 상하 반전 | `false` |
| `random_rotation` | 무작위 회전 (±각도) | `15` |
| `color_jitter` | 밝기·대비·채도·색조 무작위 변환 | `true` |
| `random_resized_crop` | 무작위 크롭 후 리사이즈 | `true` |
| `random_grayscale` | 무작위 흑백 변환 | `false` |
| `gaussian_blur` | 가우시안 블러 | `false` |

Val / Test 셋은 증강 없이 Resize(256) → CenterCrop(224) → Normalize만 적용합니다.

---

## 모델

두 모델을 동일한 데이터·설정으로 학습한 뒤 성능을 비교합니다.

| 모델 | 파라미터 수 | 특징 |
|------|:----------:|------|
| **ResNet-50** | ~25M | 잔차 연결(skip connection) 기반, 안정적인 baseline |
| **EfficientNet-B0** | ~5.3M | 복합 스케일링으로 경량·고효율 |

**학습 전략 (Transfer Learning)**

1. **Feature Extraction** (초반 `freeze_epochs`): backbone 동결, 분류 헤드만 학습
2. **Fine-Tuning** (이후 에폭): 전체 레이어 학습, 낮은 learning rate 적용

모델 이름 및 에폭 수는 `configs/config.yaml`에서 변경 가능합니다.

---

## 평가 지표

두 모델을 아래 지표로 비교합니다.

| 지표 | 설명 |
|------|------|
| **Top-1 Accuracy** | 전체 정확도 |
| **Per-class F1 Score** | 클래스별 F1 (PE/PP/PS 등 유사 클래스 집중 분석) |
| **Confusion Matrix** | 클래스 간 혼동 패턴 시각화 |
| **Inference Time** | 이미지 1장당 추론 속도 (ms) |
| **파라미터 수 / 모델 크기** | 경량화 관점의 효율성 비교 |
| **Train/Val Loss 곡선** | 학습 수렴 및 과적합 여부 확인 |

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

```bash
# 데이터 분할 (최초 1회만 실행)
python scripts/split_data.py

# 학습 (config.yaml의 model.name으로 모델 선택)
python src/train.py --config configs/config.yaml

# 두 모델 비교 평가
python src/evaluate.py \
    --checkpoints models/checkpoints/resnet50_best.pth models/checkpoints/efficientnet_b0_best.pth \
    --data_dir data/train \
    --output_dir results
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
| 보고서 및 발표 | <!-- TODO --> |

---

## 참고

- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [HuggingFace](https://huggingface.co/)
- [AI Hub](https://aihub.or.kr/)
