# 데이터셋 가이드

Trash-Smart는 아래 해외 공개 데이터셋을 기반으로 학습합니다.  
직접 다운로드 후 아래 구조에 맞게 배치해주세요.

---

## 1. TrashNet (추천)

- **출처**: Stanford University — Gary Thung, Yang, 2016
- **URL**: https://github.com/garythung/trashnet
- **클래스**: glass, paper, cardboard, plastic, metal, trash (6종, 총 2,527장)
- **라이센스**: MIT

### 매핑 (TrashNet → Trash-Smart)

| TrashNet | Trash-Smart |
|----------|-------------|
| paper | 종이 (paper) |
| cardboard | 종이 (paper) |
| plastic | 플라스틱 (plastic) |
| glass | 유리 (glass) |
| metal | 금속/캔 (metal) |
| trash | 일반쓰레기 (trash) |

---

## 2. TACO Dataset

- **출처**: Pedro F. Proença, Pedro Simões (2020)
- **URL**: http://tacodataset.org/
- **특징**: 실외 환경 쓰레기 사진, 60개 세부 카테고리 → Trash-Smart 카테고리로 매핑 필요
- **라이센스**: CC BY 4.0

---

## 3. Garbage Classification (Kaggle)

- **URL**: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- **클래스**: cardboard, glass, metal, paper, plastic, trash (6종, 총 2,467장)
- **라이센스**: Open Database License (ODbL)

---

## 디렉토리 구조

데이터 다운로드 후 아래와 같이 `data/processed/` 에 배치하세요:

```
data/processed/
├── train/
│   ├── paper/
│   ├── plastic/
│   ├── glass/
│   ├── metal/
│   ├── vinyl/
│   ├── styrofoam/
│   ├── food_waste/
│   └── trash/
├── val/
│   └── (동일 구조)
└── test/
    └── (동일 구조)
```

권장 분할 비율: **train 70% / val 15% / test 15%**

---

## 데이터 부족 카테고리 대응

TrashNet/Kaggle에 없는 카테고리(`vinyl`, `styrofoam`, `food_waste`)는 아래 방법으로 보완하세요:

1. **직접 촬영** — 스마트폰으로 직접 수집
2. **Roboflow Universe** — https://universe.roboflow.com 에서 관련 데이터 검색
3. **Open Images V7** — Google의 대규모 오픈 데이터셋
4. **데이터 증강** — `src/dataset.py`의 Albumentations 파이프라인 활용

---

## 인용

```bibtex
@misc{trashnet2016,
  author = {Gary Thung and Mindy Yang},
  title  = {TrashNet},
  year   = {2016},
  url    = {https://github.com/garythung/trashnet}
}

@article{taco2020,
  title   = {TACO: Trash Annotations in Context for Litter Detection},
  author  = {Pedro F. Proença and Pedro Simões},
  journal = {arXiv preprint arXiv:2003.06975},
  year    = {2020}
}
```
