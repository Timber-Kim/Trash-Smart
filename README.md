# Trash-Smart 🗑️

**컴퓨터 비전 기반 한국형 재활용 분류 가이드 앱**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 프로젝트 개요

Trash-Smart는 스마트폰 카메라로 쓰레기를 촬영하면 AI가 자동으로 재활용 분류를 안내해주는 앱입니다.  
한국의 재활용 분리수거 기준에 맞게 설계되어 있으며, 해외 공개 데이터셋을 기반으로 학습합니다.

## 분류 카테고리 (8종)

| 번호 | 카테고리 | 영문 | 예시 |
|------|----------|------|------|
| 0 | 종이 | Paper | 신문지, 책, 종이박스 |
| 1 | 플라스틱 | Plastic | 페트병, 플라스틱 용기 |
| 2 | 유리 | Glass | 유리병, 유리컵 |
| 3 | 금속/캔 | Metal | 알루미늄캔, 철캔 |
| 4 | 비닐/포장재 | Vinyl | 비닐봉지, 랩 |
| 5 | 스티로폼 | Styrofoam | 스티로폼 상자, 완충재 |
| 6 | 음식물쓰레기 | Food Waste | 음식 찌꺼기 |
| 7 | 일반쓰레기 | Trash | 분류 불가 쓰레기 |

## 기술 스택

- **모델**: EfficientNet-B3 (Transfer Learning)
- **프레임워크**: PyTorch
- **백엔드**: FastAPI
- **프론트엔드**: HTML/CSS/JavaScript
- **데이터 증강**: Albumentations

## 참조 데이터셋

- [TrashNet](https://github.com/garythung/trashnet) — Stanford, 6 클래스 2,527장
- [TACO Dataset](http://tacodataset.org/) — Trash Annotations in Context
- [Garbage Classification (Kaggle)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

자세한 내용은 [data/README.md](data/README.md)를 참고하세요.

## 프로젝트 구조

```
Trash-Smart/
├── data/
│   ├── README.md           # 데이터셋 다운로드 가이드
│   ├── classes.json        # 분류 카테고리 정의
│   ├── raw/                # 원본 데이터 (gitignore)
│   └── processed/          # 전처리된 데이터 (gitignore)
├── models/
│   ├── classifier.py       # 모델 아키텍처
│   └── checkpoints/        # 학습된 가중치 (gitignore)
├── src/
│   ├── dataset.py          # 데이터 로딩 및 증강
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   ├── predict.py          # 추론 스크립트
│   └── utils.py            # 유틸리티 함수
├── app/
│   ├── main.py             # FastAPI 서버
│   ├── templates/          # HTML 템플릿
│   └── static/             # CSS, JS, 이미지
├── configs/
│   └── config.yaml         # 학습 설정
├── notebooks/
│   └── exploration.ipynb   # 데이터 탐색 노트북
├── requirements.txt
└── README.md
```

## 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/your-username/Trash-Smart.git
cd Trash-Smart
pip install -r requirements.txt
```

### 2. 데이터셋 준비

```bash
# data/README.md의 가이드를 따라 데이터셋 다운로드
# 다운로드 후 아래 명령어로 전처리
python src/dataset.py --prepare
```

### 3. 모델 학습

```bash
python src/train.py --config configs/config.yaml
```

### 4. 웹 앱 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속

## 성능 목표

| 지표 | 목표값 |
|------|--------|
| Top-1 Accuracy | ≥ 85% |
| Top-3 Accuracy | ≥ 95% |
| 추론 시간 | ≤ 200ms |

## 라이센스

MIT License — 자세한 내용은 [LICENSE](LICENSE) 참고
