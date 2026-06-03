# 팀원 협업 가이드 — GitHub & Google Colab

> 코드를 몰라도 괜찮아요. 이 가이드를 순서대로 따라오면 실험을 돌리고 결과를 공유할 수 있어요.

---

## 목차

1. [GitHub — 코드 받아오기](#1-github--코드-받아오기)
2. [Google Drive — 데이터 올리기](#2-google-drive--데이터-올리기)
3. [Google Colab — 실험 환경 세팅](#3-google-colab--실험-환경-세팅)
4. [실험하기 — 설정 바꾸고 학습 돌리기](#4-실험하기--설정-바꾸고-학습-돌리기)
5. [결과 저장 & 공유하기](#5-결과-저장--공유하기)

---

## 1. GitHub — 코드 받아오기

GitHub는 코드를 공유하는 공간이에요. 팀원은 **코드를 직접 수정할 필요 없이 다운로드만** 하면 돼요.

### 처음 한 번만: 코드 내려받기

1. [https://github.com/Timber-Kim/Trash-Smart](https://github.com/Timber-Kim/Trash-Smart) 접속
2. 초록색 **`< > Code`** 버튼 클릭
3. **`Download ZIP`** 클릭 → 압축 해제

> 💡 코드가 업데이트되면 팀장이 알려줄 거예요. 그때 ZIP을 다시 받으면 돼요.

### 모델 체크포인트 받기 (학습된 가중치 파일)

팀장이 학습을 완료하면 **GitHub Releases** 페이지에 `.pth` 파일을 올릴 거예요.

1. GitHub 저장소 페이지 오른쪽 **Releases** 클릭
2. 최신 릴리즈에서 `resnet50_best.pth`, `efficientnet_b0_best.pth` 다운로드
3. 다운받은 파일을 `models/checkpoints/` 폴더 안에 넣기

---

## 2. Google Drive — 데이터 올리기

> **팀장만 1회 업로드** — 팀원은 공유 링크를 받아서 접근만 하면 돼요.

이미지 데이터는 256×256으로 미리 리사이즈한 버전(`data_256/`)을 사용해요.  
원본 ~140GB → 리사이즈 후 **약 600MB~1GB** 로 줄어들어 업로드가 빠르게 완료돼요.

### 팀장: 업로드 방법

1. [drive.google.com](https://drive.google.com) 접속
2. **`+ 새로 만들기`** → **`폴더`** → 이름: `data_256`
3. 해당 폴더에 들어가서 `train/`, `val/`, `test/` 폴더를 드래그해서 업로드

   ```
   내 드라이브/
   └── data_256/
       ├── train/
       │   ├── battery/battery/
       │   ├── can/aluminum_can/
       │   └── ... (15개 클래스)
       ├── val/
       └── test/
   ```

4. 업로드 완료 후 폴더 공유 링크를 팀원에게 전달  
   (폴더 우클릭 → **공유** → **링크 복사** → "링크가 있는 모든 사용자" 설정)

### 팀원: Drive에 바로가기 추가

1. 팀장에게 받은 링크로 접속
2. 폴더 이름 옆 **`⋮`** → **`바로가기 추가`** → 내 드라이브 선택  
   (이렇게 하면 내 Drive 용량을 쓰지 않아요)

---

## 3. Google Colab — 실험 환경 세팅

Google Colab은 구글이 제공하는 무료 GPU 환경이에요. 설치 없이 브라우저에서 코드를 돌릴 수 있어요.

### 3-1. Colab 노트북 열기

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. **`파일`** → **`새 노트북`**
3. 상단 메뉴 **`런타임`** → **`런타임 유형 변경`** → **하드웨어 가속기: `T4 GPU`** 선택 → 저장

### 3-2. 코드 받아오기 & 드라이브 연결

아래 코드 블록들을 Colab 셀에 **순서대로** 붙여넣고 ▶ 버튼으로 실행하세요.

**셀 1 — Google Drive 연결**
```python
from google.colab import drive
drive.mount('/content/drive')
```
> 팝업이 뜨면 구글 계정으로 로그인하고 허용 클릭

**셀 2 — GitHub에서 코드 받기**
```python
!git clone https://github.com/Timber-Kim/Trash-Smart.git
%cd Trash-Smart
```

**셀 3 — 필요한 패키지 설치**
```python
!pip install pyyaml seaborn scikit-learn -q
```

**셀 3-1 — Colab용 num_workers 설정** (Colab은 Linux라 멀티프로세싱 가능)
```python
import re

with open('configs/config.yaml', 'r') as f:
    cfg = f.read()

cfg = re.sub(r'num_workers:.*', 'num_workers: 4', cfg)

with open('configs/config.yaml', 'w') as f:
    f.write(cfg)

print("num_workers를 4로 변경했어요 (학습 속도 향상)")
```

**셀 4 — 데이터 폴더 연결**
```python
# Drive의 data_256 폴더를 프로젝트의 data 폴더로 연결
!ln -s /content/drive/MyDrive/data_256 data
```
> Drive에 바로가기로 추가한 경우 경로가 다를 수 있어요.  
> 그럴 때는 왼쪽 파일 탐색기(📁)에서 `drive/MyDrive/` 안의 실제 폴더명을 확인하세요.

**셀 5 — 모델 체크포인트 폴더 만들기**
```python
!mkdir -p models/checkpoints
```

**셀 6 — 데이터 잘 연결됐는지 확인**
```python
import os
for split in ['train', 'val', 'test']:
    count = sum(len(files) for _, _, files in os.walk(f'data/{split}'))
    print(f'{split}: {count}장')
```
> train: 32160장 / val: 4020장 / test: 4020장 이 나오면 정상

---

## 4. 실험하기 — 설정 바꾸고 학습 돌리기

모든 실험 설정은 **`configs/config.yaml`** 파일 하나에서 관리해요.  
이 파일만 수정하면 코드를 건드리지 않아도 돼요.

### config.yaml 열기 & 수정

Colab 왼쪽 파일 탐색기(📁 아이콘) → `Trash-Smart/configs/config.yaml` 더블클릭

```yaml
model:
  name: resnet50          # ← 'resnet50' 또는 'efficientnet_b0' 로 변경

training:
  num_epochs: 25          # ← 학습 에폭 수
  batch_size: 32          # ← 배치 크기 (메모리 부족하면 16으로 낮추기)
  lr: 0.001               # ← 학습률

augmentation:
  random_horizontal_flip: true   # ← true / false 로 켜고 끄기
  random_rotation: 15            # ← 0 이면 비활성화
  color_jitter: true
  gaussian_blur: false
```

### 학습 실행

**셀 — ResNet-50 학습**
```python
!python src/train.py --config configs/config.yaml
```

**셀 — EfficientNet-B0 학습** (`config.yaml`에서 `name: efficientnet_b0`으로 바꾼 후)
```python
!python src/train.py --config configs/config.yaml
```

> 학습이 끝나면 `Best val Acc: 0.XXXX` 가 출력되고  
> `models/checkpoints/resnet50_best.pth` 파일이 생성돼요.

---

## 5. 결과 저장 & 공유하기

### 체크포인트(.pth) Drive에 백업

```python
!cp models/checkpoints/resnet50_best.pth /content/drive/MyDrive/Trash-Smart-data/
!cp models/checkpoints/efficientnet_b0_best.pth /content/drive/MyDrive/Trash-Smart-data/
```

### 평가 실행 (두 모델 비교)

```python
!python src/evaluate.py \
    --checkpoints models/checkpoints/resnet50_best.pth \
                  models/checkpoints/efficientnet_b0_best.pth \
    --data_dir data \
    --output_dir results
```

실행 결과:
- 터미널에 Accuracy, F1, 추론 시간 비교표 출력
- `results/resnet50_confusion_matrix.png` — confusion matrix 이미지

### 결과 이미지 Drive에 저장

```python
!cp -r results/ /content/drive/MyDrive/Trash-Smart-data/results/
```

---

## 자주 생기는 문제

| 증상 | 해결 방법 |
|------|-----------|
| `data 폴더를 찾을 수 없음` | 셀 4(드라이브 연결) 다시 실행 |
| `RuntimeError: CUDA out of memory` | config.yaml에서 `batch_size: 16` 으로 낮추기 |
| 런타임 연결 끊김 | Colab 무료 버전은 90분 idle 시 자동 종료 — 다시 연결 후 셀 2부터 재실행 |
| `pth 파일 없음` | Releases에서 다운받아 `models/checkpoints/`에 넣기 |
| `data/train: 0장` | 셀 4 심링크 경로 확인 — Drive 폴더명과 일치하는지 체크 |

