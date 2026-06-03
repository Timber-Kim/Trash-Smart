# 협업 가이드 — Google Colab 실험 가이드

> 코드를 몰라도 괜찮아요. 이 가이드를 순서대로 따라오면 실험을 돌리고 결과를 공유할 수 있어요.

---

## 목차

1. [데이터 준비 (처음 한 번만)](#1-데이터-준비-처음-한-번만)
2. [Colab 환경 세팅 (처음 한 번만)](#2-colab-환경-세팅-처음-한-번만)
3. [실험하기 — 설정 바꾸고 학습 돌리기](#3-실험하기--설정-바꾸고-학습-돌리기)
4. [결과 저장 & 공유하기](#4-결과-저장--공유하기)
5. [모델 비교 (선택)](#5-모델-비교-선택)

---

## 1. 데이터 준비 (처음 한 번만)

### 1-1. 공유 파일을 내 Drive에 추가

1. 아래 링크 접속  
   👉 [data_256.zip 공유 링크](https://drive.google.com/file/d/1diKW4pCsXoBMOojGe-fKR_rnf2K6fkla/view?usp=drive_link)

2. 파일 이름 옆 **`⋮`** 또는 우클릭 → **`내 드라이브에 바로가기 추가`** 클릭  
   (이렇게 하면 내 Drive 용량을 쓰지 않아요)

   ```
   내 드라이브/
   └── data_256.zip  ← 바로가기가 여기에 생겨요
   ```

> 바로가기 추가 후 [drive.google.com](https://drive.google.com)에서 `data_256.zip`이 보이면 성공이에요.

---

## 2. Colab 환경 세팅 (처음 한 번만)

### 2-1. Colab 노트북 열기

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. **`파일`** → **`새 노트북`**
3. 상단 메뉴 **`런타임`** → **`런타임 유형 변경`** → **하드웨어 가속기: `T4 GPU`** 선택 → 저장

### 2-2. 세팅 셀 순서대로 실행

아래 셀들을 순서대로 붙여넣고 ▶ 버튼으로 실행하세요.

**셀 1 — Drive 연결**
```python
from google.colab import drive
drive.mount('/content/drive')
```
> 팝업이 뜨면 구글 계정으로 로그인하고 허용 클릭

**셀 2 — 코드 받기 & 패키지 설치**
```python
!git clone https://github.com/Timber-Kim/Trash-Smart.git
%cd Trash-Smart
!pip install pyyaml seaborn scikit-learn -q
```

**셀 3 — Colab용 설정 변경**
```python
import re

with open('configs/config.yaml', 'r') as f:
    cfg = f.read()

cfg = re.sub(r'num_workers:.*', 'num_workers: 2', cfg)

with open('configs/config.yaml', 'w') as f:
    f.write(cfg)

print("설정 완료")
```

**셀 4 — 데이터 로컬 복사 + 연결** (2~3분 소요)
```python
import os

# Drive에서 Colab 로컬로 복사 후 압축 해제 (Drive 직접 읽으면 학습이 매우 느려짐)
!cp /content/drive/MyDrive/data_256.zip /content/data_256.zip
!unzip -q /content/data_256.zip -d /content/

if os.path.islink('data') or os.path.exists('data'):
    os.remove('data')
!ln -s /content/data_256 data

print("완료! 데이터 연결 성공")
```

**셀 5 — 데이터 확인**
```python
import os

for split in ['train', 'val', 'test']:
    exists = os.path.isdir(f'data/{split}')
    print(f'data/{split}: {"존재 ✅" if exists else "없음 ❌"}')

sample_dir = 'data/train/battery/battery'
if os.path.isdir(sample_dir):
    count = len(os.listdir(sample_dir))
    print(f'\n샘플 확인 (battery): {count}장 → {"정상 ✅" if count > 0 else "비어있음 ❌"}')
```
> 세 줄 모두 `존재 ✅` + battery 장수가 나오면 정상

**셀 6 — 체크포인트 폴더 생성**
```python
!mkdir -p models/checkpoints
```

---

## 3. 실험하기 — 설정 바꾸고 학습 돌리기

모든 실험 설정은 **`configs/config.yaml`** 파일 하나에서 관리해요.

### 모델 선택

**ResNet-50 학습 (기본값)**
```python
import re

with open('configs/config.yaml', 'r') as f:
    cfg = f.read()

cfg = re.sub(r'name:.*', 'name: resnet50', cfg)

with open('configs/config.yaml', 'w') as f:
    f.write(cfg)

print("모델: resnet50")
```

**EfficientNet-B0 학습**
```python
import re

with open('configs/config.yaml', 'r') as f:
    cfg = f.read()

cfg = re.sub(r'name:.*', 'name: efficientnet_b0', cfg)

with open('configs/config.yaml', 'w') as f:
    f.write(cfg)

print("모델: efficientnet_b0")
```

> 파일 탐색기에서 직접 수정하면 저장이 안 될 수 있어요. 위 셀로 변경하는 게 확실해요.

### 기타 설정 (선택)

Colab 왼쪽 파일 탐색기(📁 아이콘) → `Trash-Smart/configs/config.yaml` 더블클릭

```yaml
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

```python
!python src/train.py --config configs/config.yaml
```

> 학습이 끝나면 `Best val Acc: 0.XXXX` 가 출력되고  
> `models/checkpoints/resnet50_best.pth` (또는 `efficientnet_b0_best.pth`) 파일이 생성돼요.

---

## 4. 결과 저장 & 공유하기

학습이 끝나면 바로 Drive에 백업하세요. **Colab은 세션 종료 시 파일이 모두 사라져요.**

```python
import os, shutil

backup_dir = '/content/drive/MyDrive/Trash-Smart-checkpoints'
os.makedirs(backup_dir, exist_ok=True)

for f in os.listdir('models/checkpoints'):
    shutil.copy(f'models/checkpoints/{f}', backup_dir)
    print(f"백업 완료: {f}")
```

---

## 5. 모델 비교

학습 없이 저장된 가중치로 바로 비교 평가할 수 있어요.  
섹션 2(환경 세팅)까지 완료한 상태에서 아래 셀을 실행하세요.

**셀 — 체크포인트 다운로드 (GitHub Releases)**
```python
!mkdir -p models/checkpoints

!wget https://github.com/Timber-Kim/Trash-Smart/releases/download/first/resnet50_best.pth \
     -O models/checkpoints/resnet50_best.pth

!wget https://github.com/Timber-Kim/Trash-Smart/releases/download/first/efficientnet_b0_best.pth \
     -O models/checkpoints/efficientnet_b0_best.pth

print("다운로드 완료!")
```

**셀 — 비교 평가 실행**
```python
!python src/evaluate.py \
    --checkpoints models/checkpoints/resnet50_best.pth \
                  models/checkpoints/efficientnet_b0_best.pth \
    --data_dir data \
    --output_dir results
```

> 실행 결과: Accuracy, Macro F1, 추론 시간 비교표 출력  
> `results/` 폴더에 confusion matrix 이미지 저장

**셀 — 결과 Drive에 백업**
```python
!cp -r results/ /content/drive/MyDrive/Trash-Smart-results/
```

---

## 6. 데모 앱 실행 (발표용)

학습 없이 저장된 가중치로 바로 실행할 수 있어요.  
섹션 2(환경 세팅) + 섹션 5(체크포인트 다운로드)까지 완료한 상태에서 실행하세요.

**셀 — Gradio 설치**
```python
!pip install gradio -q
```

**셀 — 앱 실행 (공개 링크 생성)**
```python
!python app.py --checkpoint models/checkpoints/resnet50_best.pth --model resnet50 --share
```

> 실행하면 `Running on public URL: https://xxxx.gradio.live` 링크가 나와요.  
> 해당 링크를 폰 브라우저에서 열면 카메라로 찍어서 바로 분류 가능해요.

EfficientNet으로 바꾸려면:
```python
!python app.py --checkpoint models/checkpoints/efficientnet_b0_best.pth --model efficientnet_b0 --share
```

---

## 자주 생기는 문제

| 증상 | 해결 방법 |
|------|-----------|
| `data/train: 없음 ❌` | 셀 4 다시 실행 |
| `RuntimeError: CUDA out of memory` | config.yaml에서 `batch_size: 16` 으로 낮추기 |
| 런타임 연결 끊김 | 다시 연결 후 셀 2~6 재실행 (데이터는 셀 4부터 다시) |
| `pth 파일 없음` | 학습을 먼저 돌리거나 상대방 Drive에서 복사 |
| `data_256.zip 찾을 수 없음` | 1번 단계에서 Drive 바로가기 추가했는지 확인 |
