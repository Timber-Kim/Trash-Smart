# 프로젝트 노트

## 데이터셋 정리 과정

AI-Hub에서 한국의 재활용 분류 기준에 맞춰 한국의 재활용 쓰레기 이미지를 준비 및 분류 (약 40,200장)

- 준비한 데이터셋의 결측치 제거 등 전처리를 진행하고 8:1:1 (train : val : test)로 분할
- 추후 모델별 비교 실험에 용이하도록 test 데이터를 증강할 수 있는 코드를 추가 (인자를 받아 특정 카테고리만 증강 진행)
- 모든 이미지에 대해 256px 사이즈로 리사이즈 진행 (정사각형 아님, 긴 쪽 기준 비율 유지)
- 용량 문제로 인해 Google Drive 업로드, 최종 구성:

| 분할 | 장수 |
|------|------|
| train | 32,157장 |
| val | 4,019장 |
| test | 4,020장 |
| **합계** | **40,196장** |

---

## 코드에서 AI를 사용한 부분

### dataset.py

프로젝트의 데이터셋을 불러오는 구조가 튜토리얼과 달라 AI로 수정

- `can/aluminum_can/` 폴더 안에 폴더가 있는 2단계 구조이므로 튜토리얼의 `ImageFolder`로 구현 불가
- augmentation을 config에서 동적으로 읽는 구조도 튜토리얼에 없어 AI로 추가

```python
class RecycleDataset(torch.utils.data.Dataset):
def get_dataloaders(data_dir='data', batch_size=32, num_workers=4):
def build_transforms(aug_cfg=None):
```

### models.py

비교 실험을 위해 EfficientNet-B0 모델을 AI를 사용해 추가

```python
elif model_name == 'efficientnet_b0':
```

### train.py

- 학습 루프 자체는 튜토리얼 기반으로 작성
- yaml config 연동, 경로 설계, 예외처리(`os.path.exists`) 부분은 AI 사용

### evaluate.py

- 평가에 여러 지표를 사용하기 위해 구성 (비교/실험 시 사용)
- `visualize_model()`은 튜토리얼 기반
- 저장된 `.pth` 가중치를 이용해 학습 없이 비교할 수 있도록 구현
- 다중 모델 비교, F1, confusion matrix, 추론 시간 측정은 튜토리얼에 없는 내용으로 AI 사용

### config.yaml

튜토리얼에는 없는 내용으로, 실험 편의성을 위해 AI를 사용해 작성
