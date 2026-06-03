import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int, feature_extract: bool = False):
    """
    model_name: 'resnet50' 또는 'efficientnet_b0'
    num_classes: 분류할 클래스 수 (15)
    feature_extract: True  → (튜토리얼 ConvNet as fixed feature extractor)
                     False → (튜토리얼 Finetuning the ConvNet)
    """
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')

        if feature_extract:
            # for param in model_conv.parameters(): param.requires_grad = False
            for param in model.parameters():
                param.requires_grad = False

        #           model_ft.fc = nn.Linear(num_ftrs, 2) → 15 클래스
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        # EfficientNet은 튜토리얼에 없으므로 torchvision 문서 참고
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        if feature_extract:# Parameters of newly constructed modules have requires_grad=True by default
            # ResNet과 동일 방식으로 backbone 동결
            for param in model.parameters():
                param.requires_grad = False

        # EfficientNet은 fc 대신 classifier[1]이 출력 레이어 (ResNet의 fc와 동일 역할)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}. 'resnet50' 또는 'efficientnet_b0' 사용")

    return model
