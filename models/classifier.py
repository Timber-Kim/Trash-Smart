import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


class TrashClassifier(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b3(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(num_classes: int = 8, checkpoint_path: str = None) -> TrashClassifier:
    model = TrashClassifier(num_classes=num_classes)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
    return model
