import io
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import get_device, load_classes
from models import build_model


CLASSES = load_classes()
NUM_CLASSES = len(CLASSES)

_transform = A.Compose([
    A.Resize(height=300, width=300),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class Predictor:
    def __init__(self, checkpoint_path: str):
        self.device = get_device()
        self.model = build_model(NUM_CLASSES, checkpoint_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[dict]:
        """이미지 바이트를 받아 상위 k개 분류 결과를 반환합니다."""
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        tensor = _transform(image=image)["image"].unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top_k_probs, top_k_indices = probs.topk(top_k)
        results = []
        for prob, idx in zip(top_k_probs.cpu().tolist(), top_k_indices.cpu().tolist()):
            cls = CLASSES[idx]
            results.append({
                "rank": len(results) + 1,
                "class_id": idx,
                "name_ko": cls["name_ko"],
                "name_en": cls["name_en"],
                "confidence": round(prob, 4),
                "disposal_guide": cls["disposal_guide"],
                "color": cls["color"],
            })
        return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("사용법: python src/predict.py <checkpoint_path> <image_path>")
        sys.exit(1)

    predictor = Predictor(sys.argv[1])
    with open(sys.argv[2], "rb") as f:
        results = predictor.predict(f.read())

    for r in results:
        print(f"{r['rank']}. {r['name_ko']} ({r['name_en']}) — {r['confidence']*100:.1f}%")
        print(f"   배출 방법: {r['disposal_guide']}")
