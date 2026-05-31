import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.dataset import TrashDataset
from src.utils import get_device, load_classes
from models import build_model
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(checkpoint_path: str, data_dir: str, split: str = "test", img_size: int = 300):
    device = get_device()
    classes = load_classes()
    class_names_ko = [c["name_ko"] for c in classes]
    class_names_en = [c["name_en"] for c in classes]
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, checkpoint_path=checkpoint_path).to(device)
    model.eval()

    dataset = TrashDataset(data_dir, split=split, img_size=img_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    correct_top1, correct_top3, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="평가 중"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        top3 = outputs.topk(3, dim=1).indices
        correct_top1 += (top3[:, 0] == labels).sum().item()
        correct_top3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        all_preds.extend(top3[:, 0].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"\nTop-1 Accuracy: {correct_top1/total:.4f} ({correct_top1}/{total})")
    print(f"Top-3 Accuracy: {correct_top3/total:.4f} ({correct_top3}/{total})")
    print("\n분류 리포트:")
    print(classification_report(all_labels, all_preds, target_names=class_names_ko))

    _plot_confusion_matrix(all_labels, all_preds, class_names_ko)


def _plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.ylabel("실제")
    plt.xlabel("예측")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig("models/checkpoints/confusion_matrix.png", dpi=150)
    print("  혼동 행렬 저장: models/checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/checkpoints/best.pth")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.split)
