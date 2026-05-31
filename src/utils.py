import json
import random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classes(classes_json: str = None) -> list[dict]:
    if classes_json is None:
        classes_json = Path(__file__).parent.parent / "data" / "classes.json"
    with open(classes_json, encoding="utf-8") as f:
        return json.load(f)["classes"]


def save_checkpoint(model, optimizer, epoch: int, val_acc: float, path: str):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)
    print(f"  체크포인트 저장: {path} (val_acc={val_acc:.4f})")


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training / Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  학습 곡선 저장: {save_path}")
    else:
        plt.show()
