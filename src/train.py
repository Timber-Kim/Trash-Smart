import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.utils import set_seed, get_device, save_checkpoint, plot_training_history
from models import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(config: dict):
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"장치: {device}")

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        num_workers=config.get("num_workers", 4),
    )
    print(f"학습 데이터: {len(train_loader.dataset)}장 | 검증 데이터: {len(val_loader.dataset)}장")

    model = build_model(num_classes=config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    checkpoint_dir = Path(config.get("checkpoint_dir", "models/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{config['epochs']}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            str(checkpoint_dir / "best.pth"))

    save_checkpoint(model, optimizer, config["epochs"], val_acc,
                    str(checkpoint_dir / "last.pth"))
    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          save_path=str(checkpoint_dir / "history.png"))
    print(f"\n학습 완료. 최고 검증 정확도: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
