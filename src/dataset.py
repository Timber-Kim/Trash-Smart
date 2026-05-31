import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data"


def get_transforms(split: str, img_size: int = 300):
    if split == "train":
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class TrashDataset(Dataset):
    """ImageFolder-compatible dataset for trash classification."""

    def __init__(self, root: str, split: str = "train", img_size: int = 300):
        self.root = Path(root)
        self.split = split
        self.transform = get_transforms(split, img_size)

        with open(DATA_DIR / "classes.json") as f:
            classes_info = json.load(f)
        self.class_names = [c["name_en"] for c in classes_info["classes"]]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = self.class_to_idx[class_name]
            for img_path in class_dir.glob("*.jpg"):
                samples.append((img_path, label))
            for img_path in class_dir.glob("*.png"):
                samples.append((img_path, label))
            for img_path in class_dir.glob("*.jpeg"):
                samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        augmented = self.transform(image=image)
        return augmented["image"], label


def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: int = 300, num_workers: int = 4):
    train_ds = TrashDataset(data_dir, split="train", img_size=img_size)
    val_ds = TrashDataset(data_dir, split="val", img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="데이터셋 전처리 실행")
    parser.add_argument("--data_dir", default=str(DATA_DIR / "raw"))
    args = parser.parse_args()

    if args.prepare:
        print("데이터셋 전처리를 시작합니다...")
        print("data/README.md의 가이드를 참고하여 먼저 데이터를 다운로드하세요.")
