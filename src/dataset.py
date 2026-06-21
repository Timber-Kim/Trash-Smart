# License: BSD
# Author: Sasank Chilamkurthy

import os
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class RecycleDataset(torch.utils.data.Dataset):
    """
    2단계 폴더 구조를 지원하는 커스텀 Dataset.
    예: data/train/can/aluminum_can/*.jpg
    ImageFolder는 1단계 구조만 지원하므로 os.walk로 직접 탐색.
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.transform = transform if transform is not None else data_transforms[split]
        self.samples = []      # (이미지 경로, 레이블 인덱스) 리스트
        self.class_names = []

        split_dir = os.path.join(root_dir, split)

        # 이미지를 직접 포함하는 leaf 폴더만 수집
        leaf_dirs = []
        for dirpath, _, filenames in os.walk(split_dir):
            imgs = [f for f in filenames if Path(f).suffix in IMG_EXTS]
            if imgs:
                leaf_dirs.append((dirpath, imgs))

        leaf_dirs.sort(key=lambda x: x[0])  # 재현성을 위해 정렬
        self.class_names = [os.path.basename(d) for d, _ in leaf_dirs]

        for label_idx, (dirpath, imgs) in enumerate(leaf_dirs):
            for fname in sorted(imgs):
                self.samples.append((os.path.join(dirpath, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(aug_cfg=None):
    """
    config.yaml의 augmentation 섹션을 받아 transforms 딕셔너리 반환.
    aug_cfg가 None이면 기본값 사용.
    """
    aug = aug_cfg or {}

    train_list = []

    if aug.get('random_resized_crop', True):
        train_list.append(transforms.RandomResizedCrop(224))
    else:
        train_list.extend([transforms.Resize(256), transforms.CenterCrop(224)])

    if aug.get('random_horizontal_flip', True):
        train_list.append(transforms.RandomHorizontalFlip())

    if aug.get('random_vertical_flip', False):
        train_list.append(transforms.RandomVerticalFlip())

    rotation = aug.get('random_rotation', 15)
    if rotation:
        train_list.append(transforms.RandomRotation(rotation))

    if aug.get('color_jitter', True):
        train_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

    if aug.get('random_grayscale', False):
        train_list.append(transforms.RandomGrayscale(p=0.1))

    if aug.get('gaussian_blur', False):
        train_list.append(transforms.GaussianBlur(kernel_size=3))

    train_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    eval_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    return {
        'train': transforms.Compose(train_list),
        'val':   transforms.Compose(eval_list),
        'test':  transforms.Compose(eval_list),
    }


def get_dataloaders(data_dir='data', batch_size=32, num_workers=4, aug_cfg=None):
    split_transforms = build_transforms(aug_cfg)
    image_datasets = {
        split: RecycleDataset(data_dir, split=split, transform=split_transforms[split])
        for split in ['train', 'val', 'test']
    }
    dataloaders = {
        split: torch.utils.data.DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers
        )
        for split in ['train', 'val', 'test']
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in ['train', 'val', 'test']}
    class_names = image_datasets['train'].class_names
    return dataloaders, dataset_sizes, class_names
