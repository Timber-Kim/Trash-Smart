"""
Grad-CAM 오분류 시각화

사용법:
    python scripts/gradcam.py \
        --checkpoint models/checkpoints/resnet50_best.pth \
        --model resnet50 \
        --data_dir data

옵션:
    --num_images  시각화할 오분류 이미지 수 (기본 8)
    --output_dir  저장 경로 (기본 results/gradcam)
    --num_workers DataLoader worker 수 (기본 0, Windows 권장)
"""
import os
import sys
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from dataset import get_dataloaders
from models import get_model
from gradcam import GradCAM, get_target_layer, denorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def overlay(img, cam, alpha=0.45):
    heat = plt.cm.jet(cam)[..., :3]
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 1)


def collect_failures(model, loader, max_total):
    """테스트셋에서 오분류 샘플 수집"""
    model.eval()
    failures = []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(1).cpu()
            for img, t, p in zip(imgs, labels, preds):
                if t != p:
                    failures.append((img, t.item(), p.item()))
                if len(failures) >= max_total:
                    return failures
    return failures


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--checkpoint',  required=True)
    pa.add_argument('--model',       required=True, choices=['resnet50', 'efficientnet_b0'])
    pa.add_argument('--data_dir',    default='data')
    pa.add_argument('--num_images',  type=int, default=8)
    pa.add_argument('--output_dir',  default='results/gradcam')
    pa.add_argument('--num_workers', type=int, default=0)
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    loaders, _, class_names = get_dataloaders(
        data_dir=args.data_dir, batch_size=32, num_workers=args.num_workers)

    model = get_model(args.model, num_classes=len(class_names))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device).eval()

    gcam = GradCAM(model, get_target_layer(model, args.model))

    print("오분류 샘플 수집 중...")
    failures = collect_failures(model, loaders['test'], args.num_images)
    n = len(failures)
    print(f"{n}장 수집 완료")

    # ── 시각화: 3열 (원본 | 예측 CAM | 정답 CAM) ──────────────────────────
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Grad-CAM  ·  {args.model}  ·  오분류 {n}장\n'
        '(좌) 원본    (중) 예측 클래스 CAM    (우) 정답 클래스 CAM',
        fontsize=11, fontweight='bold', y=1.01
    )

    for i, (img_t, true_idx, pred_idx) in enumerate(failures):
        x        = img_t.unsqueeze(0).to(device)
        cam_pred = gcam(x, pred_idx)   # 모델이 틀린 이유
        cam_true = gcam(x, true_idx)   # 정답 클래스를 어디서 봤어야 하나
        orig     = denorm(img_t)

        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f'True: {true_name}', fontsize=8, color='green', pad=3)

        axes[i, 1].imshow(overlay(orig, cam_pred))
        axes[i, 1].set_title(f'Pred: {pred_name}', fontsize=8, color='red', pad=3)

        axes[i, 2].imshow(overlay(orig, cam_true))
        axes[i, 2].set_title(f'True CAM: {true_name}', fontsize=8, color='green', pad=3)

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f'{args.model}_gradcam.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장 완료 → {out_path}")


if __name__ == '__main__':
    main()
