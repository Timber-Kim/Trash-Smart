#평가에 필요한 함수는 여기에 추가, 아래는 뼈대만 잡아놓음

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 없는 환경에서도 저장 가능하게
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from models import get_model

# 튜토리얼: device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 튜토리얼 "Visualizing the model predictions" — 최소 참고, 원형 유지
def visualize_model(model, dataloaders, class_names, num_images=6, save_path=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]}  /  true: {class_names[labels[j]]}')
                # 튜토리얼의 imshow를 인라인으로 처리 (ImageNet 정규화 역변환)
                img = inputs.cpu().data[j].numpy().transpose(1, 2, 0)
                img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
                ax.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    if save_path:
                        plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    return
    model.train(mode=was_training)


def evaluate(model, dataloader, dataset_size, class_names):
    """테스트셋 전체 추론 → accuracy, per-class F1, confusion matrix 반환"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / dataset_size
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, macro_f1, report, cm


def measure_inference_time(model, dataloader, num_batches=50):
    """이미지 1장당 평균 추론 시간(ms) 측정"""
    model.eval()
    total_time = 0
    total_images = 0

    if device == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            start = time.perf_counter()
            model(inputs)
            if device == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start
            total_images += inputs.size(0)

    return (total_time / total_images) * 1000  # ms


def count_params(model):
    """총 파라미터 수와 학습 가능한 파라미터 수 반환"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description='모델 평가 및 비교')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='평가할 .pth 파일 경로 (여러 개 가능)')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Test set: {dataset_sizes['test']} images, {len(class_names)} classes\n")

    results = []

    for ckpt_path in args.checkpoints:
        # checkpoint 파일명에서 모델명 추출 (예: resnet50_best.pth → resnet50)
        model_name = os.path.basename(ckpt_path).replace('_best.pth', '')
        print(f"{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        model = get_model(model_name=model_name, num_classes=len(class_names))
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)

        accuracy, macro_f1, report, cm = evaluate(
            model, dataloaders['test'], dataset_sizes['test'], class_names
        )
        inf_time = measure_inference_time(model, dataloaders['test'])
        total_params, trainable_params = count_params(model)

        print(f"Top-1 Accuracy : {accuracy:.4f}")
        print(f"Macro F1 Score : {macro_f1:.4f}")
        print(f"Inference Time : {inf_time:.2f} ms/image")
        print(f"Parameters     : {total_params:,} total / {trainable_params:,} trainable")
        print(f"\nPer-class Report:\n{report}")

        cm_path = os.path.join(args.output_dir, f'{model_name}_confusion_matrix.png')
        save_confusion_matrix(cm, class_names, cm_path, title=f'{model_name} Confusion Matrix')

        visualize_model(model, dataloaders, class_names,
                        save_path=os.path.join(args.output_dir, f'{model_name}_samples.png'))

        results.append({
            'model': model_name,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'inf_time_ms': inf_time,
            'total_params': total_params,
        })

    # 두 모델 비교 요약
    if len(results) > 1:
        print(f"\n{'='*50}")
        print("모델 비교 요약")
        print(f"{'='*50}")
        print(f"{'모델':<20} {'Accuracy':>10} {'Macro F1':>10} {'Time(ms)':>10} {'Params':>12}")
        print('-' * 65)
        for r in results:
            print(f"{r['model']:<20} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
                  f"{r['inf_time_ms']:>10.2f} {r['total_params']:>12,}")


if __name__ == '__main__':
    main()
