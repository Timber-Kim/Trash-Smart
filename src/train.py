import os
import sys
import time
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from models import get_model
cudnn.benchmark = True

# 튜토리얼: device 설정 (dataset.py에서 옮겨옴)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, num_epochs, save_path):
    since = time.time()

    # 튜토리얼: best model 저장 (TemporaryDirectory → 실제 경로로 변경)/# Create a temporary directory to save training checkpoints 부분
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def main():
    parser = argparse.ArgumentParser(description='Trash-Smart 모델 학습')
    parser.add_argument('--config', default='configs/config.yaml', help='config 파일 경로')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # dataset.py의 get_dataloaders 호출
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        data_dir=cfg['data']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
        aug_cfg=cfg.get('augmentation'),  # config.yaml의 augmentation 섹션 전달
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    # models.py의 get_model 호출
    model = get_model(
        model_name=cfg['model']['name'],
        num_classes=len(class_names),
        feature_extract=cfg['model']['feature_extract'],
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'], momentum=0.9)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg['training']['step_size'], gamma=0.1)

    save_path = os.path.join('models', 'checkpoints', f"{cfg['model']['name']}_best.pth")

    model = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes,
        num_epochs=cfg['training']['num_epochs'],
        save_path=save_path,
    )
    print(f"Best model saved → {save_path}")


if __name__ == '__main__':
    main()
