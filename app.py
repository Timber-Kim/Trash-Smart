import sys
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image

sys.path.append('src')
from models import get_model

CLASS_NAMES_KO = {
    'battery':               '건전지',
    'aluminum_can':          '알루미늄 캔',
    'iron_can':              '철 캔',
    'fluorescent_lamp':      '형광등',
    'glass bottle_brown':    '유리병 (갈색)',
    'glass bottle_colorless':'유리병 (무색)',
    'glass bottle_green':    '유리병 (녹색)',
    'paper':                 '종이',
    'PE':                    'PE (플라스틱)',
    'PP':                    'PP (플라스틱)',
    'PS':                    'PS (플라스틱)',
    'colored':               '페트병 (유색)',
    'colorless':             '페트병 (무색)',
    'styrofoam':             '스티로폼',
    'vinyl':                 '비닐',
}

CLASS_NAMES = list(CLASS_NAMES_KO.keys())

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path: str, model_name: str):
    model = get_model(model_name=model_name, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def predict(image: Image.Image, model):
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3 = torch.topk(probs, 3)
    result = {}
    for score, idx in zip(top3.values, top3.indices):
        label = CLASS_NAMES_KO[CLASS_NAMES[idx.item()]]
        result[label] = round(score.item(), 4)
    return result


def build_app(checkpoint_path: str, model_name: str = 'resnet50'):
    model = load_model(checkpoint_path, model_name)

    def inference(image):
        if image is None:
            return {}
        return predict(image, model)

    demo = gr.Interface(
        fn=inference,
        inputs=gr.Image(type='pil', label='쓰레기 이미지'),
        outputs=gr.Label(num_top_classes=3, label='분류 결과'),
        title='Trash-Smart 재활용 분류기',
        description=f'모델: {model_name} | 이미지를 업로드하거나 웹캠으로 찍어주세요.',
        examples=None,
        live=False,
    )
    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/checkpoints/resnet50_best.pth')
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--share', action='store_true', help='공개 링크 생성 (Colab용)')
    args = parser.parse_args()

    demo = build_app(checkpoint_path=args.checkpoint, model_name=args.model)
    demo.launch(share=args.share)
