import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, output):
        self.activations = output.detach()

    def _bwd(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1
        out.backward(gradient=one_hot)

        w   = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        vmin, vmax = cam.min(), cam.max()
        return (cam - vmin) / (vmax - vmin + 1e-8)


def get_target_layer(model, model_name):
    if model_name == 'resnet50':
        return model.layer4[-1]
    return model.features[-1]   # efficientnet_b0


def denorm(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)


def make_overlay_pil(img_np, cam, alpha=0.45):
    """numpy 원본 이미지 + CAM → PIL Image"""
    heat    = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(img_np * (1 - alpha) + heat * alpha, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))
