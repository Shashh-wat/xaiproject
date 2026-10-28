"""Model builders for the two baselines: MobileNetV2 and ResNet50.

Both are ImageNet-pretrained backbones with a fresh linear classifier head
sized to num_classes. `get_target_layer()` returns the last conv layer used
as the hook point for Grad-CAM (see src/explain/gradcam.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ClassifierModel(nn.Module):
    """Thin wrapper so Grad-CAM/SHAP/LIME code can be architecture-agnostic."""

    def __init__(self, backbone: nn.Module, target_layer: nn.Module,
                 name: str):
        super().__init__()
        self.backbone = backbone
        self._target_layer = target_layer
        self.name = name

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        return self._target_layer


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def build_model(name: str, num_classes: int,
                 freeze_backbone: bool = True,
                 unfreeze_last_n: int = 0) -> ClassifierModel:
    if name == "mobilenet_v2":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            _freeze(net.features)
            if unfreeze_last_n > 0:
                for block in list(net.features)[-unfreeze_last_n:]:
                    for p in block.parameters():
                        p.requires_grad = True
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
        target_layer = net.features[-1]
        return ClassifierModel(net, target_layer, "mobilenet_v2")

    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for child_name, child in net.named_children():
                if child_name not in ("fc",):
                    _freeze(child)
            if unfreeze_last_n > 0:
                layers = [net.layer4, net.layer3, net.layer2, net.layer1]
                for layer in layers[:unfreeze_last_n]:
                    for p in layer.parameters():
                        p.requires_grad = True
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
        target_layer = net.layer4[-1]
        return ClassifierModel(net, target_layer, "resnet50")

    raise ValueError(f"Unknown model name: {name!r}. "
                      "Expected 'mobilenet_v2' or 'resnet50'.")


def trainable_parameters(model: ClassifierModel, lr_head: float,
                          lr_backbone: float):
    """Two param groups: classifier head at lr_head, any unfrozen backbone
    layers at the (usually much smaller) lr_backbone."""
    head_params, backbone_params = [], []
    head_modules = set()
    if model.name == "mobilenet_v2":
        head_modules = {id(p) for p in model.backbone.classifier.parameters()}
    elif model.name == "resnet50":
        head_modules = {id(p) for p in model.backbone.fc.parameters()}

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_modules:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups
