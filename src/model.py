"""
model.py
Defines the transfer-learning model: a pretrained ResNet18 backbone
with a new classification head trained on our plant disease classes.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Loads a pretrained ResNet18 and replaces the final fully-connected
    layer to match our number of classes.

    freeze_backbone=True freezes all convolutional layers so only the
    new head is trained initially (fast, good for small datasets).
    Set to False later for fine-tuning the whole network at a low LR.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module):
    """Call this after a few epochs to fine-tune the full network."""
    for param in model.parameters():
        param.requires_grad = True
    return model
