from __future__ import annotations

from pathlib import Path

import numpy as np

from .processor import load_scan


BACKBONES = ("resnet50", "efficientnet_b0", "convnext_tiny")


def handcrafted_embedding(scan_path: str | Path) -> np.ndarray:
    img = load_scan(scan_path)
    depth_profile = img.mean(axis=1)
    lateral_profile = img.mean(axis=0)
    percentiles = np.percentile(img, [1, 5, 25, 50, 75, 95, 99])
    grad_z = np.abs(np.diff(img, axis=0)).mean()
    grad_x = np.abs(np.diff(img, axis=1)).mean()
    return np.concatenate(
        [
            np.array([img.mean(), img.std(), grad_z, grad_x], dtype=np.float32),
            percentiles.astype(np.float32),
            depth_profile[:: max(1, len(depth_profile) // 32)][:32].astype(np.float32),
            lateral_profile[:: max(1, len(lateral_profile) // 32)][:32].astype(np.float32),
        ]
    )


def cnn_embedding(scan_path: str | Path, backbone: str = "resnet50", pretrained: bool = True) -> np.ndarray:
    if backbone not in BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone}. Choose one of: {', '.join(BACKBONES)}")
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
    except Exception:
        return handcrafted_embedding(scan_path)

    img = load_scan(scan_path)
    tensor = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(tensor).unsqueeze(0)

    try:
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            model.classifier = nn.Identity()
        else:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            model.classifier = nn.Identity()
    except Exception:
        return handcrafted_embedding(scan_path)

    model.eval()
    with torch.no_grad():
        emb = model(tensor).detach().cpu().numpy().reshape(-1)
    return emb.astype(np.float32)
