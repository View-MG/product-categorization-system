"""
factory.py

Purpose:
- Provide a central model registry so callers can select an architecture by name.
- Supported names: "simple_cnn", "resnet18", "mobilenetv2"
- For pre-trained backbones (ResNet-18, MobileNet-V2) the final fully-connected
  layer is replaced to match num_classes.
- SimpleCNN is trained from scratch; its architecture is compact and well-suited
  for the 4-class, ~224×224 product-image dataset.

Usage
-----
    from src.models.factory import build_model

    model = build_model("resnet18", num_classes=4, freeze_backbone=True)
    model = build_model("mobilenetv2", num_classes=4, freeze_backbone=True)
    model = build_model("simple_cnn", num_classes=4)
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights

# ── SimpleCNN ─────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Lightweight CNN baseline trained from scratch.

    Architecture
    ------------
    3 × (Conv 3×3 → BN → ReLU → MaxPool 2×2)  +  2 × FC with Dropout

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes) logits

    Parameter count: ~1.2 M  — fast to train, useful lower-bound baseline.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 112×112

            # Block 2: 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 56×56

            # Block 3: 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 28×28

            # Block 4: 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 14×14
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))      # → 256×4×4 = 4096

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def freeze_backbone(self) -> None:
        """No-op: SimpleCNN has no pre-trained backbone to freeze."""

    def unfreeze_backbone(self) -> None:
        """No-op: SimpleCNN has no pre-trained backbone."""

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> dict:
        total = self.total_params()
        trainable = self.trainable_params()
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save(self, path) -> None:
        import torch
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "num_classes": self.num_classes}, path)

    def __repr__(self) -> str:
        return (
            f"SimpleCNN("
            f"num_classes={self.num_classes}, "
            f"trainable_params={self.trainable_params():,}"
            f")"
        )


# ── ResNet-18 wrapper ─────────────────────────────────────────────────────────

class _TransferModel(nn.Module):
    """
    Thin wrapper around a torchvision backbone that exposes the same
    interface as ProductClassifier (freeze/unfreeze, save, param_summary).
    """

    def __init__(self, backbone: nn.Module, num_classes: int, freeze_backbone: bool) -> None:
        super().__init__()
        self._backbone = backbone
        self.num_classes = num_classes
        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._backbone_frozen and mode:
            # Force backbone BN layers back to eval mode to preserve running stats
            for m in self._backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self._backbone.named_parameters():
            if not name.startswith("fc.") and not name.startswith("classifier."):
                param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for param in self._backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> dict:
        total = self.total_params()
        trainable = self.trainable_params()
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save(self, path) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "num_classes": self.num_classes}, path)

    def __repr__(self) -> str:
        name = type(self._backbone).__name__
        return (
            f"{name}Wrapper("
            f"num_classes={self.num_classes}, "
            f"frozen={self._backbone_frozen}, "
            f"trainable_params={self.trainable_params():,}"
            f")"
        )


def _build_resnet18(num_classes: int, freeze_backbone: bool, dropout: float) -> _TransferModel:
    """
    ResNet-18 with ImageNet-1K pre-trained weights.
    Final FC replaced with Dropout → Linear(num_classes).
    """
    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = backbone.fc.in_features          # 512 for ResNet-18
    backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    nn.init.xavier_uniform_(backbone.fc[1].weight)
    nn.init.zeros_(backbone.fc[1].bias)
    return _TransferModel(backbone, num_classes, freeze_backbone)


def _build_mobilenetv2(num_classes: int, freeze_backbone: bool, dropout: float) -> _TransferModel:
    """
    MobileNet-V2 with ImageNet-1K pre-trained weights.
    Final classifier replaced with Dropout → Linear(num_classes).
    """
    backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = backbone.classifier[1].in_features   # 1280 for MobileNet-V2
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    nn.init.xavier_uniform_(backbone.classifier[1].weight)
    nn.init.zeros_(backbone.classifier[1].bias)
    return _TransferModel(backbone, num_classes, freeze_backbone)


# ── Public factory ────────────────────────────────────────────────────────────

_REGISTRY = {
    "simple_cnn":  lambda nc, fb, do: SimpleCNN(num_classes=nc, dropout=do),
    "resnet18":    _build_resnet18,
    "mobilenetv2": _build_mobilenetv2,
}

ModelName = Literal["simple_cnn", "resnet18", "mobilenetv2"]


def build_model(
    name: str,
    num_classes: int = 4,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build a model by name.

    Parameters
    ----------
    name : str
        Architecture name. One of: ``"simple_cnn"``, ``"resnet18"``, ``"mobilenetv2"``.
    num_classes : int
        Number of output classes. Default = 4.
    freeze_backbone : bool
        If True, freeze all backbone parameters except the head.
        Ignored for ``simple_cnn`` (no pre-trained backbone).
    dropout : float
        Dropout probability in the classification head.

    Returns
    -------
    nn.Module
        Initialised model (not moved to device yet).

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](num_classes, freeze_backbone, dropout)


def available_models() -> list:
    """Return list of registered model names."""
    return list(_REGISTRY.keys())
