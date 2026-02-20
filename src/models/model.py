"""
model.py

Purpose:
    Define ProductClassifier — a fine-tunable image classifier built on top of a
    pre-trained EfficientNet-B0 backbone (ImageNet-1K weights).

Why EfficientNet-B0?
    * Compact (5.3 M params) — well-suited for a 4-class dataset of moderate size.
    * Native input resolution 224×224, matching the transforms pipeline in
      src/data/transforms.py.
    * Strong ImageNet baseline with compound scaling; outperforms ResNet-50 at a
      fraction of the compute.
    * Available in torchvision >= 0.13 with no extra dependencies.

Architecture
------------
    Input  (B, 3, 224, 224) float32  — normalised with ImageNet stats
        │
    EfficientNet-B0 backbone  (pre-trained, optionally frozen)
        │  features[0]  : Conv Stem
        │  features[1–8]: MBConv blocks
        │
    AdaptiveAvgPool2d(1, 1)   → (B, 1280, 1, 1)
        │
    Flatten                   → (B, 1280)
        │
    Dropout(p=0.3)
        │
    Linear(1280 → num_classes)   ← replaced classification head
        │
    Output logits  (B, num_classes)  — pair with nn.CrossEntropyLoss

Two-stage fine-tuning usage
---------------------------
    Stage 1  (head-only):   model = ProductClassifier(freeze_backbone=True)
    Stage 2  (full model):  model.unfreeze_backbone()
    Stage 2b (gradual):     model.unfreeze_last_n_blocks(n=3)
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class ProductClassifier(nn.Module):
    """
    Transfer-learning classifier for product package images.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Must match ``len(label_map)``.
        Default = 4  (beverages, snacks, dry_food, non_food) — matches label_map.json.
    freeze_backbone : bool
        If True, all backbone parameters are frozen so only the new
        classification head is updated during Stage 1 training.
        Call ``unfreeze_backbone()`` to move to full fine-tuning.
    dropout : float
        Dropout probability applied before the final Linear layer.
        EfficientNet-B0 already has internal dropout (p=0.2); this adds a
        second regularisation stage specific to the new head.
    pretrained : bool
        Load ImageNet-1K pre-trained weights for the backbone.
        Set to False only when loading a checkpoint via ``ProductClassifier.load()``.
    """

    #: Output feature dimension of EfficientNet-B0 after avgpool
    BACKBONE_OUT_FEATURES: int = 1280

    def __init__(
        self,
        num_classes: int = 4,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ── 1. Load backbone ──────────────────────────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # ── 2. Strip the built-in classifier, keep feature extractor only ─
        #    backbone.classifier == Sequential(Dropout(0.2), Linear(1280, 1000))
        #    We remove it and manage the downstream head ourselves so the
        #    dropout rate and output dimension are under our control.
        self.backbone: nn.Module = backbone.features   # Conv stem + MBConv blocks
        self.pool:     nn.Module = backbone.avgpool    # AdaptiveAvgPool2d(1, 1)

        # ── 3. New classification head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(self.BACKBONE_OUT_FEATURES, num_classes),
        )

        # ── 4. Initialise head weights ────────────────────────────────────
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

        self.num_classes = num_classes
        self._backbone_frozen = False

        if freeze_backbone:
            self.freeze_backbone()

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape ``(B, 3, 224, 224)``, normalised float32.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(B, num_classes)``. Pass directly to ``nn.CrossEntropyLoss``.
        """
        features = self.backbone(x)          # (B, 1280, 7, 7)
        pooled   = self.pool(features)       # (B, 1280, 1, 1)
        flat     = torch.flatten(pooled, 1)  # (B, 1280)
        logits   = self.classifier(flat)     # (B, num_classes)
        return logits

    # ── Backbone freeze / unfreeze helpers ────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters.

        Use for Stage 1: train the classification head only.
        This is efficient and prevents destroying pre-trained features early on.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters for full fine-tuning (Stage 2).

        Recommended to use a lower learning-rate (e.g. 1e-5) for the backbone
        compared to the head when calling this method.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def unfreeze_last_n_blocks(self, n: int = 3) -> None:
        """
        Unfreeze only the last *n* blocks of the EfficientNet-B0 backbone.

        Provides a gradual intermediate stage between head-only and full
        fine-tuning. Useful when the dataset is small and full fine-tuning
        risks overfitting.

        EfficientNet-B0 backbone structure (``backbone.features``):
            index 0   — Conv2dNormActivation (stem)
            index 1–8 — MBConv blocks (8 blocks total)

        Parameters
        ----------
        n : int
            Number of trailing blocks to unfreeze. Default = 3 (indices 6–8).
        """
        all_blocks = list(self.backbone.children())
        for block in all_blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        self._backbone_frozen = False

    # ── Parameter statistics ──────────────────────────────────────────────

    def trainable_params(self) -> int:
        """Return count of parameters with ``requires_grad=True``."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def param_summary(self) -> Dict[str, int]:
        """Return dict with total, trainable, and frozen parameter counts."""
        total     = self.total_params()
        trainable = self.trainable_params()
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    total - trainable,
        }

    # ── Checkpoint save / load ────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """
        Save model weights and metadata to *path*.

        Saved dict keys: ``model_state_dict``, ``num_classes``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes":      self.num_classes,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        num_classes: Optional[int] = None,
        map_location: str = "cpu",
    ) -> "ProductClassifier":
        """
        Load a checkpoint saved by ``ProductClassifier.save()``.

        Parameters
        ----------
        path : Path
            Path to the ``.pt`` checkpoint file.
        num_classes : int | None
            Override the number of classes stored in the checkpoint.
        map_location : str
            Device string passed to ``torch.load``.

        Returns
        -------
        ProductClassifier
            Model with weights loaded, backbone unfrozen.
        """
        ckpt = torch.load(path, map_location=map_location)
        nc   = num_classes or ckpt["num_classes"]
        model = cls(num_classes=nc, freeze_backbone=False, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    def __repr__(self) -> str:  # type: ignore[override]
        return (
            f"ProductClassifier("
            f"backbone=EfficientNet-B0, "
            f"num_classes={self.num_classes}, "
            f"frozen={self._backbone_frozen}, "
            f"trainable_params={self.trainable_params():,}"
            f")"
        )
