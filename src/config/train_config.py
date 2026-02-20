"""
train_config.py

Purpose:
- Hold all training hyper-parameters in a single dataclass.
- Mirrors the style of data_config.py (dataclass, Path fields, no argparse coupling).
- A thin CLI shim in train.py converts argparse namespace → TrainConfig.

Supported models (via src/models/factory.py):
    "simple_cnn"   — lightweight CNN trained from scratch
    "resnet18"     — ImageNet pre-trained ResNet-18, head replaced
    "mobilenetv2"  — ImageNet pre-trained MobileNet-V2, head replaced
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:

    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "resnet18"
    """One of: simple_cnn | resnet18 | mobilenetv2"""

    freeze_backbone: bool = True
    """Stage 1: freeze backbone, train head only.  Use --no-freeze to disable."""

    dropout: float = 0.3
    """Dropout probability applied in the classification head."""

    # ── Data paths ────────────────────────────────────────────────────────
    manifest: Path = Path("data_local/processed/data_v2/manifest_clean.csv")
    label_map: Path = Path("data_local/processed/data_v2/label_map.json")

    # ── Training hyper-parameters ─────────────────────────────────────────
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # ── LR scheduler ─────────────────────────────────────────────────────
    lr_scheduler: str = "cosine"
    """One of: cosine | step | none"""

    lr_step_size: int = 7
    """StepLR: decrease LR every N epochs (used only when lr_scheduler=step)."""

    lr_gamma: float = 0.1
    """StepLR: LR multiplier per step."""

    # ── Input image size ──────────────────────────────────────────────────
    image_size: int = 224

    # ── Output directories ────────────────────────────────────────────────
    output_dir: Path = Path("outputs")
    """Root directory; run-specific sub-folder is created automatically."""

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Device ───────────────────────────────────────────────────────────
    device: Optional[str] = None
    """cuda | mps | cpu.  None = auto-detect."""

    # ── Derived helpers ───────────────────────────────────────────────────

    def run_dir(self) -> Path:
        """Return (and create) the run-specific output directory."""
        d = self.output_dir / self.model_name
        d.mkdir(parents=True, exist_ok=True)
        return d
