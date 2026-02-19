"""
train.py

Purpose:
- CLI entry point for training baseline image classifiers.
- Wires together: DataConfig → datasets/dataloaders → model factory → Trainer.
- Reuses all existing infrastructure: dataset.py, transforms.py, data_config.py,
  and train_config.py.

Run examples
-----------
    python train.py --model resnet18
    python train.py --model mobilenetv2 --epochs 30 --lr 5e-4
    python train.py --model simple_cnn  --no-freeze --epochs 25
    python train.py --model resnet18    --device cpu
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Project root on sys.path (allows `python train.py` from repo root) ────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config.data_config import DataConfig
from src.config.train_config import TrainConfig
from src.data.dataset import build_datasets
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import build_model, available_models
from src.training.trainer import Trainer


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a baseline image classifier for product categorization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=available_models(),
        help="Architecture to train.",
    )
    p.add_argument(
        "--no-freeze",
        dest="freeze_backbone",
        action="store_false",
        default=True,
        help="Train the full network from epoch 1 (no head-only warm-up).",
    )
    p.add_argument("--dropout", type=float, default=0.3, help="Head dropout probability.")

    # Data
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_local/processed/data_v1/manifest_clean.csv"),
        help="Path to manifest_clean.csv.",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=Path("data_local/processed/data_v1/label_map.json"),
        help="Path to label_map.json.",
    )

    # Training
    p.add_argument("--epochs",       type=int,   default=20,    help="Number of training epochs.")
    p.add_argument("--batch-size",   type=int,   default=32,    help="Batch size.")
    p.add_argument("--num-workers",  type=int,   default=4,     help="DataLoader worker processes.")
    p.add_argument("--lr",           type=float, default=1e-3,  help="Initial learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4,  help="AdamW weight decay.")
    p.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="LR scheduler.",
    )
    p.add_argument("--image-size",   type=int,   default=224,   help="Input image resolution.")

    # Output
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory for run outputs.",
    )

    # Misc
    p.add_argument("--seed",   type=int,           default=42,   help="Random seed.")
    p.add_argument("--device", type=str,           default=None,
                   help="Force device: cuda | mps | cpu.  Default = auto-detect.")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Build TrainConfig from argparse namespace ──────────────────────────
    cfg = TrainConfig(
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        manifest=args.manifest,
        label_map=args.label_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        image_size=args.image_size,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )

    seed_everything(cfg.seed)

    # ── Validate data paths ────────────────────────────────────────────────
    if not cfg.manifest.exists():
        print(
            f"[ERROR] manifest not found: {cfg.manifest}\n"
            "  Run:  python scripts/prepare_dataset.py\n"
            "  to generate the manifest before training."
        )
        sys.exit(1)
    if not cfg.label_map.exists():
        print(f"[ERROR] label_map not found: {cfg.label_map}")
        sys.exit(1)

    # ── Datasets ──────────────────────────────────────────────────────────
    print("[train.py] Loading datasets …")
    datasets = build_datasets(
        manifest_path=cfg.manifest,
        label_map_path=cfg.label_map,
        train_transform=get_train_transforms(size=cfg.image_size),
        val_transform=get_val_transforms(size=cfg.image_size),
    )

    train_ds = datasets["train"]
    val_ds   = datasets["val"]

    print(f"  train : {train_ds}")
    print(f"  val   : {val_ds}")

    class_names = train_ds.classes   # e.g. ["beverages", "dry_food", "other", "snacks"]
    num_classes = train_ds.num_classes

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"[train.py] Building model: {cfg.model_name} …")
    model = build_model(
        name=cfg.model_name,
        num_classes=num_classes,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout,
    )

    # Print parameter summary
    if hasattr(model, "param_summary"):
        ps = model.param_summary()
        print(
            f"  params  total={ps['total']:,}  "
            f"trainable={ps['trainable']:,}  "
            f"frozen={ps['frozen']:,}"
        )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        class_names=class_names,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
