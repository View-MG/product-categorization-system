"""
train.py

Purpose:
- CLI entry point for training baseline image classifiers.
- Wires together: DataConfig → datasets/dataloaders → model factory → Trainer.
- Reuses all existing infrastructure: dataset.py, transforms.py, data_config.py,
  and train_config.py.

Run examples
-----------
    python train.py --model resnet50
    python train.py --model mobilenetv3_large --epochs 30 --lr 5e-4
    python train.py --model simple_cnn  --no-freeze --epochs 25
    python train.py --model resnet50    --device cpu
"""

import argparse
import json
import random
import sys
import re
from pathlib import Path

import numpy as np
import torch

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import default_data_collator

# ── Project root on sys.path (allows `python train.py` from repo root) ────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.config.data_config import DataConfig
from src.config.train_config import TrainConfig
from src.data.dataset import build_datasets
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import build_model, available_models

from sklearn.metrics import accuracy_score, f1_score


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Versioned Directory Helper ────────────────────────────────────────────────

def get_next_run_dir(base_dir: Path, model_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_tests = []
    pattern = re.compile(rf"^{model_name}_test(\d+)$")
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing_tests.append(int(match.group(1)))
    next_num = max(existing_tests) + 1 if existing_tests else 1
    return base_dir / f"{model_name}_test{next_num}"


# ── Metrics Helper ────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}


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
        default="resnet50",
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
        default=Path("data_local/processed/data_v2/manifest_clean.csv"),
        help="Path to manifest_clean.csv.",
    )
    p.add_argument(
        "--label-map",
        type=Path,
        default=Path("data_local/processed/data_v2/label_map.json"),
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
        default=Path("runs"),
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

    num_classes = train_ds.num_classes

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

    # ── Trainer Base Directory ────────────────────────────────────────────
    run_dir = get_next_run_dir(cfg.output_dir, cfg.model_name)
    print(f"\n[train.py] Output directory for this run: {run_dir}\n")

    # ── Hugging Face Trainer ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        dataloader_num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("[train.py] Starting Hugging Face Trainer loop ...")
    trainer.train()

    # Save the best model officially to the top level of the run directory
    trainer.save_model(str(run_dir / "best_model"))
    
    # Save metrics history
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"\n[train.py] Training complete! Results saved in {run_dir}")

if __name__ == "__main__":
    main()
