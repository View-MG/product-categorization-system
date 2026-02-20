"""
trainer.py

Purpose:
- Self-contained Trainer class that runs the full train → val loop.
- Uses existing dataset/transform infrastructure from src/data/.
- Saves best model checkpoint based on val F1-macro.
- Delegates metric logging to CSVLogger and plot saving to logger.py.

Public API
----------
    trainer = Trainer(model, train_loader, val_loader, cfg, class_names)
    trainer.fit()
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.training.metrics import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
)
from src.training.logger import (
    CSVLogger,
    plot_loss_curves,
    plot_accuracy_curve,
    plot_confusion_matrix,
)


class Trainer:
    """
    Generic training loop for image classification.

    Parameters
    ----------
    model : nn.Module
        Any model from src/models/factory.py (or ProductClassifier).
        Must accept (B, 3, H, W) → (B, num_classes) logits.
    train_loader : DataLoader
        Training split dataloader.
    val_loader : DataLoader
        Validation split dataloader.
    cfg : TrainConfig
        All hyper-parameters and output paths.
    class_names : list[str]
        Ordered list of class label strings (e.g. ["beverages", "snacks", ...]).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig,
        class_names: List[str],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.class_names = class_names

        # ── Device ────────────────────────────────────────────────────────
        self.device = self._resolve_device(cfg.device)
        self.model.to(self.device)

        # ── Optimiser ─────────────────────────────────────────────────────
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # ── LR Scheduler ──────────────────────────────────────────────────
        self.scheduler = self._build_scheduler()

        # ── Logging ───────────────────────────────────────────────────────
        run_dir = cfg.run_dir()
        self.csv_logger = CSVLogger(run_dir / "metrics.csv")
        self.best_ckpt_path = run_dir / "best_checkpoint.pt"

        self._best_f1: float = -1.0

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(requested: Optional[str]) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_scheduler(self):
        sched = self.cfg.lr_scheduler.lower()
        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs
            )
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.lr_step_size,
                gamma=self.cfg.lr_gamma,
            )
        return None  # "none"

    # ── One-epoch loops ───────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                avg = running_loss / (batch_idx + 1)
                print(
                    f"  [epoch {epoch}  batch {batch_idx+1}/{n_batches}]  "
                    f"train_loss={avg:.4f}",
                    flush=True,
                )

        return running_loss / n_batches

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, dict, List[int], List[int]]:
        """Run one validation epoch. Returns (avg_loss, metric_dict, labels, preds)."""
        self.model.eval()
        running_loss = 0.0
        all_labels: List[int] = []
        all_preds: List[int] = []

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            running_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        avg_loss = running_loss / len(self.val_loader)
        metrics = compute_metrics(all_labels, all_preds, self.class_names)

        return avg_loss, metrics, all_labels, all_preds

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _save_best_checkpoint(self, val_f1: float, epoch: int) -> bool:
        """Save checkpoint if val F1-macro improved.  Returns True if saved."""
        if val_f1 > self._best_f1:
            self._best_f1 = val_f1
            self.model.save(self.best_ckpt_path)  # type: ignore[attr-defined]
            print(
                f"  ✓ New best checkpoint saved  "
                f"(val_f1_macro={val_f1:.4f}, epoch={epoch})"
            )
            return True
        return False

    # ── Main fit loop ─────────────────────────────────────────────────────

    def fit(self) -> None:
        """
        Run full training for cfg.epochs epochs.

        After training:
        - Saves loss curve PNG to  <run_dir>/loss_curve.png
        - Saves accuracy curve PNG to  <run_dir>/accuracy_curve.png
        - Saves confusion matrix PNG to  <run_dir>/confusion_matrix.png
        - Prints final classification report
        """
        print(f"\n{'='*60}")
        print(f"  Model    : {self.cfg.model_name}")
        print(f"  Device   : {self.device}")
        print(f"  Epochs   : {self.cfg.epochs}")
        print(f"  LR       : {self.cfg.lr}  scheduler={self.cfg.lr_scheduler}")
        print(f"  Run dir  : {self.cfg.run_dir()}")
        print(f"{'='*60}\n")

        best_val_labels: List[int] = []
        best_val_preds: List[int] = []

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"── Epoch {epoch}/{self.cfg.epochs} ──")

            # Training
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss, val_metrics, val_labels, val_preds = self._val_epoch()

            # LR step
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            self.csv_logger.log(
                epoch=epoch,
                split="train",
                loss=train_loss,
                accuracy=0.0,   # not tracked for train to keep it fast
                f1_macro=0.0,
            )
            self.csv_logger.log(
                epoch=epoch,
                split="val",
                loss=val_loss,
                **val_metrics,
            )

            # Console summary
            print(
                f"  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['f1_macro']:.4f}"
            )

            # Checkpoint
            if self._save_best_checkpoint(val_metrics["f1_macro"], epoch):
                best_val_labels, best_val_preds = val_labels, val_preds

        # ── Post-training outputs ─────────────────────────────────────────
        run_dir = self.cfg.run_dir()
        csv_path = run_dir / "metrics.csv"

        print(f"\n── Generating plots → {run_dir} ──")
        plot_loss_curves(csv_path, run_dir / "loss_curve.png")
        plot_accuracy_curve(csv_path, run_dir / "accuracy_curve.png")

        # Confusion matrix on best val epoch
        cm = get_confusion_matrix(best_val_labels, best_val_preds, self.class_names)
        plot_confusion_matrix(
            cm,
            self.class_names,
            run_dir / "confusion_matrix.png",
            title=f"Confusion Matrix — {self.cfg.model_name}",
        )

        print(f"\n── Final Classification Report (val) ──")
        report = get_classification_report(
            best_val_labels, best_val_preds, self.class_names
        )
        print(report)

        print(f"\n── Confusion Matrix (val) ──")
        _print_confusion_matrix(cm, self.class_names)

        print(f"\nBest val F1-macro : {self._best_f1:.4f}")
        print(f"Best checkpoint   : {self.best_ckpt_path}")
        print(f"Metrics CSV       : {csv_path}")


def _print_confusion_matrix(cm, class_names: List[str]) -> None:
    """Print a plain-text confusion matrix to stdout."""
    col_w = max(12, max(len(n) for n in class_names) + 2)
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row_str = f"{name:<{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(class_names)))
        print(row_str)
