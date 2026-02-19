"""
logger.py

Purpose:
- CSV-based metric logger (one row per epoch per split).
- Plotting utilities for train/val loss and val accuracy curves.
- Confusion matrix heatmap as a PNG.

All outputs are written to a run-specific directory (passed at construction).
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class CSVLogger:
    """
    Append-mode CSV logger for epoch metrics.

    File layout (columns):
        epoch | split | loss | accuracy | f1_macro | f1_<class0> | ...

    Example
    -------
    >>> logger = CSVLogger(run_dir / "metrics.csv")
    >>> logger.log(epoch=1, split="train", loss=0.9, accuracy=0.6, f1_macro=0.58)
    >>> logger.log(epoch=1, split="val",   loss=0.7, accuracy=0.72, f1_macro=0.70)
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._header_written = self.path.exists()

    def log(self, epoch: int, split: str, **metrics: float) -> None:
        """Append one row.  *metrics* are arbitrary key=float pairs."""
        row = {"epoch": epoch, "split": split, **metrics}
        write_header = not self._header_written
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_loss_curves(csv_path: Path, out_path: Path) -> None:
    """
    Plot train_loss vs val_loss over epochs and save as PNG.

    Reads the CSV written by CSVLogger.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[logger] matplotlib not installed — skipping loss curve plot.")
        return

    rows = _read_csv(csv_path)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows   = [r for r in rows if r["split"] == "val"]

    if not train_rows or not val_rows:
        return

    epochs     = [int(r["epoch"]) for r in train_rows]
    train_loss = [float(r["loss"]) for r in train_rows]
    val_loss   = [float(r["loss"]) for r in val_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss,   label="val_loss",   marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_accuracy_curve(csv_path: Path, out_path: Path) -> None:
    """
    Plot val accuracy and val F1-macro over epochs and save as PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[logger] matplotlib not installed — skipping accuracy curve plot.")
        return

    rows = _read_csv(csv_path)
    val_rows = [r for r in rows if r["split"] == "val"]

    if not val_rows:
        return

    epochs   = [int(r["epoch"]) for r in val_rows]
    accuracy = [float(r["accuracy"]) for r in val_rows]
    f1       = [float(r["f1_macro"]) for r in val_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, accuracy, label="val_accuracy", marker="o", markersize=3)
    ax.plot(epochs, f1,       label="val_f1_macro", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Val Accuracy & F1-Macro")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a confusion matrix heatmap as PNG.

    Parameters
    ----------
    cm : np.ndarray
        Square confusion matrix (rows = true, cols = predicted).
    class_names : list[str]
        Label names in the same order as matrix indices.
    out_path : Path
        Destination file (.png).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[logger] matplotlib not installed — skipping confusion matrix plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
