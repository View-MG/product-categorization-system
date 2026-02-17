import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    seed: int = 42
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


def _alloc_counts(n: int, train_f: float, val_f: float, test_f: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    val = max(1, int(round(n * val_f)))
    test = max(1, int(round(n * test_f)))
    if val + test >= n:
        val = 1
        test = 1
    train = n - val - test
    if train <= 0:
        train = max(1, n - 2)
        val = 1 if n - train >= 1 else 0
        test = n - train - val
    return train, val, test


def split_by_barcode(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, Dict]:
    if "barcode" not in df.columns or "label_coarse" not in df.columns:
        raise ValueError("df must contain barcode and label_coarse")

    rng = np.random.default_rng(cfg.seed)

    pairs = df[["barcode", "label_coarse"]].drop_duplicates()
    barcode_to_label = dict(zip(pairs["barcode"], pairs["label_coarse"]))

    split_map: Dict[str, str] = {}

    for lbl, g in pairs.groupby("label_coarse"):
        barcodes = g["barcode"].tolist()
        rng.shuffle(barcodes)

        n = len(barcodes)
        n_train, n_val, n_test = _alloc_counts(n, cfg.train_frac, cfg.val_frac, cfg.test_frac)

        train_ids = barcodes[:n_train]
        val_ids = barcodes[n_train : n_train + n_val]
        test_ids = barcodes[n_train + n_val : n_train + n_val + n_test]

        for b in train_ids:
            split_map[b] = "train"
        for b in val_ids:
            split_map[b] = "val"
        for b in test_ids:
            split_map[b] = "test"

    out = df.copy()
    out["split"] = out["barcode"].map(split_map).fillna("train")

    splits = {"train": [], "val": [], "test": []}
    for b, s in split_map.items():
        splits[s].append(b)

    meta = {
        "seed": cfg.seed,
        "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac},
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "barcode_label": barcode_to_label,
    }
    return out, meta


def save_splits_json(meta: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
