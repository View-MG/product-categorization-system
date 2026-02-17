import re
from pathlib import Path
from typing import Optional, Tuple

import os

import numpy as np
import pandas as pd


def norm_barcode(x: object) -> str:
    s = re.sub(r"\D", "", str(x or ""))
    return s.zfill(13) if s else ""


def load_metadata(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    return df


def add_paths(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        images_dir = raw_dir

    df = df.copy()
    df["barcode"] = df["barcode"].map(norm_barcode)
    df["image_id"] = df["image_id"].astype("string").fillna("").str.strip()

    # ทำให้ "/" กลายเป็น separator ของเครื่อง + กันเคสขึ้นต้นด้วย / หรือ \
    rel = (
        df["image_id"]
        .str.replace("/", os.sep, regex=False)
        .str.lstrip("\\/")
    )

    df["abs_path"] = rel.map(lambda r: str(images_dir / r))

    return df



def basic_clean(
    df: pd.DataFrame,
    labels: Optional[list] = None,
    dedup_by_barcode: bool = True,
    cap_per_label: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    df = df.copy()

    if "label_coarse" not in df.columns:
        raise ValueError("metadata must contain label_coarse")

    df["label_coarse"] = df["label_coarse"].astype(str).str.strip()

    df = df[df["barcode"].astype(str).str.len() > 0]
    df = df[df["image_id"].astype(str).str.len() > 0]
    df = df[df["abs_path"].astype(str).str.len() > 0]

    if labels:
        df = df[df["label_coarse"].isin(labels)]

    df = df.drop_duplicates(subset=["abs_path"], keep="first")

    if dedup_by_barcode:
        df = df.sort_values(["barcode", "label_coarse", "image_id"])
        df = df.drop_duplicates(subset=["barcode"], keep="first")

    if cap_per_label is not None:
        rng = np.random.default_rng(seed)
        kept = []
        for lbl, g in df.groupby("label_coarse"):
            if len(g) <= cap_per_label:
                kept.append(g)
            else:
                idx = rng.choice(g.index.to_numpy(), size=cap_per_label, replace=False)
                kept.append(df.loc[idx])
        df = pd.concat(kept, ignore_index=True)

    df = df.reset_index(drop=True)
    return df


def attach_label_map(labels: list) -> dict:
    return {lbl: i for i, lbl in enumerate(labels)}
