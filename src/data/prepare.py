"""
src/data/prepare.py

Purpose:
- Load metadata from CSV.
- Normalize barcode.
- Build image_path (relative) and abs_path (for validation).
- Basic cleaning (required fields, label filtering, dedup, cap).
- Build train-ready manifest by:
  - dropping unwanted columns
  - removing rows with "" in required fields
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MANIFEST_DROP_COLS = ["image_id","split", "img_ok", "w", "h", "file_size", "image_url", "source", "license_db", "license_images"]


def norm_barcode(x: object) -> str:
    """
    Normalize a barcode into a 13-digit numeric string.

    Why:
    - OFF codes can include non-digits or inconsistent formatting.
    - We strip non-digits and zero-pad to 13 digits for consistent grouping/splitting.
    """
    s = re.sub(r"\D", "", str(x or ""))
    return s.zfill(13) if s else ""


def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Load metadata from CSV into a DataFrame."""
    return pd.read_csv(meta_path)


def add_paths(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """
    Create:
    - image_path: relative path used by training/data loader (portable)
    - abs_path: absolute local path used by validate_images()

    Notes:
    - If metadata has image_id, we convert it to image_path.
    - Base directory is raw_dir/images if exists, otherwise raw_dir.
    """
    raw_dir = Path(raw_dir)
    images_dir = raw_dir / "images"
    if not images_dir.exists():
        images_dir = raw_dir

    out = df.copy()

    # barcode
    out["barcode"] = out.get("barcode", "").map(norm_barcode)

    # image_path (prefer existing image_path, fallback to image_id)
    if "image_path" in out.columns:
        src = out["image_path"]
    elif "image_id" in out.columns:
        src = out["image_id"]
    else:
        raise ValueError("metadata must contain image_id or image_path")

    # normalize to a clean RELATIVE path
    rel = (
        src.astype("string")
        .fillna("")
        .str.strip()
        .str.replace("\\", "/", regex=False)  # unify
        .str.lstrip("/")                     # force relative
    )

    out["image_path"] = rel

    # abs_path for validation (OS-specific)
    rel_os = rel.str.replace("/", os.sep, regex=False).str.lstrip("\\/")
    out["abs_path"] = rel_os.map(lambda r: str(images_dir / r) if r else "")

    return out


def basic_clean(
    df: pd.DataFrame,
    labels: Optional[list] = None,
    dedup_by_barcode: bool = True,
    cap_per_label: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Basic cleaning:
    - Ensure required columns exist
    - Remove empty required fields
    - Optional label filtering
    - Drop duplicates by abs_path
    - dedup by barcode (keep 1 image per product)
    - cap per label
    """
    out = df.copy()

    for col in ["barcode", "image_path", "abs_path", "label_coarse"]:
        if col not in out.columns:
            raise ValueError(f"metadata must contain {col}")

    out["label_coarse"] = out["label_coarse"].astype("string").fillna("").str.strip()
    out["barcode"] = out["barcode"].astype("string").fillna("").str.strip()
    out["image_path"] = out["image_path"].astype("string").fillna("").str.strip()
    out["abs_path"] = out["abs_path"].astype("string").fillna("").str.strip()

    # remove empty required fields
    out = out[(out["barcode"] != "") & (out["image_path"] != "") & (out["abs_path"] != "") & (out["label_coarse"] != "")]

    if labels:
        out = out[out["label_coarse"].isin(labels)]

    out = out.drop_duplicates(subset=["abs_path"], keep="first")

    if dedup_by_barcode:
        out = out.sort_values(["barcode", "label_coarse", "image_path"])
        out = out.drop_duplicates(subset=["barcode"], keep="first")

    if cap_per_label is not None:
        rng = np.random.default_rng(seed)
        kept = []
        for lbl, g in out.groupby("label_coarse"):
            if len(g) <= cap_per_label:
                kept.append(g)
            else:
                idx = rng.choice(g.index.to_numpy(), size=cap_per_label, replace=False)
                kept.append(out.loc[idx])
        out = pd.concat(kept, ignore_index=True)

    return out.reset_index(drop=True)


def attach_label_map(labels: list) -> dict:
    """Create label -> integer id mapping."""
    return {lbl: i for i, lbl in enumerate(labels)}


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final train-ready manifest:
    - drop all columns except MANIFEST_CLEAN_COLS (drop-columns approach)
    """
    out = df.copy()

    # drop everything else (this is the "drop columns" approach)
    drop_cols = [c for c in MANIFEST_DROP_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
        
    return out.reset_index(drop=True)
