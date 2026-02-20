"""
scripts/prepare_dataset.py

Purpose:
- End-to-end dataset preparation pipeline for image classification.
- Downloads a raw .tar from Hugging Face, extracts it, loads metadata.csv,
  canonicalizes columns, cleans rows, validates images, and performs a barcode-level split
  to avoid data leakage.

Outputs (under data_local/processed/<tar_stem>/):
- manifest_clean.csv  : minimal manifest for training/testing
- metadata_min.csv    : minimal metadata (barcode, product_name, categories_tags_en)
- splits.json         : split assignment by barcode (reproducible)
- label_map.json      : label -> integer id mapping
- stats.json          : dataset summary statistics
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

from src.config.data_config import DataConfig
from src.data.loader import download_raw_tar, ensure_extracted
from src.data.prepare import load_metadata, add_paths, basic_clean, attach_label_map, build_manifest
from src.data.validate import validate_images, keep_only_ok
from src.data.split import SplitConfig, split_by_barcode, save_splits_json
from src.data.stats import compute_stats, save_stats

def main() -> None:
    """
    Run the full preparation pipeline.
    """
    
    cfg = DataConfig()

    # --- Build paths and ensure output dirs exist ---
    p = cfg.paths()
    p["raw_dir"].mkdir(parents=True, exist_ok=True)
    p["proc_dir"].mkdir(parents=True, exist_ok=True)

    # --- Download + extract raw dataset tar ---
    raw_tar = download_raw_tar(
        repo_id=cfg.repo_id,
        path_in_repo=cfg.raw_tar_in_repo,
        repo_type=cfg.repo_type,
        revision=cfg.revision,
        token=cfg.token,
    )

    ensure_extracted(raw_tar, p["raw_dir"])

    if not p["raw_metadata"].exists():
        raise FileNotFoundError(f"metadata.csv not found at {p['raw_metadata']}")

    df = load_metadata(p["raw_metadata"])
    df = add_paths(df, p["raw_dir"])

    df = basic_clean(
        df,
        labels=cfg.labels,
        dedup_by_barcode=cfg.dedup_by_barcode,
        cap_per_label=cfg.cap_per_label,
        seed=cfg.seed,
    )

    df = validate_images(
        df,
        min_side=cfg.min_side,
        do_verify=cfg.do_verify,
        num_workers=cfg.num_workers,
    )

    df = keep_only_ok(df)

    split_cfg = SplitConfig(
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )
    df, split_meta = split_by_barcode(df, split_cfg)

    manifest = build_manifest(df)
    manifest.to_csv(p["manifest_clean"], index=False)
    save_splits_json(split_meta, p["splits"])

    label_map = attach_label_map(cfg.labels)
    p["label_map"].write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = compute_stats(df)
    save_stats(stats, p["stats"])

    print("OK")
    print(p["manifest_clean"])
    print(p["splits"])
    print(p["stats"])


if __name__ == "__main__":
    main()
