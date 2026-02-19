"""
prepare_dataset.py

Purpose:
- Build a clean image classification manifest from a raw tar (downloaded from Hugging Face or local file).
- Validate images, remove broken/too-small images, then split data by barcode (group split) to avoid leakage.
- Export: manifest_clean.csv, splits.json, label_map.json, stats.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dotenv import load_dotenv

from src.config.data_config import DataConfig
from src.data.loader import download_raw_tar, ensure_extracted
from src.data.prepare import load_metadata, add_paths, basic_clean, attach_label_map
from src.data.validate import validate_images, keep_only_ok
from src.data.split import SplitConfig, split_by_barcode, save_splits_json
from src.data.stats import compute_stats, save_stats

def main() -> None:
    cfg = DataConfig()

    p = cfg.paths()
    p["raw_dir"].mkdir(parents=True, exist_ok=True)
    p["proc_dir"].mkdir(parents=True, exist_ok=True)

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

    df.to_csv(p["manifest_clean"], index=False)
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
