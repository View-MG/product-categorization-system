import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.data_config import DataConfig
from src.data.loader import download_raw_tar, ensure_extracted
from src.data.prepare import load_metadata, add_paths, basic_clean, attach_label_map
from src.data.validate import validate_images, keep_only_ok
from src.data.split import SplitConfig, split_by_barcode, save_splits_json
from src.data.stats import compute_stats, save_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", type=str, default=None)
    ap.add_argument("--raw-tar", type=str, default=None)
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--min-side", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--cap-per-label", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--test-frac", type=float, default=None)
    args = ap.parse_args()

    cfg = DataConfig()
    if args.repo_id:
        cfg.repo_id = args.repo_id
    if args.raw_tar:
        cfg.raw_tar_in_repo = args.raw_tar
    if args.revision:
        cfg.revision = args.revision
    if args.min_side is not None:
        cfg.min_side = args.min_side
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.cap_per_label is not None:
        cfg.cap_per_label = args.cap_per_label
    if args.seed is not None:
        cfg.seed = args.seed
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.val_frac is not None:
        cfg.val_frac = args.val_frac
    if args.test_frac is not None:
        cfg.test_frac = args.test_frac
    if args.verify:
        cfg.do_verify = True

    p = cfg.paths()
    p["raw_dir"].mkdir(parents=True, exist_ok=True)
    p["proc_dir"].mkdir(parents=True, exist_ok=True)

    raw_tar = download_raw_tar(
        repo_id=cfg.repo_id,
        path_in_repo=cfg.raw_tar_in_repo,
        repo_type=cfg.repo_type,
        revision=cfg.revision,
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
