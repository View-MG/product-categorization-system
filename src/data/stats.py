import json
from pathlib import Path
from typing import Dict

import pandas as pd


def compute_stats(df: pd.DataFrame) -> Dict:
    total = int(len(df))
    by_label = df["label_coarse"].value_counts().to_dict() if "label_coarse" in df.columns else {}
    by_split = df["split"].value_counts().to_dict() if "split" in df.columns else {}

    by_label_split = {}
    if "label_coarse" in df.columns and "split" in df.columns:
        tmp = df.groupby(["label_coarse", "split"]).size().reset_index(name="n")
        by_label_split = {
            r["label_coarse"]: by_label_split.get(r["label_coarse"], {}) | {r["split"]: int(r["n"])}
            for _, r in tmp.iterrows()
        }

    img_ok_rate = None
    if "img_ok" in df.columns:
        img_ok_rate = float(df["img_ok"].mean()) if len(df) else 0.0

    return {
        "total": total,
        "by_label": {k: int(v) for k, v in by_label.items()},
        "by_split": {k: int(v) for k, v in by_split.items()},
        "by_label_split": by_label_split,
        "img_ok_rate": img_ok_rate,
        "columns": list(df.columns),
    }


def save_stats(stats: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
