from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _check_one(path: str, min_side: int, do_verify: bool) -> Tuple[int, int, int, int]:
    p = Path(path)
    if not p.exists():
        return 0, 0, 0, 0
    try:
        if do_verify:
            with Image.open(p) as im:
                im.verify()
        with Image.open(p) as im:
            w, h = im.size
        ok = int(min(w, h) >= min_side)
        size = int(p.stat().st_size)
        return ok, w, h, size
    except Exception:
        return 0, 0, 0, 0


def validate_images(
    df: pd.DataFrame,
    min_side: int = 128,
    do_verify: bool = False,
    num_workers: int = 8,
) -> pd.DataFrame:
    if "abs_path" not in df.columns:
        raise ValueError("df must contain abs_path")

    paths = df["abs_path"].astype(str).tolist()
    results = [None] * len(paths)

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futs = {
            ex.submit(_check_one, paths[i], min_side, do_verify): i for i in range(len(paths))
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="validate images"):
            i = futs[fut]
            results[i] = fut.result()

    out = df.copy()
    out["img_ok"] = [r[0] for r in results]
    out["w"] = [r[1] for r in results]
    out["h"] = [r[2] for r in results]
    out["file_size"] = [r[3] for r in results]
    return out


def keep_only_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "img_ok" not in df.columns:
        raise ValueError("df must contain img_ok")
    return df[df["img_ok"] == 1].copy().reset_index(drop=True)
