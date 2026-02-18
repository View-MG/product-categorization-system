"""
loader.py

Purpose:
- Download a raw tar from Hugging Face.
- Extract tar safely and cache extraction by a marker file.
"""

import os
import tarfile
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def _extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir.resolve()

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        for m in members:
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tf.extractall(out_dir)


def download_raw_tar(
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    )


def ensure_extracted(raw_tar: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted.ok"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_tar(raw_tar, extract_dir)
    marker.write_text("ok", encoding="utf-8")
