"""
src/data/loader.py

Purpose:
- Download a raw .tar file from Hugging Face Hub.
- Extract tar safely (path traversal protection).
- Cache extraction with a marker file to avoid repeated work.
"""

import os
import tarfile
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


def _extract_tar(tar_path: Path, out_dir: Path) -> None:
    """
    Safely extract a tar archive into out_dir.

    Security best practice:
    - Prevent path traversal by ensuring every extracted path stays under out_dir.
    """
    
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
    """
    Download a file from Hugging Face Hub (cached by hf_hub_download).

    Token behavior:
    - If token is None, tries environment variable HF_TOKEN.
    - If still None, hf_hub_download may work for public repos.
    """
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
    """
    Extract raw_tar into extract_dir only once using a marker file.
    """
    marker = extract_dir / ".extracted.ok"
    if marker.exists():
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    _extract_tar(raw_tar, extract_dir)
    marker.write_text("ok", encoding="utf-8")
