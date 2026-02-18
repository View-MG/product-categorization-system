"""
data_config.py

Purpose:
- Hold static configuration for dataset preparation.
- Only secret read from environment is HF_TOKEN.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    
    # --- Hugging Face settings ---
    repo_id: str = "Phathanan/product-categorization-system"
    repo_type: str = "dataset"
    raw_tar_in_repo: str = "data/raw/data_v1.tar"
    token: Optional[str] = None
    revision: Optional[str] = None

    # --- Local paths ---
    dataset_dir: Path = Path("data_local")
    raw_extract_dirname: str = "raw_extracted"
    processed_dirname: str = "processed"
    raw_metadata_name: str = "metadata.csv"

    labels: List[str] = field(default_factory=lambda: ["beverages", "snacks", "dry_food", "other"])
    dedup_by_barcode: bool = False
    cap_per_label: Optional[int] = None
    
    # --- Image validation ---
    min_side: int = 128
    do_verify: bool = False
    num_workers: int = 8

    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    def paths(self) -> dict[str, Path]:
        """
        Build all important local paths used by the pipeline.

        Output structure (local):
        data/
          raw_extracted/<tar_stem>/...
          processed/<tar_stem>/manifest_clean.csv, splits.json, stats.json, label_map.json
        """
        
        tar_stem = Path(self.raw_tar_in_repo).name
        # e.g. raw_v1.tar -> raw_v1
        if tar_stem.endswith(".tar"):
            tar_stem = tar_stem[:-4]
        else:
            tar_stem = Path(tar_stem).stem
            
        raw_dir = self.dataset_dir / self.raw_extract_dirname / tar_stem
        proc_dir = self.dataset_dir / self.processed_dirname / tar_stem

        return {
            "raw_dir": raw_dir,
            "proc_dir": proc_dir,
            "raw_metadata": raw_dir / self.raw_metadata_name,
            "manifest_clean": proc_dir / "manifest_clean.csv",
            "splits": proc_dir / "splits.json",
            "stats": proc_dir / "stats.json",
            "label_map": proc_dir / "label_map.json",
        }
