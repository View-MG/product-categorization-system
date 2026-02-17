from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    repo_id: str = "Phathanan/product-categorization-system"
    repo_type: str = "dataset"
    revision: Optional[str] = None
    raw_tar_in_repo: str = "raw_data.tar"

    local_root: Path = Path("data_local")
    raw_subdir: Path = Path("raw/raw_v1")
    proc_subdir: Path = Path("processed/off_v1")

    labels: List[str] = field(default_factory=lambda: ["beverages", "snacks", "dry_food", "other"])

    min_side: int = 128
    do_verify: bool = False
    num_workers: int = 8

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42

    dedup_by_barcode: bool = True
    cap_per_label: Optional[int] = None

    @property
    def raw_dir(self) -> Path:
        return self.local_root / self.raw_subdir

    @property
    def proc_dir(self) -> Path:
        return self.local_root / self.proc_subdir

    @property
    def raw_images_dir(self) -> Path:
        return self.raw_dir / "images"

    @property
    def raw_metadata(self) -> Path:
        return self.raw_dir / "metadata.csv"

    @property
    def manifest_clean(self) -> Path:
        return self.proc_dir / "manifest_clean.csv"

    @property
    def splits_json(self) -> Path:
        return self.proc_dir / "splits.json"

    @property
    def stats_json(self) -> Path:
        return self.proc_dir / "stats.json"

    @property
    def label_map_json(self) -> Path:
        return self.proc_dir / "label_map.json"

    @property
    def validate_cache_csv(self) -> Path:
        return self.proc_dir / "validate_cache.csv"

    def paths(self) -> Dict[str, Path]:
        return {
            "raw_dir": self.raw_dir,
            "proc_dir": self.proc_dir,
            "raw_images_dir": self.raw_images_dir,
            "raw_metadata": self.raw_metadata,
            "manifest_clean": self.manifest_clean,
            "splits": self.splits_json,
            "stats": self.stats_json,
            "label_map": self.label_map_json,
            "validate_cache": self.validate_cache_csv,
        }
