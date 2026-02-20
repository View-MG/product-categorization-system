"""
dataset.py

Purpose:
- Provide a PyTorch Dataset that wraps manifest_clean.csv + label_map.json
  produced by scripts/prepare_dataset.py.
- Returns (image_tensor, label_int) pairs ready for DataLoader.

Expected CSV columns (from prepare.py / validate.py pipeline):
    abs_path, label_coarse, split, barcode, img_ok, w, h, file_size

Expected label_map.json format  (from prepare.py::attach_label_map):
    {"beverages": 0, "snacks": 1, "dry_food": 2, "non_food": 3}
"""

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ProductDataset(Dataset):
    """
    PyTorch Dataset for product package image classification.

    Parameters
    ----------
    manifest : pd.DataFrame | Path | str
        Either a pre-loaded DataFrame or a path to manifest_clean.csv.
        The DataFrame must contain columns: ``abs_path``, ``label_coarse``.
    label_map : Dict[str, int] | Path | str
        Either a pre-built {class_name: int} dict or a path to label_map.json.
    split : str | None
        If not None, filter the manifest to rows where ``split == split``.
        Typical values: ``"train"``, ``"val"``, ``"test"``.
    transform : Callable | None
        torchvision transform pipeline applied to the PIL image.
        Use ``get_train_transforms()`` / ``get_val_transforms()`` from transforms.py.

    Returns  (via __getitem__)
    -------
    image  : torch.FloatTensor  shape (C, H, W) after transform
    label  : torch.LongTensor   scalar integer class index
    """

    def __init__(
        self,
        manifest: Union[pd.DataFrame, Path, str],
        label_map: Union[Dict[str, int], Path, str],
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        # ── 1. Load manifest ──────────────────────────────────────────────
        manifest_path = manifest if isinstance(manifest, (str, Path)) else None
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(manifest)

        if "abs_path" not in manifest.columns:
            raise ValueError("manifest must contain column 'abs_path'")
        if "label_coarse" not in manifest.columns:
            raise ValueError("manifest must contain column 'label_coarse'")

        # ── 1b. Filter out corrupted images ───────────────────────────────
        if "img_ok" in manifest.columns:
            manifest = manifest[manifest["img_ok"] == True].copy()
        elif "file_size" in manifest.columns:
            manifest = manifest[manifest["file_size"] > 0].copy()

        # ── 2. Filter by split ────────────────────────────────────────────
        if split is not None:
            if "split" not in manifest.columns:
                if manifest_path is not None:
                    splits_path = Path(manifest_path).parent / "splits.json"
                    if splits_path.exists():
                        splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
                        barcodes = set(splits_data.get("splits", {}).get(split, []))
                        
                        # We need 'barcode' as a string without '.0' etc
                        # Pandas sometimes reads big numbers as float if there are NaNs
                        manifest["_tmp_bc"] = manifest.get("barcode", "").astype(str).str.replace(r"\.0$", "", regex=True)
                        manifest = manifest[manifest["_tmp_bc"].isin(barcodes)].copy()
                        manifest = manifest.drop(columns=["_tmp_bc"])
                    else:
                        raise ValueError(f"split '{split}' requested, no 'split' column, and no splits.json found at {splits_path}")
                else:
                    raise ValueError(
                        f"split='{split}' requested but manifest has no 'split' column. "
                        "Run scripts/prepare_dataset.py first."
                    )
            else:
                manifest = manifest[manifest["split"] == split].copy()
                
            if len(manifest) == 0:
                raise ValueError(
                    f"No rows found for split='{split}'. "
                    "Check that prepare_dataset.py completed successfully."
                )

        self._df = manifest.reset_index(drop=True)

        # ── 3. Load label_map ─────────────────────────────────────────────
        if isinstance(label_map, (str, Path)):
            label_map = json.loads(Path(label_map).read_text(encoding="utf-8"))

        self._label_map: Dict[str, int] = label_map

        # Pre-validate that every label in manifest is known
        unknown = set(self._df["label_coarse"].unique()) - set(self._label_map.keys())
        if unknown:
            raise ValueError(
                f"Labels found in manifest but missing from label_map: {unknown}"
            )

        self.transform = transform
        self.classes: list = sorted(self._label_map, key=self._label_map.get)  # type: ignore[arg-type]
        self.num_classes: int = len(self._label_map)

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]

        # ── Load image (PIL, RGB) ────────────────────────────────────────
        img_path = Path(str(row["abs_path"]))
        try:
            image: Image.Image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open image at index {idx}: {img_path}"
            ) from exc

        # ── Apply transform pipeline ─────────────────────────────────────
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback: at minimum convert PIL → tensor if no transform supplied
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(image)  # type: ignore[assignment]

        # ── Encode label ─────────────────────────────────────────────────
        label_int: int = self._label_map[row["label_coarse"]]
        label = torch.tensor(label_int, dtype=torch.long)

        return image, label  # type: ignore[return-value]

    # ── Convenience helpers ───────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ProductDataset("
            f"n={len(self)}, "
            f"split={self._df['split'].unique().tolist() if 'split' in self._df.columns else 'N/A'}, "
            f"classes={self.classes}"
            f")"
        )

    @property
    def label_map(self) -> Dict[str, int]:
        return dict(self._label_map)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_datasets(
    manifest_path: Union[Path, str],
    label_map_path: Union[Path, str],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Dict[str, "ProductDataset"]:
    """
    Build train / val / test datasets in one call.

    Example
    -------
    >>> from src.data.transforms import get_train_transforms, get_val_transforms
    >>> from src.config.data_config import DataConfig
    >>> cfg = DataConfig()
    >>> p = cfg.paths()
    >>> datasets = build_datasets(
    ...     p["manifest_clean"], p["label_map"],
    ...     train_transform=get_train_transforms(),
    ...     val_transform=get_val_transforms(),
    ... )
    >>> datasets["train"], datasets["val"], datasets["test"]
    """
    manifest = pd.read_csv(manifest_path)
    
    if "split" not in manifest.columns:
        splits_path = Path(manifest_path).parent / "splits.json"
        if splits_path.exists():
            splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
            barcode_to_split = {}
            for sp, bcs in splits_data.get("splits", {}).items():
                for bc in bcs:
                    barcode_to_split[str(bc)] = sp
            
            manifest["_tmp_bc"] = manifest.get("barcode", "").astype(str).str.replace(r"\.0$", "", regex=True)
            manifest["split"] = manifest["_tmp_bc"].map(barcode_to_split)
            manifest = manifest.drop(columns=["_tmp_bc"])
            manifest = manifest.dropna(subset=["split"])

    label_map: Dict[str, int] = json.loads(
        Path(label_map_path).read_text(encoding="utf-8")
    )

    ds: Dict[str, ProductDataset] = {}
    for split_name in ("train", "val", "test"):
        transform = train_transform if split_name == "train" else val_transform
        ds[split_name] = ProductDataset(
            manifest=manifest,
            label_map=label_map,
            split=split_name,
            transform=transform,
        )
    return ds