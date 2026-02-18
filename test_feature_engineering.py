"""
test_feature_engineering.py

Smoke tests for the feature engineering pipeline.
Checks transforms, dataset, label encoding, and model forward pass.

Run with:
    .venv/bin/python test_feature_engineering.py
"""

import sys
import json
from pathlib import Path

import torch
from PIL import Image
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MANIFEST  = ROOT / "data_local/processed/data_v1/manifest_clean.csv"
LABEL_MAP = ROOT / "data_local/processed/data_v1/label_map.json"

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

errors: list[str] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS}  {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"{FAIL}  {msg}")
        errors.append(msg)

# ── 1. Transforms output shape ────────────────────────────────────────────────
print("\n── 1. Transforms ─────────────────────────────────────────────────────")
from src.data.transforms import get_train_transforms, get_val_transforms, IMAGENET_MEAN, IMAGENET_STD

dummy = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

train_t = get_train_transforms()
val_t   = get_val_transforms()

train_out = train_t(dummy)
val_out   = val_t(dummy)

check("train transform output shape == (3, 224, 224)", tuple(train_out.shape) == (3, 224, 224),
      str(tuple(train_out.shape)))
check("val transform output shape == (3, 224, 224)",   tuple(val_out.shape)   == (3, 224, 224),
      str(tuple(val_out.shape)))
check("train transform output dtype == float32", train_out.dtype == torch.float32,
      str(train_out.dtype))
check("val transform output dtype == float32",   val_out.dtype   == torch.float32,
      str(val_out.dtype))

# ── 2. Normalization range ─────────────────────────────────────────────────────
print("\n── 2. Normalization ──────────────────────────────────────────────────")
# After ImageNet normalization the range is roughly (-3, 3), NOT [0, 1]
check("val output NOT in [0,1] (i.e. normalization was applied)",
      not (val_out.min() >= 0.0 and val_out.max() <= 1.0),
      f"min={val_out.min():.3f} max={val_out.max():.3f}")
check("val output within plausible normalized range (-4, 4)",
      val_out.min() > -4 and val_out.max() < 4,
      f"min={val_out.min():.3f} max={val_out.max():.3f}")

# Verify constants match spec
check("IMAGENET_MEAN == [0.485, 0.456, 0.406]", IMAGENET_MEAN == [0.485, 0.456, 0.406])
check("IMAGENET_STD  == [0.229, 0.224, 0.225]", IMAGENET_STD  == [0.229, 0.224, 0.225])

# ── 3. Train-only augmentation (stochasticity) ────────────────────────────────
print("\n── 3. Augmentation (train-only stochasticity) ────────────────────────")
torch.manual_seed(0)
a = train_t(dummy)
torch.manual_seed(999)
b = train_t(dummy)
check("train transform produces different outputs on different seeds (stochastic)",
      not torch.allclose(a, b))

val_c = val_t(dummy)
val_d = val_t(dummy)
check("val transform produces identical outputs (deterministic)",
      torch.allclose(val_c, val_d))

# ── 4. Label map ──────────────────────────────────────────────────────────────
print("\n── 4. Label map ──────────────────────────────────────────────────────")
label_map: dict = json.loads(LABEL_MAP.read_text())
expected_classes = {"beverages", "snacks", "dry_food", "other"}
check("label_map has exactly 4 classes",        len(label_map) == 4, str(set(label_map)))
check("label_map keys match expected classes",  set(label_map) == expected_classes,
      str(set(label_map)))
check("label_map values are 0–3",
      set(label_map.values()) == {0, 1, 2, 3}, str(set(label_map.values())))

# ── 5. ProductDataset ──────────────────────────────────────────────────────────
print("\n── 5. ProductDataset ─────────────────────────────────────────────────")
from src.data.dataset import ProductDataset

ds_train = ProductDataset(MANIFEST, LABEL_MAP, split="train", transform=get_train_transforms())
ds_val   = ProductDataset(MANIFEST, LABEL_MAP, split="val",   transform=get_val_transforms())
ds_test  = ProductDataset(MANIFEST, LABEL_MAP, split="test",  transform=get_val_transforms())

check("train split non-empty",  len(ds_train) > 0, f"n={len(ds_train)}")
check("val   split non-empty",  len(ds_val)   > 0, f"n={len(ds_val)}")
check("test  split non-empty",  len(ds_test)  > 0, f"n={len(ds_test)}")

# __getitem__
img, lbl = ds_train[0]
check("__getitem__ returns image tensor (3,224,224)", tuple(img.shape) == (3, 224, 224),
      str(tuple(img.shape)))
check("__getitem__ returns scalar LongTensor label",
      lbl.shape == torch.Size([]) and lbl.dtype == torch.long,
      f"shape={lbl.shape} dtype={lbl.dtype}")
check("label value is in valid range (0–3)", 0 <= lbl.item() <= 3, str(lbl.item()))

# Val determinism end-to-end
img2, lbl2 = ds_val[0]
img3, lbl3 = ds_val[0]
check("val dataset is deterministic (same item twice)", torch.allclose(img2, img3))

# ── 6. Model forward pass ─────────────────────────────────────────────────────
print("\n── 6. Model forward pass ─────────────────────────────────────────────")
from src.models.model import ProductClassifier

model = ProductClassifier(num_classes=4, freeze_backbone=True, pretrained=False)
model.eval()

batch = torch.stack([img, img], dim=0)  # (2, 3, 224, 224)
with torch.no_grad():
    logits = model(batch)

check("model output shape == (2, 4)",    tuple(logits.shape) == (2, 4), str(tuple(logits.shape)))
check("model output dtype == float32",   logits.dtype == torch.float32, str(logits.dtype))
check("model output finite (no NaN/Inf)", torch.isfinite(logits).all().item())

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n──────────────────────────────────────────────────────────────────────")
if errors:
    print(f"\033[91m{len(errors)} test(s) FAILED:\033[0m")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    total = 20  # update if you add checks
    print(f"\033[92mAll checks passed.\033[0m")
    sys.exit(0)
