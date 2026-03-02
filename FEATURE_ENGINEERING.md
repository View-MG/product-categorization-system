# 4. Feature Engineering

Documents the feature engineering pipeline for the product package image classification system.
All details reference the actual source code in this repository.

---

## 4.1 Key Features

### 4.1.1 Engineered Features — Preprocessing Pipeline

Defined in `src/data/transforms.py`. Two pipelines are provided:

- **`get_train_transforms()`** — used during training (includes augmentation)
- **`get_val_transforms()`** — used for validation and test (deterministic)

---

#### a) Image Resizing

Raw product images vary in size and are resized to a resolution relative to the target `size` (default 224) before being fed to the model. We first resize the shortest edge to `size + 32` (e.g. 256) to allow crop headroom.

| Pipeline       | Resize              | Crop                                        | Output Size      |
| :------------- | :------------------ | :------------------------------------------ | :--------------- |
| **Training**   | `Resize(size + 32)` | `RandomResizedCrop(size, scale=(0.7, 1.0))` | `size × size` px |
| **Validation** | `Resize(size + 32)` | `CenterCrop(size)`                          | `size × size` px |

For default size 224, `RandomResizedCrop` samples a random region (scale `0.7–1.0`, aspect ratio `0.75–1.33`) then resizes to 224 × 224, increasing viewpoint diversity per epoch. `CenterCrop` is deterministic, ensuring reproducible evaluation.

---

#### b) Data Augmentation — Training Only

Three augmentation techniques are applied in `get_train_transforms()` to improve robustness and reduce overfitting.

| Transform              | Parameters                                               | Rationale                                                                |
| :--------------------- | :------------------------------------------------------- | :----------------------------------------------------------------------- |
| `RandomHorizontalFlip` | `p=0.5`                                                  | Products can face either direction; model should be orientation-agnostic |
| `RandomRotation`       | `degrees=15` → range $[-15°,\ +15°]$                     | Handheld photos from Open Food Facts often have slight tilt              |
| `ColorJitter`          | `brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05` | Simulates varying lighting and camera conditions                         |

`ColorJitter` parameter ranges:

| Parameter    | Range             |
| :----------- | :---------------- |
| `brightness` | $[0.7,\ 1.3]$     |
| `contrast`   | $[0.7,\ 1.3]$     |
| `saturation` | $[0.8,\ 1.2]$     |
| `hue`        | $[-0.05,\ +0.05]$ |

> Augmentation is applied **only** in `get_train_transforms()`. Validation and test pipelines are augmentation-free for reliable metric measurement.

---

#### c) Tensor Conversion

After cropping, PIL Images (`uint8`, range `[0, 255]`) are converted to `torch.FloatTensor` in `(C, H, W)` format with range `[0.0, 1.0]` via `transforms.ToTensor()`:

$$x_{\text{tensor}} = \frac{x_{\text{pixel}}}{255.0}$$

Output shape: `(3, 224, 224)` — 3 RGB channels, height 224, width 224.

---

#### d) Normalization

The final step normalizes each channel using ImageNet statistics, defined as constants in `src/data/transforms.py`:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # per channel: R, G, B
IMAGENET_STD  = [0.229, 0.224, 0.225]  # per channel: R, G, B
```

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}$$

| Symbol      | Meaning                                         | R / G / B values            |
| :---------- | :---------------------------------------------- | :-------------------------- |
| $x_c$       | Pixel value before normalize, $\in [0.0,\ 1.0]$ | —                           |
| $\mu_c$     | ImageNet channel mean                           | $0.485$ / $0.456$ / $0.406$ |
| $\sigma_c$  | ImageNet channel std                            | $0.229$ / $0.224$ / $0.225$ |
| $\hat{x}_c$ | Pixel value after normalize                     | —                           |

The result is a tensor with mean ≈ 0 and std ≈ 1 per channel, matching the distribution EfficientNet-B0 was pre-trained on.

---

#### Pipeline Summary

```text
Training Pipeline — get_train_transforms()
──────────────────────────────────────────────────────
PIL Image (any size)
    │
    ├─ Resize(size+32)                    shortest side → e.g. 256 px
    ├─ RandomResizedCrop(size, 0.7–1.0)   random crop + resize → e.g. 224×224
    ├─ RandomHorizontalFlip(p=0.5)        horizontal mirror
    ├─ RandomRotation(±15°)               slight rotation
    ├─ ColorJitter(b=0.3,c=0.3,s=0.2,h=0.05)
    ├─ ToTensor()                         uint8[0,255] → float32[0.0,1.0]
    └─ Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ▼
torch.FloatTensor (3, 224, 224)


Validation / Test Pipeline — get_val_transforms()
──────────────────────────────────────────────────────
PIL Image (any size)
    │
    ├─ Resize(size+32)
    ├─ CenterCrop(size)                   deterministic center crop
    ├─ ToTensor()
    └─ Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ▼
torch.FloatTensor (3, 224, 224)
```

---

### 4.1.2 Learned Features — Representation Learning

The system uses **EfficientNet-B0** (pre-trained on ImageNet-1K) as an automatic feature extractor, implemented as `ProductClassifier` in `src/models/model.py`.

```text
Input: torch.FloatTensor (B, 3, 224, 224)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  EfficientNet-B0 Backbone (backbone.features)    │
│                                                  │
│  features[0]    Stem Conv         Low-level:  edges, gradients            │
│  features[1–3]  MBConv (early)    Mid-level:  textures, patterns          │
│  features[4–6]  MBConv (middle)   Mid-level:  shapes, object parts        │
│  features[7–8]  MBConv (late)     High-level: semantic representations    │
└──────────────────────────────────────────────────┘
    │
    ├─ AdaptiveAvgPool2d(1,1)    → (B, 1280, 1, 1)
    ├─ Flatten                   → (B, 1280)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Classification Head                             │
│  Dropout(p=0.3)                                  │
│  Linear(1280 → 2)                                │
└──────────────────────────────────────────────────┘
    │
    ▼
Output Logits: (B, 4)   — [beverage, snack]
```

**Transfer Learning Strategy:**

Initialized from `EfficientNet_B0_Weights.IMAGENET1K_V1`. `ProductClassifier` supports staged training:

| Stage     | Method                      | Backbone         | Head      | Purpose                                                 |
| :-------- | :-------------------------- | :--------------- | :-------- | :------------------------------------------------------ |
| Stage 1   | `freeze_backbone=True`      | Frozen           | Trainable | Train head first without disturbing pre-trained weights |
| Stage 2   | `unfreeze_backbone()`       | Trainable        | Trainable | Full fine-tune to adapt backbone to this dataset        |
| Stage 1.5 | `unfreeze_last_n_blocks(3)` | Partial (last 3) | Trainable | Gradual unfreezing for gradient stability               |

---

### 4.1.3 Label Encoding

Managed by `ProductDataset` in `src/data/dataset.py`. The label mapping is loaded from `data_local/processed/data_v2/label_map.json`, generated via `attach_label_map()` in `src/data/prepare.py`.

```json
{
  "beverage": 0,
  "snack": 1
}
```

`ProductDataset` validates that every label in `manifest_clean.csv` exists in the mapping before training begins. Unknown labels raise a `ValueError` immediately.

`__getitem__` returns:

```python
image : torch.FloatTensor  # shape (3, 224, 224)
label : torch.LongTensor   # scalar class index (0–1)
```

---

## 4.2 Feature Rationale

### 4.2.1 Why 224 × 224 pixels?

224 × 224 is EfficientNet-B0's **native input resolution**. At this size, the backbone produces a `(1280, 7, 7)` feature map, which `AdaptiveAvgPool2d(1, 1)` compresses into a 1,280-dim global vector with maximum efficiency. Deviating from this size may alter the spatial representation that the model was pre-trained on.

### 4.2.2 Why ImageNet Normalization?

Two reasons:

- **Weight compatibility:** EfficientNet-B0 was pre-trained with ImageNet-normalized inputs. Un-normalized images shift early-layer activations outside the expected range, degrading transfer learning quality.
- **Gradient stability:** Centering pixel values around 0 stabilizes gradient flow during backpropagation.

### 4.2.3 Why These Augmentations?

Images come from Open Food Facts, so they vary in angle, lighting, and framing.

| Transform              | Parameter         | Target variation                                            |
| :--------------------- | :---------------- | :---------------------------------------------------------- |
| `RandomResizedCrop`    | `scale=(0.7,1.0)` | Products occupy different fractions of the frame            |
| `RandomHorizontalFlip` | `p=0.5`           | Labels face left or right; model must be direction-agnostic |
| `RandomRotation`       | `±15°`            | Handheld shots are often slightly tilted                    |
| `ColorJitter`          | `b=0.3, c=0.3`    | Lighting varies across shelf, outdoor, and studio shots     |

---

## 4.3 Unavailable / Excluded Features

### 4.3.1 Dropped Metadata Fields

During data preparation (`src/data/prepare.py`), certain extraneous fields from the raw dataset are explicitly dropped and do **not** appear in `manifest_clean.csv`. These fields are removed to prevent data leakage or simply because they hold no predictive value:

| Field                 | Reason for exclusion                                                            |
| :-------------------- | :------------------------------------------------------------------------------ |
| `image_url` / `split` | Contains URL patterns or split info that shouldn't leak during training         |
| `image_id` / `source` | Identifiers and provenance metadata (Open Food Facts); no predictive value      |
| `license_db` etc.     | Copyright and licensing metadata is irrelevant to the product's visual category |
| `w`, `h`, `file_size` | Purely diagnostic metadata regarding the raw image file's properties            |

### 4.3.2 Retained, But Unused Text Features

The final `manifest_clean.csv` retains semantic metadata such as `product_name` and `categories_tags_en`. **However, they are completely excluded from the model pipeline.**

Although these fields are present in the dataset CSV to enable future multimodal (image + text) experimentation, the `ProductDataset` in `src/data/dataset.py` ignores these text fields entirely and reads **only** the `abs_path` and `label_coarse`.

The current system is strictly an **image-only classifier** because:

1. The goal is to measure the model's classification ability from **visual appearance alone**.
2. Avoids managing multilingual, or possibly inconsistent, user-generated text.
3. Prevents direct label leakage — text fields like `categories_tags_en` or `product_name` often contain the explicit ground truth label (e.g. "drinking water").

**Note on `barcode`:** Retained and used mostly as a grouping key in `split_by_barcode()` (`src/data/split.py`) to prevent instance-level data leakage (i.e. to ensure multiple photos of the same barcode go to the same split). It is **not** passed to the model as a feature.

### 4.3.3 Background Removal

No background removal or segmentation is applied. Raw images are processed whole, including whatever background is present. Background noise reliance is mitigated implicitly via `RandomResizedCrop` and `ColorJitter`.

---

## Summary

| Aspect             | Implementation                                                             | Reference file           |
| :----------------- | :------------------------------------------------------------------------- | :----------------------- |
| Image size         | (size) × (size) pixels (default: 224)                                      | `src/data/transforms.py` |
| Normalization      | ImageNet Mean/Std                                                          | `src/data/transforms.py` |
| Train augmentation | RandomResizedCrop, RandomHorizontalFlip, RandomRotation(±15°), ColorJitter | `src/data/transforms.py` |
| Val/Test transform | Resize → CenterCrop → Normalize                                            | `src/data/transforms.py` |
| Text / Metadata    | **Ignored completely during training**                                     | `src/data/dataset.py`    |
| Backbone           | EfficientNet-B0 (ImageNet pre-trained)                                     | `src/models/model.py`    |
| Output tensor      | `FloatTensor (3, 224, 224)` + `LongTensor scalar`                          | `src/data/dataset.py`    |

---

_This document reflects the system at the time of writing. Update accordingly if the pipeline or architecture changes._
