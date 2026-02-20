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

Raw product images vary in size and are resized to a fixed resolution before being fed to the model.

| Pipeline       | Resize        | Crop                                       | Output Size  |
| :------------- | :------------ | :----------------------------------------- | :----------- |
| **Training**   | `Resize(256)` | `RandomResizedCrop(224, scale=(0.7, 1.0))` | 224 × 224 px |
| **Validation** | `Resize(256)` | `CenterCrop(224)`                          | 224 × 224 px |

`RandomResizedCrop` samples a random region (scale `0.7–1.0`, aspect ratio `0.75–1.33`) then resizes to 224 × 224, increasing viewpoint diversity per epoch. `CenterCrop` is deterministic, ensuring reproducible evaluation.

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

```
Training Pipeline — get_train_transforms()
──────────────────────────────────────────────────────
PIL Image (any size)
    │
    ├─ Resize(256)                        shortest side → 256 px
    ├─ RandomResizedCrop(224, 0.7–1.0)    random crop + resize → 224×224
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
    ├─ Resize(256)
    ├─ CenterCrop(224)                    deterministic center crop
    ├─ ToTensor()
    └─ Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ▼
torch.FloatTensor (3, 224, 224)
```

---

### 4.1.2 Learned Features — Representation Learning

The system uses **EfficientNet-B0** (pre-trained on ImageNet-1K) as an automatic feature extractor, implemented as `ProductClassifier` in `src/models/model.py`.

```
Input: torch.FloatTensor (B, 3, 224, 224)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  EfficientNet-B0 Backbone (backbone.features)    │
│                                                  │
│  features[0]    Stem Conv         Low-level:  edges, corners, gradients       │
│  features[1–3]  MBConv (early)    Mid-level:  textures, color patterns        │
│  features[4–6]  MBConv (middle)   Mid-level:  shapes, partial objects         │
│  features[7–8]  MBConv (late)     High-level: semantic representations        │
└──────────────────────────────────────────────────┘
    │
    ├─ AdaptiveAvgPool2d(1,1)    → (B, 1280, 1, 1)
    ├─ Flatten                   → (B, 1280)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Classification Head                             │
│  Dropout(p=0.3)                                  │
│  Linear(1280 → 4)                                │
└──────────────────────────────────────────────────┘
    │
    ▼
Output Logits: (B, 4)   — [beverages, snacks, dry_food, other]
```

**Feature hierarchy:**

| Level      | Layers                    | What is learned                                      |
| :--------- | :------------------------ | :--------------------------------------------------- |
| Low-level  | `features[0]`             | Edges, corners, color gradients, pixel structure     |
| Mid-level  | `features[1–6]`           | Textures, color patterns, graphic structures, shapes |
| High-level | `features[7–8]` + Pooling | Semantic representations (1,280-dim vector)          |

**Transfer Learning Strategy:**

Initialized from `EfficientNet_B0_Weights.IMAGENET1K_V1`. `ProductClassifier` supports staged training:

| Stage     | Method                      | Backbone         | Head      | Purpose                                                 |
| :-------- | :-------------------------- | :--------------- | :-------- | :------------------------------------------------------ |
| Stage 1   | `freeze_backbone=True`      | Frozen           | Trainable | Train head first without disturbing pre-trained weights |
| Stage 2   | `unfreeze_backbone()`       | Trainable        | Trainable | Full fine-tune to adapt backbone to this dataset        |
| Stage 1.5 | `unfreeze_last_n_blocks(3)` | Partial (last 3) | Trainable | Gradual unfreezing for gradient stability               |

---

### 4.1.3 Label Encoding

Managed by `ProductDataset` in `src/data/dataset.py`. The label mapping is loaded from `data_local/processed/data_v2/label_map.json`, generated by `scripts/prepare_dataset.py` via `attach_label_map()` in `src/data/prepare.py`.

```json
{
  "beverages": 0,
  "snacks": 1,
  "dry_food": 2,
  "other": 3
}
```

`ProductDataset` validates that every label in `manifest_clean.csv` exists in the mapping before training begins. Unknown labels raise a `ValueError` immediately to prevent silent errors.

`__getitem__` returns:

```python
image : torch.FloatTensor  # shape (3, 224, 224)
label : torch.LongTensor   # scalar class index (0–3)
```

---

## 4.2 Feature Rationale

### 4.2.1 Why 224 × 224 pixels?

224 × 224 is EfficientNet-B0's **native input resolution** (Tan & Le, 2019). At this size, the backbone produces a `(1280, 7, 7)` feature map, which `AdaptiveAvgPool2d(1, 1)` compresses into a 1,280-dim global vector with maximum efficiency. Deviating from this size may degrade performance.

### 4.2.2 Why ImageNet Normalization?

Two reasons:

- **Weight compatibility:** EfficientNet-B0 was pre-trained with ImageNet-normalized inputs. Supplying un-normalized images shifts early-layer activations outside the optimized range, degrading transfer learning quality.
- **Gradient stability:** Centering pixel values around 0 stabilizes gradient flow during backpropagation, reducing vanishing/exploding gradient risk and enabling faster, more stable convergence.

### 4.2.3 Why These Augmentations?

Images come from Open Food Facts (user-generated), so they vary in angle, lighting, and framing. Each augmentation targets a specific real-world variation:

| Transform              | Parameter         | Target variation                                            |
| :--------------------- | :---------------- | :---------------------------------------------------------- |
| `RandomResizedCrop`    | `scale=(0.7,1.0)` | Products occupy different fractions of the frame            |
| `RandomHorizontalFlip` | `p=0.5`           | Labels face left or right; model must be direction-agnostic |
| `RandomRotation`       | `±15°`            | Handheld shots are often slightly tilted                    |
| `ColorJitter`          | `b=0.3, c=0.3`    | Lighting varies across shelf, outdoor, and studio shots     |

Together they expand effective dataset size and encourage the model to learn transformation-**invariant** features.

### 4.2.4 Why EfficientNet-B0?

| Criterion               | EfficientNet-B0       | Reason                                                   |
| :---------------------- | :-------------------- | :------------------------------------------------------- |
| Parameters              | ~5.3 M                | Appropriate for a medium-sized dataset; low overfit risk |
| Input size              | 224 × 224             | Matches the pipeline's standard resolution               |
| ImageNet Top-1 Accuracy | ~77.7 %               | Strong baseline for transfer learning                    |
| Feature dimension       | 1,280                 | Sufficient capacity for 4 classes                        |
| Availability            | `torchvision >= 0.13` | No extra dependencies required                           |

---

## 4.3 Excluded Features

### 4.3.1 Metadata Fields

All additional fields from `data_local/raw_extracted/data_v2/metadata.csv` are excluded from training features:

| Field                 | Reason for exclusion                                                                        |
| :-------------------- | :------------------------------------------------------------------------------------------ |
| `image_url`           | Contains the category label in the directory path (`images/beverages/…`) — **data leakage** |
| `product_name`        | May directly name the category (e.g. "drinking water"), bypassing visual learning           |
| `abs_path` / Filename | File path format `images/<label>/<barcode>.jpg` contains the ground truth label directly    |
| Timestamps            | No relationship to a product's visual category                                              |
| `source`              | Provenance metadata (Open Food Facts); no predictive value                                  |

**Note on `barcode`:** Used only as a grouping key in `split_by_barcode()` (`src/data/split.py`) to prevent instance-level data leakage. It is **not** used as a prediction feature — new products unseen at training time must be classifiable from their image alone.

### 4.3.2 Text Features

Although `categories_tags_en` and `product_name` are available, the current system is an **image-only classifier** for three reasons:

1. To measure the model's classification ability from **visual appearance alone**.
2. To avoid managing multilingual, inconsistent user-generated text.
3. To prevent direct label leakage — `categories_tags_en` may contain words equivalent to the ground truth label.

Multimodal (image + text) features may be considered in a future version as noted in `README.md`.

### 4.3.3 Background Removal

No background removal or segmentation is applied. Raw images are processed whole, including background. Background noise is mitigated indirectly by:

1. **`RandomResizedCrop`** — forces the model to learn from different image regions each epoch, reducing reliance on fixed background patterns.
2. **`ColorJitter`** — reduces sensitivity to specific background colors, encouraging the model to focus on consistent foreground features.

---

## Summary

| Aspect             | Implementation                                                             | Reference file                                     |
| :----------------- | :------------------------------------------------------------------------- | :------------------------------------------------- |
| Image size         | 224 × 224 pixels                                                           | `src/data/transforms.py`                           |
| Normalization      | ImageNet Mean/Std                                                          | `src/data/transforms.py`                           |
| Train augmentation | RandomResizedCrop, RandomHorizontalFlip, RandomRotation(±15°), ColorJitter | `src/data/transforms.py`                           |
| Val/Test transform | Resize → CenterCrop → Normalize                                            | `src/data/transforms.py`                           |
| Backbone           | EfficientNet-B0 (ImageNet pre-trained)                                     | `src/models/model.py`                              |
| Feature dimension  | 1,280                                                                      | `src/models/model.py`                              |
| Number of classes  | 4                                                                          | `src/models/model.py`, `src/config/data_config.py` |
| Label encoding     | Integer index via `label_map.json`                                         | `src/data/dataset.py`                              |
| Output tensor      | `FloatTensor (3, 224, 224)` + `LongTensor scalar`                          | `src/data/dataset.py`                              |

---

_This document reflects the `feat/feature-engineering` branch at the time of writing. Update accordingly if the pipeline or architecture changes._
