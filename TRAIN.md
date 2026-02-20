# Smart Product Categorization System (Image Classification)

A prototype ML system that classifies **product package images** into coarse categories:

- `beverages`
- `snacks`
- `dry_food`
- `other`

---

## Repository Dataset (Hugging Face)

This project stores the dataset snapshot in a Hugging Face Datasets repository:

- Dataset repo: `Phathanan/product-categorization-system`
- Link: https://huggingface.co/datasets/Phathanan/product-categorization-system

> The code in this project uses the dataset location **from config** (`src/config/data_config.py`).

---

## 1) Data Documentation (Where the data comes from)

### 1.1 Open Food Facts (OFF) — Primary Source

**Open Food Facts (OFF)** is the main data source for the current dataset version, used to collect:

- product package images (mainly **front** images)
- metadata such as `barcode`, `product_name`, `categories_tags_en`

**How we access OFF:**

- query products using the **Open Food Facts API**
- download product images from the returned image URLs (front image preferred)
- store image + metadata rows in CSV

**Licensing / Attribution (short):**

- OFF database is under **ODbL**
- Product images are under **CC BY-SA**
- We keep `source`, `image_url`, and license fields in metadata for traceability and attribution.

---

### 1.2 Open Beauty Facts (OBF) — Optional Future Source

In future versions, **Open Beauty Facts (OBF)** may be added to improve coverage for personal care items.  
OBF uses a similar concept: barcode + front image + categories/metadata.

Current version focuses on OFF only.

---

## 2) What We Collect

### 2.1 Images

- We collect mainly **front package images**
- Stored as `.jpg`
- File name uses **barcode** (normalized to 13 digits) for easy tracing

Example layout:

```text
images/
  beverages/0000000000123.jpg
  snacks/0000000000456.jpg
  dry_food/0000000000789.jpg
  other/0000000000999.jpg
```

### 2.2 Metadata (`metadata.csv`)

| Column               | Description                                         |
| :------------------- | :-------------------------------------------------- |
| `barcode`            | 13-digit normalized EAN barcode                     |
| `product_name`       | Product name (user-generated, may be multilingual)  |
| `categories_tags_en` | English category tags from OFF                      |
| `label_coarse`       | Our coarse label: `beverages/snacks/dry_food/other` |
| `image_id`           | Relative path to image file                         |
| `image_url`          | Original URL (for attribution)                      |
| `source`             | Data source identifier (`off`)                      |

---

## 3) Project Structure

```text
product-categorization-system/
├── data_local/
│   ├── raw_extracted/data_v2/     ← extracted tar (images + metadata.csv)
│   └── processed/data_v2/        ← manifest_clean.csv, label_map.json, splits.json, stats.json
├── scripts/
│   └── prepare_dataset.py        ← end-to-end data preparation pipeline
├── src/
│   ├── config/
│   │   ├── data_config.py        ← DataConfig dataclass (paths, HF settings, split fractions)
│   │   └── train_config.py       ← Training hyperparameters config
│   ├── data/
│   │   ├── loader.py             ← Download & extract tar from Hugging Face
│   │   ├── prepare.py            ← Clean & normalize metadata
│   │   ├── validate.py           ← Image file validation (size, readability)
│   │   ├── split.py              ← Barcode-stratified train/val/test split
│   │   ├── stats.py              ← Dataset statistics
│   │   ├── transforms.py         ← torchvision transform pipelines
│   │   └── dataset.py            ← ProductDataset (PyTorch Dataset)
│   ├── models/
│   │   ├── model.py              ← ProductClassifier (EfficientNet-B0 + custom head)
│   │   └── factory.py            ← Model builder utilities
│   └── training/
│       └── logger.py             ← Training logger
├── train.py                      ← Training entry point
├── test_feature_engineering.py   ← Smoke tests for the pipeline
├── FEATURE_ENGINEERING.md        ← Feature engineering documentation
├── requirements.txt              ← Python dependencies
└── README.md
```

---

## 4) Feature Engineering

See [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) for full details. Summary:

| Aspect             | Implementation                                                             |
| :----------------- | :------------------------------------------------------------------------- |
| Image size         | 224 × 224 pixels                                                           |
| Normalization      | ImageNet Mean/Std                                                          |
| Train augmentation | RandomResizedCrop, RandomHorizontalFlip, RandomRotation(±15°), ColorJitter |
| Val/Test transform | Resize(256) → CenterCrop(224) → Normalize                                  |
| Backbone           | EfficientNet-B0 (ImageNet pre-trained)                                     |
| Feature dimension  | 1,280                                                                      |
| Number of classes  | 4                                                                          |
| Label encoding     | Integer index via `label_map.json`                                         |

---

## 5) Setup & Installation

### 5.1 Clone the repository

```bash
git clone https://github.com/View-MG/product-categorization-system.git
cd product-categorization-system
```

### 5.2 Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 5.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 5.4 Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

> `HF_TOKEN` is required to download the dataset from Hugging Face.

---

## 6) Data Preparation

Run the full data preparation pipeline:

```bash
python scripts/prepare_dataset.py
```

This will:

1. Download `data_v2.tar` from the Hugging Face dataset repository
2. Extract images and `metadata.csv` to `data_local/raw_extracted/data_v2/`
3. Clean and validate metadata
4. Validate images (remove broken / too-small files)
5. Split data by barcode (group split, no leakage) into train / val / test
6. Save outputs to `data_local/processed/data_v2/`:
   - `manifest_clean.csv`
   - `label_map.json`
   - `splits.json`
   - `stats.json`

---

## 7) Training

```bash
python train.py
```

Training follows a **two-stage fine-tuning** strategy:

| Stage   | Backbone        | Head      | Purpose                               |
| :------ | :-------------- | :-------- | :------------------------------------ |
| Stage 1 | Frozen          | Trainable | Train classification head only        |
| Stage 2 | Fully trainable | Trainable | Full fine-tune at lower learning rate |

---

## 8) Smoke Tests

Run the feature engineering smoke tests to verify the pipeline is working correctly:

```bash
python test_feature_engineering.py
```

Tests cover:

- Transform output shapes and dtypes
- ImageNet normalization applied correctly
- Train transform stochasticity / val transform determinism
- Label map correctness (4 classes, values 0–3)
- `ProductDataset` `__getitem__` output shapes
- Model forward pass output shape and dtype

---

## 9) Model Architecture

```
Input: FloatTensor (B, 3, 224, 224)
    │
EfficientNet-B0 Backbone (pre-trained ImageNet-1K)
    │  features[0]    → Stem Conv         (edges, corners, gradients)
    │  features[1–3]  → MBConv (early)    (textures, color patterns)
    │  features[4–6]  → MBConv (middle)   (shapes, partial objects)
    │  features[7–8]  → MBConv (late)     (semantic representations)
    │
AdaptiveAvgPool2d(1,1)  → (B, 1280, 1, 1)
    │
Flatten                 → (B, 1280)
    │
Dropout(p=0.3)
    │
Linear(1280 → 4)
    │
Output Logits: (B, 4)   ← pair with nn.CrossEntropyLoss
```

---

## 10) Future Work

- Add **Open Beauty Facts** as a second data source
- Explore **multimodal** (image + text) classification using `product_name` or `categories_tags_en`
- Experiment with larger EfficientNet variants (B1–B4)
- Add inference / prediction script
