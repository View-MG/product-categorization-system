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
