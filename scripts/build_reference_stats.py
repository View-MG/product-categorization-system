from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import convnext_tiny
from tqdm.auto import tqdm

import cv2


MANIFEST_PATH = Path("data_local/processed/data_v2")
MODEL_PATH = Path("model.safetensors")

OUTPUT_DIR = Path("reference")
REFERENCE_STATS_PATH = OUTPUT_DIR / "reference_stats.json"
REFERENCE_EMBEDDINGS_PATH = OUTPUT_DIR / "reference_embeddings.npz"

IMAGE_COL = "abs_path"
LABEL_COL = "label_coarse"

NUM_CLASSES = 2
CLASS_NAMES = ["beverage", "snack"]
ARCHITECTURE = "convnext_tiny"

INPUT_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

SAMPLE_EMBEDDINGS = 500
RANDOM_STATE = 42


def build_model():
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

def load_model(model_path: Path):
    model = build_model()

    weights = load_file(str(model_path), device="cpu")

    backbone_weights = {}
    for k, v in weights.items():
        if k.startswith("_backbone."):
            new_key = k[len("_backbone."):]
            backbone_weights[new_key] = v

    if not backbone_weights:
        raise ValueError("No keys starting with '_backbone.' were found in model.safetensors")

    missing_keys, unexpected_keys = model.load_state_dict(backbone_weights, strict=False)

    print("missing_keys =", missing_keys)
    print("unexpected_keys =", unexpected_keys)

    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

def compute_brightness(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return float(gray.mean())

def compute_blur_var(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    if cv2 is not None:
        return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())

    lap = (
        gray[:-2, 1:-1] +
        gray[2:, 1:-1] +
        gray[1:-1, :-2] +
        gray[1:-1, 2:] -
        4 * gray[1:-1, 1:-1]
    )
    return float(lap.var())

def extract_embedding_and_confidence(model, image_path: str, transform):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    brightness = compute_brightness(image)
    blur_var = compute_blur_var(image)

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        x = model.features(x)
        x = model.avgpool(x)
        x = model.classifier[0](x)
        embedding = model.classifier[1](x)
        logits = model.classifier[2](embedding)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return {
        "embedding": embedding.squeeze(0).cpu().numpy(),
        "logits": logits.squeeze(0).cpu().numpy(),
        "confidence": float(confidence.item()),
        "pred_idx": int(pred_idx.item()),
        "brightness": brightness,
        "blur_var": blur_var,
        "width": int(width),
        "height": int(height),
    }


def compute_confidence_stats(confidences):
    confidences = np.asarray(confidences, dtype=np.float32)
    bin_edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
    counts, edges = np.histogram(confidences, bins=bin_edges)

    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "mean": float(confidences.mean()),
        "std": float(confidences.std()),
        "percentile_10": float(np.percentile(confidences, 10)),
        "percentile_25": float(np.percentile(confidences, 25)),
        "percentile_75": float(np.percentile(confidences, 75)),
        "percentile_90": float(np.percentile(confidences, 90)),
    }


def compute_class_ratio(labels):
    series = pd.Series(labels)
    counts = series.value_counts(normalize=True)

    ratio = {}
    for class_name in CLASS_NAMES:
        ratio[class_name] = float(counts.get(class_name, 0.0))

    ratio["total_samples"] = int(len(labels))
    return ratio


def compute_percentiles(values, keys):
    arr = np.asarray(values, dtype=np.float32)
    return {key: float(np.percentile(arr, p)) for key, p in keys.items()}


def build_reference_stats(records, checkpoint_path: Path):
    embeddings = np.stack([r["embedding"] for r in records], axis=0)
    confidences = [r["confidence"] for r in records]
    labels = [r["label"] for r in records]
    brightnesses = [r["brightness"] for r in records]
    blur_vars = [r["blur_var"] for r in records]
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]

    reference_stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "model_info": {
            "checkpoint_path": str(checkpoint_path).replace("\\", "/"),
            "architecture": ARCHITECTURE,
            "classes": CLASS_NAMES,
            "input_size": INPUT_SIZE,
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
            "embedding_dim": int(embeddings.shape[1]),
        },

        "class_distribution_train": compute_class_ratio(labels),

        "confidence_histogram": compute_confidence_stats(confidences),

        "embedding_stats": {
            "mean": embeddings.mean(axis=0).tolist(),
            "std": embeddings.std(axis=0).tolist(),
            "sample_path": str(REFERENCE_EMBEDDINGS_PATH).replace("\\", "/"),
            "n_samples": int(min(SAMPLE_EMBEDDINGS, len(records))),
        },

        "quality_percentiles": {
            "brightness": compute_percentiles(
                brightnesses,
                {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90},
            ),
            "blur_var": compute_percentiles(
                blur_vars,
                {"p10": 10, "p25": 25, "p50": 50, "p75": 75, "p90": 90},
            ),
            "width": compute_percentiles(
                widths,
                {"p10": 10, "p50": 50},
            ),
            "height": compute_percentiles(
                heights,
                {"p10": 10, "p50": 50},
            ),
        },
    }

    return reference_stats


def stratified_sample_embeddings(records, sample_size=500, random_state=42):
    total = len(records)
    if total == 0:
        raise ValueError("records is empty")

    if sample_size >= total:
        embeddings = np.stack([r["embedding"] for r in records], axis=0)
        labels = np.array([r["label"] for r in records])
        image_paths = np.array([r["image_path"] for r in records])
        return {
            "embeddings": embeddings,
            "labels": labels,
            "image_paths": image_paths
        }

    df_records = pd.DataFrame({
        "idx": np.arange(total),
        "label": [r["label"] for r in records]
    })

    class_counts = df_records["label"].value_counts().sort_index()
    expected = class_counts / total * sample_size
    base_counts = np.floor(expected).astype(int)

    remainder = sample_size - int(base_counts.sum())
    fractions = (expected - base_counts).sort_values(ascending=False)

    final_counts = base_counts.copy()

    for label in fractions.index:
        if remainder == 0:
            break
        if final_counts[label] < class_counts[label]:
            final_counts[label] += 1
            remainder -= 1

    rng = np.random.default_rng(random_state)
    sampled_indices = []

    for label, n in final_counts.items():
        class_indices = df_records.loc[df_records["label"] == label, "idx"].to_numpy()
        chosen = rng.choice(class_indices, size=int(n), replace=False)
        sampled_indices.extend(chosen.tolist())

    rng.shuffle(sampled_indices)

    sampled_records = [records[i] for i in sampled_indices]

    embeddings = np.stack([r["embedding"] for r in sampled_records], axis=0)
    labels = np.array([r["label"] for r in sampled_records])
    image_paths = np.array([r["image_path"] for r in sampled_records])

    return {
        "embeddings": embeddings,
        "labels": labels,
        "image_paths": image_paths
    }


def save_reference_embeddings_npz(sampled_data, output_path=REFERENCE_EMBEDDINGS_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=sampled_data["embeddings"].astype(np.float32),
        labels=sampled_data["labels"],
        image_paths=sampled_data["image_paths"]
    )


def main():
    csv_path = MANIFEST_PATH / "manifest_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = [IMAGE_COL, LABEL_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"manifest_clean.csv missing columns: {missing}")

    model = load_model(MODEL_PATH)
    transform = build_transform()

    records = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Building reference stats",
        unit="img"
    ):
        image_path = row[IMAGE_COL]
        label = row[LABEL_COL]

        result = extract_embedding_and_confidence(model, image_path, transform)

        records.append({
            "image_path": image_path,
            "label": label,
            "embedding": result["embedding"],
            "confidence": result["confidence"],
            "pred_idx": result["pred_idx"],
            "brightness": result["brightness"],
            "blur_var": result["blur_var"],
            "width": result["width"],
            "height": result["height"]
        })

    sampled_data = stratified_sample_embeddings(
        records,
        sample_size=SAMPLE_EMBEDDINGS,
        random_state=RANDOM_STATE
    )

    save_reference_embeddings_npz(
        sampled_data,
        output_path=REFERENCE_EMBEDDINGS_PATH
    )

    reference_stats = build_reference_stats(
        records=records,
        checkpoint_path=MODEL_PATH
    )

    with open(REFERENCE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(reference_stats, f, indent=2, ensure_ascii=False)

    print(f"saved: {REFERENCE_STATS_PATH}")
    print(f"saved: {REFERENCE_EMBEDDINGS_PATH}")
    print(pd.Series(sampled_data['labels']).value_counts().sort_index())


if __name__ == "__main__":
    main()