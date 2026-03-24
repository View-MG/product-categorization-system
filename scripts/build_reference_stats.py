from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import convnext_tiny
from tqdm.auto import tqdm
from datetime import datetime

import cv2

MANIFEST_PATH = Path("data_local/processed/data_v2")
MODEL_PATH = Path("model.safetensors")
IMAGE_COL = "abs_path"
LABEL_COL = "label_coarse"
NUM_CLASSES = 2
INPUT_SIZE = 224
SAMPLE_EMBEDDINGS = 1000
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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
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
        "std": float(confidences.std())
    }

def compute_class_ratio(labels):
    series = pd.Series(labels)
    ratio = series.value_counts(normalize=True).sort_index().to_dict()
    ratio["total_samples"] = int(len(labels))
    return ratio

def compute_basic_stats(values):
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max())
    }

def build_reference_stats(records):
    embeddings = np.stack([r["embedding"] for r in records], axis=0)
    confidences = [r["confidence"] for r in records]
    labels = [r["label"] for r in records]
    brightnesses = [r["brightness"] for r in records]
    blur_vars = [r["blur_var"] for r in records]
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]

    reference_stats = {
        "embedding_stats": {
            "mean": embeddings.mean(axis=0).tolist(),
            "std": embeddings.std(axis=0).tolist(),
            "embedding_dim": int(embeddings.shape[1]),
            "num_samples": int(embeddings.shape[0])
        },
        "confidence_histogram": compute_confidence_stats(confidences),
        "class_distribution_train": compute_class_ratio(labels),
        "quality_stats": {
            "brightness": compute_basic_stats(brightnesses),
            "blur_var": compute_basic_stats(blur_vars)
        },
        "image_size_stats": {
            "width": compute_basic_stats(widths),
            "height": compute_basic_stats(heights)
        }
    }

    return reference_stats

def stratified_sample_embeddings(records, sample_size=1000, random_state=42):
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

def save_reference_embeddings_npz(sampled_data, output_path="reference_embeddings.npz"):
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

    reference_stats = build_reference_stats(records)

    with open("reference_stats.json", "w", encoding="utf-8") as f:
        json.dump(reference_stats, f, indent=2, ensure_ascii=False)

    sampled_data = stratified_sample_embeddings(
        records,
        sample_size=SAMPLE_EMBEDDINGS,
        random_state=RANDOM_STATE
    )

    save_reference_embeddings_npz(
        sampled_data,
        output_path="reference_embeddings.npz"
    )

    print("saved: reference_stats.json")
    print("saved: reference_embeddings.npz")
    print(pd.Series(sampled_data["labels"]).value_counts().sort_index())

if __name__ == "__main__":
    main()