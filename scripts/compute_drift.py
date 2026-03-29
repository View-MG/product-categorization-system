import base64
import io
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import convnext_tiny


DB_PATH = Path("data/monitoring.db")
REFERENCE_STATS_PATH = Path("reference/reference_stats.json")
REFERENCE_EMBEDDINGS_PATH = Path("reference/reference_embeddings.npz")

CHECKPOINT_PATH = Path("model.safetensors")

WINDOW_SIZE = 100
BATCH_SIZE = 32

EMBEDDING_THRESHOLD = 1.95
CONFIDENCE_THRESHOLD = 0.25
CLASS_THRESHOLD = 0.1


class ConvNextTinyWithEmbedding(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        embedding = torch.flatten(x, 1)
        logits = self.backbone.classifier[2](embedding)
        return logits, embedding

def build_model(num_classes: int):
    model = ConvNextTinyWithEmbedding(num_classes=num_classes)
    
    weights = load_file(str(CHECKPOINT_PATH), device="cpu")

    backbone_weights = {}
    for k, v in weights.items():
        if k.startswith("_backbone."):
            new_key = "backbone." + k[len("_backbone."):]
            backbone_weights[new_key] = v

    if not backbone_weights:
        raise ValueError("No keys starting with '_backbone.' were found in model.safetensors")

    missing_keys, unexpected_keys = model.load_state_dict(backbone_weights, strict=False)

    print("missing_keys =", missing_keys)
    print("unexpected_keys =", unexpected_keys)

    model.eval()
    return model


def build_transform(input_size: int, mean: list[float], std: list[float]):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_reference_stats():
    with open(REFERENCE_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_embedding_mean(ref_stats: dict):
    raw_mean = ref_stats.get("embedding_stats", {}).get("mean")

    if isinstance(raw_mean, list):
        ref_mean = np.asarray(raw_mean, dtype=np.float32)
        if ref_mean.ndim == 1 and ref_mean.size > 0:
            return ref_mean
        
    data = np.load(REFERENCE_EMBEDDINGS_PATH, allow_pickle=True)

    if "embeddings" in data:
        ref_embeddings = data["embeddings"]
    else:
        ref_embeddings = data[data.files[0]]

    ref_embeddings = np.asarray(ref_embeddings, dtype=np.float32)
    
    return ref_embeddings.mean(axis=0)

def decode_image_data_url(image_data_url: str) -> Image.Image:
    if not image_data_url:
        raise ValueError("image_data_url is empty")

    if "," not in image_data_url:
        raise ValueError("invalid image_data_url format")

    _, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def normalize_label(name: str) -> str:
    s = str(name).strip().lower()
    mapping = {
        "beverage": "beverage",
        "beverages": "beverage",
        "snack": "snack",
        "snacks": "snack",
    }
    if s not in mapping:
        raise ValueError(f"invalid class label: {name}")
    return mapping[s]

def load_n_latest_predictions(window_size: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(prediction_events)").fetchall()
    }

    required = {"id", "predicted_class", "confidence", "image_data_url"}
    missing = required - columns
    if missing:
        conn.close()
        raise ValueError(f"prediction_events missing columns: {missing}")

    rows = conn.execute(
        """
        SELECT id, predicted_class, confidence, image_data_url
        FROM prediction_events
        WHERE image_data_url IS NOT NULL
          AND TRIM(image_data_url) != ''
        ORDER BY id DESC
        LIMIT ?
        """,
        (window_size,),
    ).fetchall()

    conn.close()

    if len(rows) < window_size:
        raise ValueError(
            f"not enough prediction rows: need {window_size}, got {len(rows)}"
        )

    rows = rows[::-1]

    ids = [int(r["id"]) for r in rows]
    classes = [normalize_label(r["predicted_class"]) for r in rows]
    confidences = np.array([float(r["confidence"]) for r in rows], dtype=np.float32)
    image_data_urls = [str(r["image_data_url"]) for r in rows]

    if np.any(confidences < 0.0) or np.any(confidences > 1.0):
        raise ValueError("confidence must be between 0 and 1")

    for image_data_url in image_data_urls:
        if not image_data_url:
            raise ValueError("some rows have empty image_data_url")

    return ids, classes, confidences, image_data_urls



def infer_recent_embeddings(
    model: nn.Module,
    image_data_urls: list[str],
    transform,
    device: torch.device,
    batch_size: int = 32,
):
    all_embeddings = []

    for start in range(0, len(image_data_urls), batch_size):
        batch_urls = image_data_urls[start:start + batch_size]
        batch_tensors = []

        for image_data_url in batch_urls:
            image = decode_image_data_url(image_data_url)
            tensor = transform(image)
            batch_tensors.append(tensor)

        x = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            _, embeddings = model(x)

        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def compute_embedding_drift(ref_mean: np.ndarray, recent_embeddings: np.ndarray) -> float:
    # Part: Embedding Drift
    # สูตรที่ใช้คือ L2 distance ระหว่าง mean embedding ของ reference กับ mean embedding ของ window ล่าสุด
    recent_mean = recent_embeddings.mean(axis=0)
    return float(np.linalg.norm(recent_mean - ref_mean))


def compute_psi(expected_probs: np.ndarray, actual_probs: np.ndarray, eps: float = 1e-8) -> float:
    expected_probs = np.asarray(expected_probs, dtype=np.float64)
    actual_probs = np.asarray(actual_probs, dtype=np.float64)

    expected_probs = np.clip(expected_probs, eps, None)
    actual_probs = np.clip(actual_probs, eps, None)

    expected_probs = expected_probs / expected_probs.sum()
    actual_probs = actual_probs / actual_probs.sum()

    psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
    return float(psi)


def compute_confidence_drift(ref_stats: dict, recent_confidences: np.ndarray) -> float:
    # Part: Confidence Drift
    # สูตรที่ใช้คือ PSI เปรียบเทียบ distribution ของ confidence
    conf_hist = ref_stats["confidence_histogram"]

    bin_edges = np.asarray(conf_hist["bin_edges"], dtype=np.float32)
    ref_counts = np.asarray(conf_hist["counts"], dtype=np.float32)

    clipped = np.clip(
        recent_confidences,
        bin_edges[0] + 1e-6,
        bin_edges[-1] - 1e-6,
    )

    recent_counts, _ = np.histogram(clipped, bins=bin_edges)

    ref_probs = ref_counts / ref_counts.sum()
    recent_probs = recent_counts / recent_counts.sum()

    return compute_psi(ref_probs, recent_probs)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log2(p / q)))


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_class_ratio_drift(ref_stats: dict, recent_classes: list[str]) -> float:
    # Part: Class Ratio Drift
    # สูตรที่ใช้คือ JSD เปรียบเทียบสัดส่วน class ระหว่าง reference กับ window ล่าสุด
    raw_class_names = ref_stats["model_info"]["classes"]
    class_names = [normalize_label(c) for c in raw_class_names]

    ref_dist_raw = ref_stats["class_distribution_train"]
    ref_probs_map = {"beverage": 0.0, "snack": 0.0}

    for k, v in ref_dist_raw.items():
        if k == "total_samples":
            continue
        ref_probs_map[normalize_label(k)] += float(v)

    ref_probs = np.array([ref_probs_map[c] for c in class_names], dtype=np.float32)
    ref_probs = ref_probs / ref_probs.sum()

    counter = Counter(recent_classes)
    recent_probs = np.array([counter.get(c, 0) for c in class_names], dtype=np.float32)
    recent_probs = recent_probs / recent_probs.sum()

    return float(compute_jsd(ref_probs, recent_probs))


def save_drift_event(result: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO drift_events (
            embedding_score,
            confidence_score,
            class_score,
            embedding_drifted,
            confidence_drifted,
            class_drifted,
            is_drift
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            float(result["embedding_score"]),
            float(result["confidence_score"]),
            float(result["class_score"]),
            int(result["embedding_drifted"]),
            int(result["confidence_drifted"]),
            int(result["class_drifted"]),
            int(result["is_drift"]),
        ),
    )
    conn.commit()
    conn.close()


def main():
    ref_stats = load_reference_stats()
    ref_mean = load_reference_embedding_mean(ref_stats)

    input_size = int(ref_stats["model_info"]["input_size"])
    mean = ref_stats["model_info"]["norm_mean"]
    std = ref_stats["model_info"]["norm_std"]
    class_names = ref_stats["model_info"]["classes"]
    num_classes = len(class_names)

    ids, recent_classes, recent_confidences, image_data_urls = load_n_latest_predictions(WINDOW_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes).to(device)
    transform = build_transform(input_size=input_size, mean=mean, std=std)

    recent_embeddings = infer_recent_embeddings(
        model=model,
        image_data_urls=image_data_urls,
        transform=transform,
        device=device,
        batch_size=BATCH_SIZE,
    )

    embedding_score = compute_embedding_drift(ref_mean, recent_embeddings)
    confidence_score = compute_confidence_drift(ref_stats, recent_confidences)
    class_score = compute_class_ratio_drift(ref_stats, recent_classes)

    embedding_drifted = embedding_score > EMBEDDING_THRESHOLD
    confidence_drifted = confidence_score > CONFIDENCE_THRESHOLD
    class_drifted = class_score > CLASS_THRESHOLD

    result = {
        "prediction_ids": ids,
        "window_size": WINDOW_SIZE,
        "embedding_score": embedding_score,
        "confidence_score": confidence_score,
        "class_score": class_score,
        "embedding_drifted": embedding_drifted,
        "confidence_drifted": confidence_drifted,
        "class_drifted": class_drifted,
        "is_drift": embedding_drifted or confidence_drifted or class_drifted,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_drift_event(result)


if __name__ == "__main__":
    main()