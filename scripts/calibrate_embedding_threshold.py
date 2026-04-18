import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import convnext_tiny

REFERENCE_STATS_PATH = Path("reference/reference_stats.json")
REFERENCE_EMBEDDINGS_PATH = Path("reference/reference_embeddings.npz")
CHECKPOINT_PATH = Path("model.safetensors")
TEST_ROOT = Path("test")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PER_CLASS_COUNT = 50
NUM_ROUNDS = 30
SEED = 42
BATCH_SIZE = 32


class ConvNextTinyWithEmbedding(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier[0](x)
        embedding = torch.flatten(x, 1)
        logits = self.backbone.classifier[2](embedding)
        return logits, embedding


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


def load_reference_stats() -> dict:
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(f"reference_stats.json not found: {REFERENCE_STATS_PATH}")

    with open(REFERENCE_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_embedding_mean(ref_stats: dict) -> np.ndarray:
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


def build_model(num_classes: int):
    model = ConvNextTinyWithEmbedding(num_classes=num_classes)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"model.safetensors not found: {CHECKPOINT_PATH}")

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


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    if not files:
        raise ValueError(f"no image files found under: {folder}")

    return sorted(files)


def sample_no_drift_images(
    test_root: Path,
    per_class_count: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    sampled = []

    for class_name in ["beverage", "snack"]:
        folder = test_root / "no_drift" / class_name
        candidates = list_images(folder)

        if len(candidates) < per_class_count:
            raise ValueError(
                f"not enough images in {folder}: need {per_class_count}, got {len(candidates)}"
            )

        chosen = rng.sample(candidates, per_class_count)
        sampled.extend(chosen)

    rng.shuffle(sampled)
    return sampled


def infer_embeddings_from_paths(
    model: nn.Module,
    image_paths: list[Path],
    transform,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_tensors = []

        for image_path in batch_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                tensor = transform(img)
                batch_tensors.append(tensor)

        x = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            _, embeddings = model(x)

        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


def compute_embedding_drift(ref_mean: np.ndarray, recent_embeddings: np.ndarray) -> float:
    recent_mean = recent_embeddings.mean(axis=0)
    return float(np.linalg.norm(recent_mean - ref_mean))


def main():
    ref_stats = load_reference_stats()
    ref_mean = load_reference_embedding_mean(ref_stats)

    input_size = int(ref_stats["model_info"]["input_size"])
    mean = ref_stats["model_info"]["norm_mean"]
    std = ref_stats["model_info"]["norm_std"]
    class_names = ref_stats["model_info"]["classes"]
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=num_classes).to(device)
    transform = build_transform(input_size=input_size, mean=mean, std=std)

    scores = []

    for round_idx in range(NUM_ROUNDS):
        round_seed = SEED + round_idx

        image_paths = sample_no_drift_images(
            test_root=TEST_ROOT,
            per_class_count=PER_CLASS_COUNT,
            seed=round_seed,
        )

        recent_embeddings = infer_embeddings_from_paths(
            model=model,
            image_paths=image_paths,
            transform=transform,
            device=device,
            batch_size=BATCH_SIZE,
        )

        score = compute_embedding_drift(ref_mean, recent_embeddings)
        scores.append(score)

        print(
            f"round={round_idx + 1:02d} "
            f"sample_size={len(image_paths)} "
            f"embedding_score={score:.6f}"
        )

    scores = np.asarray(scores, dtype=np.float64)

    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    min_score = float(scores.min())
    max_score = float(scores.max())
    p90 = float(np.percentile(scores, 90))
    p95 = float(np.percentile(scores, 95))
    p99 = float(np.percentile(scores, 99))
    mean_plus_2std = float(mean_score + 2.0 * std_score)
    mean_plus_3std = float(mean_score + 3.0 * std_score)

    summary = {
        "num_rounds": int(NUM_ROUNDS),
        "per_class_count": int(PER_CLASS_COUNT),
        "window_size": int(PER_CLASS_COUNT * 2),
        "scores": [float(x) for x in scores],
        "stats": {
            "min": min_score,
            "max": max_score,
            "mean": mean_score,
            "std": std_score,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "mean_plus_2std": mean_plus_2std,
            "mean_plus_3std": mean_plus_3std,
        },
        "recommended_thresholds": {
            "conservative": p99,
            "balanced": p95,
            "strict": mean_plus_3std,
        },
    }

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    output_path = Path("reference/embedding_threshold_calibration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nsaved to: {output_path}")


if __name__ == "__main__":
    main()