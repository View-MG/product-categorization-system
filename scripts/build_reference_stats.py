from __future__ import annotations

import importlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent

MANIFEST_PATH = PROJECT_ROOT / "manifest_clean.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "model_checkpoint.pth"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "reference"

IMAGE_COL = "abs_path"
LABEL_COL = "label_coarse"

MODEL_BUILDER = "your_module:build_model"
MODEL_KWARGS: dict[str, Any] = {}
TRANSFORM_BUILDER = ""
EMBEDDING_LAYER = "backbone.avgpool"

BATCH_SIZE = 32
NUM_WORKERS = 0
INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SAMPLE_SIZE = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFERENCE_VERSION = "v1"
MODEL_VERSION = "unknown"

BRIGHTNESS_LOW_THRESHOLD = 60.0
BRIGHTNESS_HIGH_THRESHOLD = 220.0
BLUR_THRESHOLD = 80.0


def import_from_spec(spec: str) -> Callable[..., Any]:
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def normalize_path(raw_path: Any) -> Path:
    s = str(raw_path).strip().replace("\\", "/")
    p = Path(s)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def compute_brightness(image_np: np.ndarray) -> float:
    gray = np.asarray(Image.fromarray(image_np).convert("L"), dtype=np.float32)
    return float(gray.mean())


def compute_blur_variance(image_np: np.ndarray) -> float:
    gray = np.asarray(Image.fromarray(image_np).convert("L"), dtype=np.uint8)
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gray_f = gray.astype(np.float32)
    gy, gx = np.gradient(gray_f)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    lap = gxx + gyy
    return float(np.var(lap))


def default_transform(input_size: int, mean: list[float], std: list[float]) -> Callable[[Image.Image], torch.Tensor]:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class ManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        label_col: str,
        transform: Callable[[Image.Image], torch.Tensor],
        brightness_low: float,
        brightness_high: float,
        blur_threshold: float,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.blur_threshold = blur_threshold

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = normalize_path(row[self.image_col])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        brightness = compute_brightness(image_np)
        blur_var = compute_blur_variance(image_np)
        height, width = image_np.shape[:2]

        warnings = []
        if brightness < self.brightness_low:
            warnings.append("low_brightness")
        if brightness > self.brightness_high:
            warnings.append("high_brightness")
        if blur_var < self.blur_threshold:
            warnings.append("blurry")

        return {
            "tensor": self.transform(image),
            "path": str(img_path),
            "label": str(row[self.label_col]),
            "brightness": float(brightness),
            "blur_var": float(blur_var),
            "width": int(width),
            "height": int(height),
            "warnings": warnings,
        }


class EmbeddingExtractor:
    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.buffer: torch.Tensor | None = None
        self.handle = self._register_hook(layer_name)

    def _register_hook(self, layer_name: str):
        modules = dict(self.model.named_modules())
        if layer_name not in modules:
            raise ValueError(f"Embedding layer '{layer_name}' not found")

        def hook(_module, _inputs, output):
            self.buffer = normalize_embedding_output(output)

        return modules[layer_name].register_forward_hook(hook)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.buffer = None
        out = self.model(x)
        logits = normalize_logits_output(out)
        if self.buffer is None:
            raise RuntimeError(f"Hook on '{self.layer_name}' did not capture output")
        return logits, self.buffer

    def close(self):
        self.handle.remove()


def normalize_logits_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for key in ("logits", "output", "pred", "prediction"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(f"Unsupported model output type: {type(output)}")


def normalize_embedding_output(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                output = item
                break

    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                output = value
                break

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported embedding output type: {type(output)}")

    if output.ndim == 4:
        output = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
    elif output.ndim == 3:
        output = output.mean(dim=1)
    elif output.ndim == 1:
        output = output.unsqueeze(0)
    elif output.ndim != 2:
        output = output.flatten(1)

    return output


def build_model() -> nn.Module:
    builder = import_from_spec(MODEL_BUILDER)
    model = builder(**MODEL_KWARGS)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_state_dict[k.replace("module.", "")] = v

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model


def build_transform():
    if TRANSFORM_BUILDER:
        return import_from_spec(TRANSFORM_BUILDER)()
    return default_transform(INPUT_SIZE, MEAN, STD)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tensor": torch.stack([item["tensor"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "label": [item["label"] for item in batch],
        "brightness": np.array([item["brightness"] for item in batch], dtype=np.float32),
        "blur_var": np.array([item["blur_var"] for item in batch], dtype=np.float32),
        "width": np.array([item["width"] for item in batch], dtype=np.int32),
        "height": np.array([item["height"] for item in batch], dtype=np.int32),
        "warnings": [item["warnings"] for item in batch],
    }


def stratified_sample_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(labels)

    if sample_size >= n:
        return np.arange(n)

    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    allocated = np.floor(proportions * sample_size).astype(int)

    remainder = sample_size - allocated.sum()
    fractional = proportions * sample_size - allocated
    order = np.argsort(-fractional)

    for i in range(remainder):
        allocated[order[i % len(order)]] += 1

    picked = []
    for label, take in zip(unique, allocated):
        idxs = np.where(labels == label)[0]
        if take >= len(idxs):
            picked.extend(idxs.tolist())
        else:
            picked.extend(rng.choice(idxs, size=take, replace=False).tolist())

    picked = np.array(picked, dtype=np.int64)
    rng.shuffle(picked)
    return picked


def safe_histogram_probs(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, used_bins = np.histogram(values, bins=bins)
    probs = counts.astype(np.float64)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs, used_bins


def summarize_warning_rates(all_warnings: list[list[str]], n_total: int) -> dict[str, float]:
    counter = defaultdict(int)
    for warnings in all_warnings:
        for w in warnings:
            counter[w] += 1
    return {k: counter[k] / n_total for k in sorted(counter)}


def to_builtin(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    return x


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUTPUT_DIR / "reference_stats.json"
    npz_path = OUTPUT_DIR / "reference_embeddings.npz"

    df = pd.read_csv(MANIFEST_PATH)
    required_cols = {IMAGE_COL, LABEL_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {sorted(missing_cols)}")

    dataset = ManifestDataset(
        df=df,
        image_col=IMAGE_COL,
        label_col=LABEL_COL,
        transform=build_transform(),
        brightness_low=BRIGHTNESS_LOW_THRESHOLD,
        brightness_high=BRIGHTNESS_HIGH_THRESHOLD,
        blur_threshold=BLUR_THRESHOLD,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and DEVICE.startswith("cuda"),
        collate_fn=collate_fn,
    )

    model = build_model().to(DEVICE)
    extractor = EmbeddingExtractor(model, EMBEDDING_LAYER)

    all_labels = []
    all_paths = []
    all_brightness = []
    all_blur_var = []
    all_width = []
    all_height = []
    all_warnings = []
    all_confidence = []
    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            x = batch["tensor"].to(DEVICE)
            logits, emb = extractor.forward(x)

            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values.detach().cpu().numpy().astype(np.float32)
            emb_np = emb.detach().cpu().numpy().astype(np.float32)

            all_labels.extend(batch["label"])
            all_paths.extend(batch["path"])
            all_brightness.extend(batch["brightness"].tolist())
            all_blur_var.extend(batch["blur_var"].tolist())
            all_width.extend(batch["width"].tolist())
            all_height.extend(batch["height"].tolist())
            all_warnings.extend(batch["warnings"])
            all_confidence.extend(confidence.tolist())
            all_embeddings.extend(list(emb_np))

    extractor.close()

    labels_np = np.array(all_labels)
    brightness_np = np.array(all_brightness, dtype=np.float32)
    blur_np = np.array(all_blur_var, dtype=np.float32)
    width_np = np.array(all_width, dtype=np.int32)
    height_np = np.array(all_height, dtype=np.int32)
    confidence_np = np.array(all_confidence, dtype=np.float32)
    embeddings_np = np.stack(all_embeddings).astype(np.float32)

    class_counts = pd.Series(labels_np).value_counts().sort_index()
    class_distribution = (class_counts / class_counts.sum()).to_dict()

    conf_bins = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    conf_probs, conf_bins = safe_histogram_probs(confidence_np, conf_bins)

    sampled_idx = stratified_sample_indices(labels_np, SAMPLE_SIZE, SEED)
    sampled_embeddings = embeddings_np[sampled_idx]
    sampled_labels = labels_np[sampled_idx]
    sampled_paths = np.array(all_paths, dtype=object)[sampled_idx]

    np.savez_compressed(
        npz_path,
        sample_embeddings=sampled_embeddings,
        sample_labels=sampled_labels,
        sample_paths=sampled_paths,
        embedding_mean=embeddings_np.mean(axis=0).astype(np.float32),
        embedding_std=embeddings_np.std(axis=0).astype(np.float32),
    )

    stats = {
        "reference_version": REFERENCE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(MANIFEST_PATH),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "model_version": MODEL_VERSION,
        "train_size": int(len(df)),
        "label_column": LABEL_COL,
        "image_column": IMAGE_COL,
        "class_distribution": class_distribution,
        "confidence_distribution": {
            "mean": float(confidence_np.mean()),
            "std": float(confidence_np.std()),
            "min": float(confidence_np.min()),
            "max": float(confidence_np.max()),
            "q05": float(np.quantile(confidence_np, 0.05)),
            "q25": float(np.quantile(confidence_np, 0.25)),
            "q50": float(np.quantile(confidence_np, 0.50)),
            "q75": float(np.quantile(confidence_np, 0.75)),
            "q95": float(np.quantile(confidence_np, 0.95)),
            "histogram_bins": conf_bins.tolist(),
            "histogram_probs": conf_probs.tolist(),
        },
        "embedding_stats": {
            "dimension": int(embeddings_np.shape[1]),
            "sample_size_requested": int(SAMPLE_SIZE),
            "sample_size_actual": int(len(sampled_idx)),
            "npz_file": str(npz_path),
            "keys": {
                "sample_embeddings": "sample_embeddings",
                "sample_labels": "sample_labels",
                "sample_paths": "sample_paths",
                "embedding_mean": "embedding_mean",
                "embedding_std": "embedding_std",
            },
        },
        "image_quality": {
            "brightness_mean": float(brightness_np.mean()),
            "brightness_std": float(brightness_np.std()),
            "brightness_min": float(brightness_np.min()),
            "brightness_max": float(brightness_np.max()),
            "blur_var_mean": float(blur_np.mean()),
            "blur_var_std": float(blur_np.std()),
            "blur_var_min": float(blur_np.min()),
            "blur_var_max": float(blur_np.max()),
            "width_mean": float(width_np.mean()),
            "height_mean": float(height_np.mean()),
            "warning_rates": summarize_warning_rates(all_warnings, len(df)),
            "thresholds": {
                "brightness_low": float(BRIGHTNESS_LOW_THRESHOLD),
                "brightness_high": float(BRIGHTNESS_HIGH_THRESHOLD),
                "blur_var_low": float(BLUR_THRESHOLD),
            },
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(stats), f, ensure_ascii=False, indent=2)

    print(f"Saved: {stats_path}")
    print(f"Saved: {npz_path}")
    print(f"train_size={len(df)}, embedding_dim={embeddings_np.shape[1]}, sampled={len(sampled_idx)}")


if __name__ == "__main__":
    main()