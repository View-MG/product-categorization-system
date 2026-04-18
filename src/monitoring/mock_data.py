import sys
import io
import time
import base64
import random
import sqlite3
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import convnext_tiny

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.monitoring.store import DB_PATH, insert_prediction, insert_feedback

REFERENCE_STATS_PATH = ROOT_DIR / "reference" / "reference_stats.json"
CHECKPOINT_PATH = ROOT_DIR / "model.safetensors"
TEST_ROOT = ROOT_DIR / "test"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PER_CLASS_COUNT = 50
SEED = 42
CLEAR_FIRST = True
WRITE_FEEDBACK = True
RUN_MODE = "no_drift"  # "no_drift", "drift", "both"


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


def load_reference_stats() -> dict:
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(f"reference_stats.json not found: {REFERENCE_STATS_PATH}")

    import json
    with open(REFERENCE_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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

    model.load_state_dict(backbone_weights, strict=False)
    model.eval()
    return model


def build_transform(input_size: int, mean: list[float], std: list[float]):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def pil_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def compute_image_metrics(img: Image.Image):
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    blur_var = float(np.var(gx) + np.var(gy))

    width, height = img.size

    warnings = []
    if brightness < 35:
        warnings.append("low_brightness")
    if blur_var < 20:
        warnings.append("low_blur")

    return round(brightness, 2), round(blur_var, 2), width, height, warnings


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


def extract_true_label_after_prediction(image_path: Path) -> str:
    stem = image_path.stem.lower()
    tokens = [t for t in re.split(r"[^a-zA-Z]+", stem) if t]

    for t in tokens:
        if t in ("beverage", "beverages"):
            return "beverage"
        if t in ("snack", "snacks"):
            return "snack"

    parts = [p.lower() for p in image_path.parts]
    if "beverage" in parts or "beverages" in parts:
        return "beverage"
    if "snack" in parts or "snacks" in parts:
        return "snack"

    raise ValueError(f"cannot infer true label from filename/path: {image_path}")


def clear_prediction_related_tables(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM human_feedback")
    conn.execute("DELETE FROM drift_events")
    conn.execute("DELETE FROM alerts")
    conn.execute("DELETE FROM prediction_events")
    conn.commit()
    conn.close()


def predict_one(model, transform, device, class_names_singular: list[str], image: Image.Image):
    x = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    pred_idx = int(torch.argmax(probs).item())
    predicted_class = class_names_singular[pred_idx]
    confidence = float(probs[pred_idx].item())

    return predicted_class, confidence, round(latency_ms, 2)


def sample_balanced_images_from_bucket(
    test_root: Path,
    bucket: str,
    per_class_count: int,
    seed: int,
) -> list[dict]:
    if bucket not in {"no_drift", "drift"}:
        raise ValueError(f"invalid bucket: {bucket}")

    rng = random.Random(seed)
    sampled = []

    class_names = ["beverage", "snack"]

    for class_name in class_names:
        folder = test_root / bucket / class_name
        candidates = list_images(folder)

        if len(candidates) < per_class_count:
            raise ValueError(
                f"not enough images in {folder}: need {per_class_count}, got {len(candidates)}"
            )

        chosen = rng.sample(candidates, per_class_count)

        for path in chosen:
            sampled.append({
                "bucket": bucket,
                "path": path,
            })

    rng.shuffle(sampled)
    return sampled


def run_bucket_prediction_flow(
    bucket: str,
    test_root: Path = TEST_ROOT,
    per_class_count: int = PER_CLASS_COUNT,
    clear_first: bool = False,
    write_feedback: bool = True,
    seed: int = SEED,
):
    ref_stats = load_reference_stats()

    raw_class_names = ref_stats["model_info"]["classes"]
    class_names_singular = [normalize_label(c) for c in raw_class_names]

    input_size = int(ref_stats["model_info"]["input_size"])
    mean = ref_stats["model_info"]["norm_mean"]
    std = ref_stats["model_info"]["norm_std"]

    model = build_model(num_classes=len(class_names_singular))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = build_transform(input_size=input_size, mean=mean, std=std)

    samples = sample_balanced_images_from_bucket(
        test_root=test_root,
        bucket=bucket,
        per_class_count=per_class_count,
        seed=seed,
    )

    if clear_first:
        clear_prediction_related_tables()

    inserted = 0
    skipped = 0
    pred_counter = {"beverage": 0, "snack": 0}

    for item in samples:
        image_path = item["path"]

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                image_data_url = pil_image_to_data_url(img)

                predicted_class, confidence, latency_ms = predict_one(
                    model=model,
                    transform=transform,
                    device=device,
                    class_names_singular=class_names_singular,
                    image=img,
                )

                brightness, blur_var, width, height, warnings = compute_image_metrics(img)

            prediction_id = insert_prediction(
                predicted_class=predicted_class,
                confidence=confidence,
                latency_ms=latency_ms,
                brightness=brightness,
                blur_var=blur_var,
                width=width,
                height=height,
                quality_warnings=warnings,
                image_data_url=image_data_url,
            )

            if write_feedback:
                true_label = extract_true_label_after_prediction(image_path)
                insert_feedback(
                    prediction_id=prediction_id,
                    true_label=true_label,
                )
            else:
                true_label = None

            inserted += 1
            pred_counter[predicted_class] += 1

            print(
                f"[OK] id={prediction_id} "
                f"bucket={bucket} "
                f"true={true_label} "
                f"pred={predicted_class} "
                f"conf={confidence:.4f} "
                f"file={image_path.name}"
            )

        except Exception as e:
            skipped += 1
            print(f"[SKIP] bucket={bucket} file={image_path}: {e}")

    print()
    print(f"bucket={bucket} done. inserted={inserted}, skipped={skipped}")
    print(f"bucket={bucket} pred_counter={pred_counter}")


def run_no_drift_prediction_flow(
    test_root: Path = TEST_ROOT,
    per_class_count: int = PER_CLASS_COUNT,
    clear_first: bool = True,
    write_feedback: bool = True,
    seed: int = SEED,
):
    run_bucket_prediction_flow(
        bucket="no_drift",
        test_root=test_root,
        per_class_count=per_class_count,
        clear_first=clear_first,
        write_feedback=write_feedback,
        seed=seed,
    )


def run_drift_prediction_flow(
    test_root: Path = TEST_ROOT,
    per_class_count: int = PER_CLASS_COUNT,
    clear_first: bool = True,
    write_feedback: bool = True,
    seed: int = SEED,
):
    run_bucket_prediction_flow(
        bucket="drift",
        test_root=test_root,
        per_class_count=per_class_count,
        clear_first=clear_first,
        write_feedback=write_feedback,
        seed=seed,
    )


if __name__ == "__main__":
    if RUN_MODE == "no_drift":
        run_no_drift_prediction_flow(
            test_root=TEST_ROOT,
            per_class_count=PER_CLASS_COUNT,
            clear_first=CLEAR_FIRST,
            write_feedback=WRITE_FEEDBACK,
            seed=SEED,
        )
    elif RUN_MODE == "drift":
        run_drift_prediction_flow(
            test_root=TEST_ROOT,
            per_class_count=PER_CLASS_COUNT,
            clear_first=CLEAR_FIRST,
            write_feedback=WRITE_FEEDBACK,
            seed=SEED,
        )
    elif RUN_MODE == "both":
        run_no_drift_prediction_flow(
            test_root=TEST_ROOT,
            per_class_count=PER_CLASS_COUNT,
            clear_first=CLEAR_FIRST,
            write_feedback=WRITE_FEEDBACK,
            seed=SEED,
        )

        run_drift_prediction_flow(
            test_root=TEST_ROOT,
            per_class_count=PER_CLASS_COUNT,
            clear_first=False,
            write_feedback=WRITE_FEEDBACK,
            seed=SEED + 1,
        )
    else:
        raise ValueError(f"invalid RUN_MODE: {RUN_MODE}")