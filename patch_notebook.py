import json
from pathlib import Path

notebook_path = Path("product-categorization-system.ipynb")
with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])

    # 1. Imports
    if "from huggingface_hub import hf_hub_download" in source:
        source = source.replace("from sklearn.metrics import (", "from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, default_data_collator\nfrom sklearn.metrics import (")
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 2. DataConfig
    if "class DataConfig:" in source:
        source = source.replace('["beverages", "snacks", "dry_food", "non_food"]', '["beverage", "snack"]')
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 3. TrainConfig
    if "class TrainConfig:" in source:
        source = source.replace('model_name: str = "resnet18"', 'model_name: str = "resnet50"')
        source = source.replace('output_dir: Path = Path("outputs")', 'output_dir: Path = Path("runs")')
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 4. ProductDataset
    if "class ProductDataset(Dataset):" in source:
        # Replace return type hints
        source = source.replace('image  : torch.FloatTensor  shape (C, H, W) after transform\n    label  : torch.LongTensor   scalar integer class index', 'dict with keys:\n      "pixel_values" : torch.FloatTensor  shape (C, H, W) after transform\n      "labels"       : torch.LongTensor   scalar integer class index')

        # Add logic to remap and filter labels
        filter_code = """
        # ── 1a. Remap and Filter to 2 classes ("snack", "beverage") ─────────────
        def remap_label(lbl: str) -> str:
            lbl = str(lbl).lower().strip()
            if lbl in ["snacks", "snack"]: return "snack"
            if lbl in ["beverages", "beverage"]: return "beverage"
            return lbl

        manifest["label_coarse"] = manifest["label_coarse"].apply(remap_label)
        manifest = manifest[manifest["label_coarse"].isin(["snack", "beverage"])].copy()

        # ── 1b. Filter out corrupted images ───────────────────────────────"""
        source = source.replace('# ── 2. Filter by split', filter_code + '\n\n        # ── 2. Filter by split')
        
        # Override label map
        label_map_code = """        # ── 3. Load label_map ─────────────────────────────────────────────
        # Ignore external label map and enforce 2 classes
        self._label_map: Dict[str, int] = {"beverage": 0, "snack": 1}"""
        source = source.replace('        # ── 3. Load label_map ─────────────────────────────────────────────\n        if isinstance(label_map, (str, Path)):\n            label_map = json.loads(Path(label_map).read_text(encoding="utf-8"))\n\n        self._label_map: Dict[str, int] = label_map', label_map_code)

        # Fix getitem
        source = source.replace('def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:', 'def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:')
        source = source.replace('return image, label  # type: ignore[return-value]', 'return {"pixel_values": image, "labels": label}')

        # Fix build_datasets
        source = source.replace('    label_map: Dict[str, int] = json.loads(\n        Path(label_map_path).read_text(encoding="utf-8")\n    )', '    label_map = {"beverage": 0, "snack": 1}')
        
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 5. _TransferModel
    if "class _TransferModel(nn.Module):" in source:
        forward_replacement = """    def forward(self, pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None):
        inputs = pixel_values if pixel_values is not None else x
        logits = self._backbone(inputs)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return (loss, logits)
        return logits"""
        source = source.replace('    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self._backbone(x)', forward_replacement)
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 6. Factory functions
    if "def _build_resnet18" in source:
        resnet50_code = """def _build_resnet50(num_classes: int, freeze_backbone: bool, dropout: float) -> _TransferModel:
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    nn.init.xavier_uniform_(backbone.fc[1].weight)
    nn.init.zeros_(backbone.fc[1].bias)
    return _TransferModel(backbone, num_classes, freeze_backbone)


def _build_mobilenetv3_large(num_classes: int, freeze_backbone: bool, dropout: float) -> _TransferModel:
    backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    nn.init.xavier_uniform_(backbone.classifier[-1][1].weight)
    nn.init.zeros_(backbone.classifier[-1][1].bias)
    return _TransferModel(backbone, num_classes, freeze_backbone)"""
        
        import re
        source = re.sub(r'def _build_resnet18[\s\S]*?return _TransferModel\(backbone, num_classes, freeze_backbone\)', resnet50_code, source)

        registry_code = """_REGISTRY = {
    "efficientnet_b0": lambda nc, fb, do: ProductClassifier(num_classes=nc, freeze_backbone=fb, dropout=do),
    "simple_cnn": lambda nc, fb, do: SimpleCNN(num_classes=nc, dropout=do),
    "resnet50": _build_resnet50,
    "mobilenetv3_large": _build_mobilenetv3_large,
}

ModelName = Literal["efficientnet_b0", "simple_cnn", "resnet50", "mobilenetv3_large"]"""
        
        source = re.sub(r'_REGISTRY = \{[\s\S]*?ModelName = Literal\[.*?\]', registry_code, source)
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 7. Trainer class removal (we just replace the cell to define versioning and metrics)
    if "class Trainer:" in source:
        trainer_replacement = """def get_next_run_dir(base_dir: Path, model_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_tests = []
    pattern = re.compile(rf"^{model_name}_test(\d+)$")
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing_tests.append(int(match.group(1)))
    next_num = max(existing_tests) + 1 if existing_tests else 1
    return base_dir / f"{model_name}_test{next_num}"

# HF compute metrics
def compute_metrics_hf(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}
"""
        cell["source"] = [line + "\n" for line in trainer_replacement.split("\n")[:-1]] + [trainer_replacement.split("\n")[-1]]

    # 8. Train Settings
    if "MODEL_NAME = " in source:
        source = source.replace('MODEL_NAME = "resnet18"          # "efficientnet_b0" | "resnet18" | "mobilenetv2" | "simple_cnn"', 'MODEL_NAME = "resnet50"          # "efficientnet_b0" | "resnet50" | "mobilenetv3_large" | "simple_cnn"')
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 9. Train Execute
    if "trainer = Trainer(" in source:
        train_huggingface = """# ── Trainer Base Directory ────────────────────────────────────────────
run_dir = get_next_run_dir(cfg.output_dir, cfg.model_name)
print(f"\\nOutput directory for this run: {run_dir}\\n")

# ── Hugging Face Trainer ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(run_dir),
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=cfg.lr,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.epochs,
    weight_decay=cfg.weight_decay,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    dataloader_num_workers=cfg.num_workers,
    seed=cfg.seed,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics_hf,
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting Hugging Face Trainer loop ...")
trainer.train()

trainer.save_model(str(run_dir / "best_model"))
with open(run_dir / "metrics.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"\\nTraining complete! Results saved in {run_dir}")
"""
        import re
        source = re.sub(r'# Train\ntrainer = Trainer\([\s\S]*?trainer\.fit\(\)', train_huggingface, source)
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

    # 10. Eval Test set
    if "device = trainer.device" in source:
        # Simple fix: device is model device, best ckpt is best_model
        source = source.replace('device = trainer.device', 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
        source = source.replace('best_path = trainer.best_ckpt_path', 'best_path = run_dir / "best_model"')
        source = source.replace('ckpt = torch.load(best_path, map_location=str(device))\n    model.load_state_dict(ckpt["model_state_dict"])', 'from transformers import PreTrainedModel\n    # Wait, we can just use the model object directly if we didn\'t overwrite it, load_best_model implies `model` is the best')
        cell["source"] = [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]]

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched.")
