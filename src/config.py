"""Central configuration: paths, hyperparameters, dataset constants.

Nothing in here talks to disk or the network on import -- it's just values,
so every other module can `from src.config import CFG` safely.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- paths ---
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw" / "plantvillage"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    checkpoints_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    eda_dir: Path = PROJECT_ROOT / "outputs" / "eda"
    eval_dir: Path = PROJECT_ROOT / "outputs" / "eval"
    explanations_dir: Path = PROJECT_ROOT / "outputs" / "explanations"
    comparison_dir: Path = PROJECT_ROOT / "outputs" / "comparison"
    robustness_dir: Path = PROJECT_ROOT / "outputs" / "robustness"

    # --- Kaggle dataset ---
    kaggle_dataset: str = "emmarex/plantdisease"

    # --- data ---
    image_size: int = 224
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42

    # --- training ---
    batch_size: int = 32
    # 0 by default: PyTorch DataLoader multiprocessing workers can hang
    # indefinitely on macOS (fork/spawn + framework libs like matplotlib/
    # torch loaded pre-fork) -- confirmed hanging on this machine. Raise
    # this only after verifying >0 actually works in your environment.
    num_workers: int = 0
    epochs: int = 10
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    # --- explainability ---
    shap_background_samples: int = 50
    lime_num_samples: int = 500

    def ensure_dirs(self) -> None:
        for d in [
            self.data_root,
            self.outputs_dir,
            self.checkpoints_dir,
            self.eda_dir,
            self.eval_dir,
            self.explanations_dir,
            self.comparison_dir,
            self.robustness_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
CFG.ensure_dirs()

# Model registry: name -> (target layer accessor name, human label)
SUPPORTED_MODELS = {
    "mobilenet_v2": "MobileNetV2",
    "resnet50": "ResNet50",
}

DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"
try:
    import torch

    if os.environ.get("FORCE_CPU") == "1":
        DEVICE = "cpu"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except Exception:  # pragma: no cover - torch not installed yet
    DEVICE = "cpu"
