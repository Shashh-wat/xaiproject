"""PyTorch Dataset over the cached explanation maps built by
build_explanation_dataset.py. Each (image, method) pair is one sample; its
label is the *predicted* class, since that's what all three explanations
are justifying and is the contrastive target for SupCon.

Split: by image, never by map. In fold k, every fifth image (sample_id % 5
== k) is held out. All three maps of an image go to the same side, so the
encoder never trains on one method's map of an image and is then tested on
another method's map of that same image. Fold 4 is the default split used by
train_contrastive.py and analyze.py; cross_validate.py uses all five.

Label for the contrastive loss:
  "class"  the class the classifier predicted (maps of the same class are
           positives)
  "image"  the image itself (only the three maps of one image are positives)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.correlate.build_explanation_dataset import MAPS_DIR, META_PATH

N_FOLDS = 5
DEFAULT_FOLD = 4


def is_held_out(sample_id: int, fold: int = DEFAULT_FOLD) -> bool:
    return int(sample_id) % N_FOLDS == fold


class ExplanationMapDataset(Dataset):
    def __init__(self, model_name: str | None = None, split: str | None = None,
                 fold: int = DEFAULT_FOLD, label: str = "class"):
        """split: None (all maps), "train" or "test" (held-out images).
        fold: which fifth of the images is held out. label: "class" or "image"."""
        if label not in ("class", "image"):
            raise ValueError("label must be 'class' or 'image'")
        self.label_mode = label
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"No cached maps at {META_PATH}. Run "
                "`python -m src.correlate.build_explanation_dataset "
                "--model <name>` first."
            )
        df = pd.read_csv(META_PATH)
        if model_name is not None:
            df = df[df["model"] == model_name]
        if len(df) == 0:
            raise ValueError(f"No cached maps for model={model_name!r}.")

        # class index is built from ALL images so train/test share one mapping
        self.classes = sorted(df["pred_class"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split is not None:
            if split not in ("train", "test"):
                raise ValueError("split must be None, 'train' or 'test'")
            held = df["sample_id"].map(lambda i: is_held_out(i, fold))
            df = df[held] if split == "test" else df[~held]
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = np.load(MAPS_DIR / row["file"])
        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # [1, S, S]
        if self.label_mode == "class":
            label = self.class_to_idx[row["pred_class"]]
        else:
            label = int(row["sample_id"])
        return tensor, label, row["method"], row["sample_id"]
