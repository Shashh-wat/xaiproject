"""Evaluate a trained checkpoint on the held-out test set.

Usage:
    python -m src.evaluate --model mobilenet_v2
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
from torch.utils.data import DataLoader

from src.config import CFG, DEVICE
from src.data.dataset import build_datasets
from src.models.factory import build_model


def load_checkpoint(model_name: str):
    ckpt_path = CFG.checkpoints_dir / f"{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run "
            f"`python -m src.train --model {model_name}` first."
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(model_name, len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["class_names"]


@torch.no_grad()
def predict_all(model, loader):
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        y_pred += logits.argmax(1).cpu().tolist()
        y_true += y.tolist()
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    args = parser.parse_args()

    model, class_names = load_checkpoint(args.model)
    _, _, test_ds, _ = build_datasets()
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=CFG.num_workers)

    print(f"[evaluate] running inference on {len(test_ds)} test images ...")
    y_true, y_pred = predict_all(model, loader)

    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[evaluate] {args.model}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
    report_path = CFG.eval_dir / f"{args.model}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, **report}, f,
                   indent=2)
    print(f"[evaluate] wrote {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{args.model} confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    cm_path = CFG.eval_dir / f"{args.model}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] wrote {cm_path}")


if __name__ == "__main__":
    main()
