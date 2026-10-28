"""Trains the Siamese/SupCon explanation-correlation encoder on the cached
Grad-CAM / SHAP / LIME maps.

Trains on the training images only. The SupCon loss on the held-out images
is printed next to the training loss, so over-fitting is visible.

Usage:
    python -m src.correlate.build_explanation_dataset --model mobilenet_v2 --n_images 60
    python -m src.correlate.train_contrastive --model mobilenet_v2 --epochs 100
    (add --label image to treat only the three maps of one image as positives)
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.config import CFG, DEVICE
from src.correlate.dataset_loader import ExplanationMapDataset
from src.correlate.encoder import ExplanationEncoder
from src.correlate.losses import SupConLoss

CKPT_DIR = CFG.outputs_dir / "correlate" / "checkpoints"


@torch.no_grad()
def held_out_loss(encoder, ds, criterion):
    if len(ds) == 0:
        return float("nan")
    encoder.eval()
    maps = torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)
    labels = torch.tensor([ds[i][1] for i in range(len(ds))]).to(DEVICE)
    return criterion(encoder(maps), labels).item()


def fit(train_ds, test_ds, epochs=100, batch_size=64, embed_dim=64,
        temperature=0.1, lr=1e-3, seed=0, log=True):
    """Trains an encoder on train_ds. Returns (encoder, history)."""
    torch.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                         drop_last=len(train_ds) > batch_size)
    encoder = ExplanationEncoder(embed_dim=embed_dim).to(DEVICE)
    criterion = SupConLoss(temperature=temperature)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    history = {"train_loss": [], "held_out_loss": []}
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss, n_batches = 0.0, 0
        for maps, labels, _methods, _ids in loader:
            maps, labels = maps.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(encoder(maps), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        ho_loss = held_out_loss(encoder, test_ds, criterion)
        history["train_loss"].append(avg_loss)
        history["held_out_loss"].append(ho_loss)
        if log and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"[correlate-train] epoch {epoch}/{epochs} "
                  f"train_loss={avg_loss:.4f} held_out_loss={ho_loss:.4f}")
    encoder.eval()
    return encoder, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"],
                         help="Which cached-map set to train on (see "
                              "build_explanation_dataset.py).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=["class", "image"], default="class")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = ExplanationMapDataset(model_name=args.model, split="train",
                                     label=args.label)
    test_ds = ExplanationMapDataset(model_name=args.model, split="test",
                                    label=args.label)
    print(f"[correlate-train] train maps: {len(train_ds)} "
          f"({len(train_ds) // 3} images), held-out maps: {len(test_ds)} "
          f"({len(test_ds) // 3} images), label: {args.label}")

    encoder, history = fit(train_ds, test_ds, epochs=args.epochs,
                           batch_size=args.batch_size, embed_dim=args.embed_dim,
                           temperature=args.temperature, lr=args.lr,
                           seed=args.seed)

    ckpt_path = CKPT_DIR / f"encoder_{args.model}.pt"
    torch.save({
        "state_dict": encoder.state_dict(),
        "embed_dim": args.embed_dim,
        "classes": train_ds.classes,
        "trained_on_model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "label": args.label,
        "epochs": args.epochs,
    }, ckpt_path)
    with open(CKPT_DIR / f"encoder_{args.model}_history.json", "w") as f:
        json.dump(history, f)
    print(f"[correlate-train] saved encoder -> {ckpt_path}")


if __name__ == "__main__":
    main()
