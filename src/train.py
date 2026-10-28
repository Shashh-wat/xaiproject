"""Train a baseline model (MobileNetV2 or ResNet50) on PlantVillage.

Usage:
    python -m src.train --model mobilenet_v2 --epochs 10
    python -m src.train --model resnet50 --epochs 10 --unfreeze_last_n 2
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG, DEVICE
from src.data.dataset import build_datasets, class_weights
from src.models.factory import build_model, trainable_parameters


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train_mode:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--unfreeze_last_n", type=int, default=0,
                         help="Unfreeze the last N backbone blocks/layers "
                              "for fine-tuning (0 = frozen backbone, "
                              "linear-probe only).")
    args = parser.parse_args()

    print(f"[train] device={DEVICE}")
    train_ds, val_ds, test_ds, class_names = build_datasets()
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val / "
          f"{len(test_ds)} test images, {len(class_names)} classes")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=CFG.num_workers)

    model = build_model(args.model, len(class_names),
                         unfreeze_last_n=args.unfreeze_last_n).to(DEVICE)

    weights = class_weights(train_ds.labels, len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    param_groups = trainable_parameters(model, CFG.lr_head, CFG.lr_backbone)
    optimizer = torch.optim.Adam(param_groups, weight_decay=CFG.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": []}
    best_val_acc = -1.0
    ckpt_path = CFG.checkpoints_dir / f"{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                     optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[train] epoch {epoch}/{args.epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, ckpt_path)
            print(f"[train] new best val_acc={val_acc:.4f}, "
                  f"saved to {ckpt_path}")

    hist_path = CFG.checkpoints_dir / f"{args.model}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] done. best val_acc={best_val_acc:.4f}. "
          f"history -> {hist_path}")


if __name__ == "__main__":
    main()
