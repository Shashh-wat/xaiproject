"""5-fold cross-validation of the correlation encoder over all cached images.

In fold k the images with sample_id % 5 == k are held out, an encoder is
trained on the rest, and linking is measured on the held-out images. After 5
folds every image has been held out once. A single 12-image split is too small
to trust; this uses all 60.

The settings tried are listed on the command line and ALL of them are
reported, so no result is picked out after the fact.

Usage:
    python -m src.correlate.cross_validate --model mobilenet_v2 \
        --labels class image --epochs 30 100
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.config import CFG, DEVICE
from src.correlate.dataset_loader import N_FOLDS, ExplanationMapDataset
from src.correlate.encoder import ExplanationEncoder
from src.correlate.link import embed, load_split, raw_embed, retrieval_metrics
from src.correlate.maps import METHODS
from src.correlate.train_contrastive import fit

OUT_DIR = CFG.outputs_dir / "correlate" / "analysis"
KEYS = ("top1_same_image", "top3_same_image", "top1_same_class",
        "chance_top1_same_image", "chance_top3_same_image",
        "chance_top1_same_class", "mean_sim_same_image", "mean_sim_other_image")


def pair_average(metrics: dict) -> dict:
    """Average the 6 ordered method pairs."""
    return {k: float(np.mean([v[k] for v in metrics.values()])) for k in KEYS}


def summarize(per_fold: list) -> dict:
    out = {}
    for k in KEYS:
        vals = [f[k] for f in per_fold]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "per_fold": vals}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--labels", nargs="+", default=["class", "image"],
                        choices=["class", "image"])
    parser.add_argument("--epochs", nargs="+", type=int, default=[30, 100])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(123)
    random_encoder = ExplanationEncoder().to(DEVICE).eval()

    result = {"model": args.model, "n_folds": N_FOLDS, "seed": args.seed,
              "configs": {}, "untrained_encoder": None, "raw_maps": None}
    baseline = {"random_init": [], "raw": []}

    for label in args.labels:
        for epochs in args.epochs:
            name = f"label={label},epochs={epochs}"
            per_fold, t0 = [], time.time()
            for fold in range(N_FOLDS):
                train_ds = ExplanationMapDataset(args.model, "train", fold, label)
                test_ds = ExplanationMapDataset(args.model, "test", fold, label)
                encoder, _ = fit(train_ds, test_ds, epochs=epochs, seed=args.seed,
                                 log=False)
                _, maps, classes, _ = load_split(args.model, "test", fold)
                emb = {m: embed(encoder, maps[m]) for m in METHODS}
                per_fold.append(pair_average(retrieval_metrics(emb, classes)))
                if not baseline["raw"] or len(baseline["raw"]) < N_FOLDS:
                    e_rand = {m: embed(random_encoder, maps[m]) for m in METHODS}
                    e_raw = {m: raw_embed(maps[m]) for m in METHODS}
                    baseline["random_init"].append(
                        pair_average(retrieval_metrics(e_rand, classes)))
                    baseline["raw"].append(
                        pair_average(retrieval_metrics(e_raw, classes)))
            result["configs"][name] = summarize(per_fold)
            c = result["configs"][name]
            print(f"[cv] {name:24s} top1 image {c['top1_same_image']['mean']:.3f} "
                  f"(chance {c['chance_top1_same_image']['mean']:.3f}) | top3 "
                  f"{c['top3_same_image']['mean']:.3f} | top1 class "
                  f"{c['top1_same_class']['mean']:.3f} "
                  f"(chance {c['chance_top1_same_class']['mean']:.3f})  "
                  f"[{time.time() - t0:.0f}s]")

    result["untrained_encoder"] = summarize(baseline["random_init"])
    result["raw_maps"] = summarize(baseline["raw"])
    for label, key in (("untrained encoder", "untrained_encoder"), ("raw maps", "raw_maps")):
        c = result[key]
        print(f"[cv] {label:24s} top1 image {c['top1_same_image']['mean']:.3f} | top3 "
              f"{c['top3_same_image']['mean']:.3f} | top1 class "
              f"{c['top1_same_class']['mean']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.model}_cross_validation.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[cv] wrote {path}")


if __name__ == "__main__":
    main()
