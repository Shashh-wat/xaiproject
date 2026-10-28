"""Analyzes the trained explanation-correlation encoder.

For the held-out images (never seen in training) and, for reference, the
training images, it measures how well a map from one method can be linked to
the maps of another method for the same image. Three embeddings are compared:

  learned      the trained encoder
  random_init  the same encoder with random weights, untrained
  raw          no encoder; Pearson correlation of the raw 64x64 maps

Writes to outputs/correlate/analysis/:
  <model>_summary.json                all numbers
  <model>_per_image_correlation.csv   per image, per method pair
  <model>_correlation_matrix.png      mean same-image similarity (held-out)
  <model>_tsne.png                    all maps, by method and by class

Usage:
    python -m src.correlate.analyze --model mobilenet_v2
"""
from __future__ import annotations

import argparse
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from src.config import CFG, DEVICE
from src.correlate.encoder import ExplanationEncoder
from src.correlate.link import embed, load_split, raw_embed, retrieval_metrics
from src.correlate.maps import METHODS
from src.correlate.train_contrastive import CKPT_DIR

OUT_DIR = CFG.outputs_dir / "correlate" / "analysis"


def load_encoder(model_name: str):
    ckpt_path = CKPT_DIR / f"encoder_{model_name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained encoder at {ckpt_path}. Run "
            f"`python -m src.correlate.train_contrastive --model {model_name}` "
            "first."
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    encoder = ExplanationEncoder(embed_dim=ckpt["embed_dim"]).to(DEVICE)
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return encoder


def embeddings_for(kind: str, maps: dict, encoder, random_encoder):
    if kind == "learned":
        return {m: embed(encoder, maps[m]) for m in METHODS}
    if kind == "random_init":
        return {m: embed(random_encoder, maps[m]) for m in METHODS}
    return {m: raw_embed(maps[m]) for m in METHODS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         choices=["mobilenet_v2", "resnet50"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(args.model)
    torch.manual_seed(123)
    random_encoder = ExplanationEncoder(
        embed_dim=encoder.projection[-1].out_features).to(DEVICE).eval()

    summary = {"model": args.model}
    per_image_rows = []
    split_data = {}
    for split in ("test", "train"):
        ids, maps, classes, names = load_split(args.model, split)
        split_data[split] = (ids, maps, classes, names)
        summary[f"n_images_{split}"] = len(ids)
        summary[split] = {}
        for kind in ("learned", "random_init", "raw"):
            emb = embeddings_for(kind, maps, encoder, random_encoder)
            summary[split][kind] = retrieval_metrics(emb, classes)
            if kind == "learned":
                for i, sid in enumerate(ids):
                    for m1, m2 in itertools.combinations(METHODS, 2):
                        per_image_rows.append({
                            "split": split, "sample_id": sid,
                            "image": names[i], "predicted_class": classes[i],
                            "pair": f"{m1}-{m2}",
                            "cosine_similarity": float(emb[m1][i] @ emb[m2][i]),
                        })

    # Raw-map correlation needs no training, so it can be measured on every
    # cached image at once (a larger pool than the held-out split).
    _, all_maps, all_classes, _ = load_split(args.model, None)
    summary["raw_all_images"] = {
        "n_images": len(all_classes),
        "pairs": retrieval_metrics({m: raw_embed(all_maps[m]) for m in METHODS},
                                   all_classes),
    }

    pd.DataFrame(per_image_rows).to_csv(
        OUT_DIR / f"{args.model}_per_image_correlation.csv", index=False)
    with open(OUT_DIR / f"{args.model}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- print a short table ----
    print(f"[analyze] {args.model}: {summary['n_images_train']} train images, "
          f"{summary['n_images_test']} held-out images")
    for split in ("test", "train"):
        print(f"[analyze] --- {split} ---")
        for kind in ("learned", "random_init", "raw"):
            r = summary[split][kind]
            t1 = np.mean([v["top1_same_image"] for v in r.values()])
            t3 = np.mean([v["top3_same_image"] for v in r.values()])
            c1 = np.mean([v["top1_same_class"] for v in r.values()])
            ch1 = np.mean([v["chance_top1_same_image"] for v in r.values()])
            chc = np.mean([v["chance_top1_same_class"] for v in r.values()])
            print(f"[analyze] {kind:12s} top1 same image {t1:.2f} "
                  f"(chance {ch1:.2f}) | top3 {t3:.2f} | top1 same class "
                  f"{c1:.2f} (chance {chc:.2f})")

    r = summary["raw_all_images"]
    print(f"[analyze] raw map correlation, all {r['n_images']} images, per pair "
          f"(same image / other image):")
    for k, v in r["pairs"].items():
        print(f"[analyze]   {k:14s} {v['mean_sim_same_image']:.2f} / "
              f"{v['mean_sim_other_image']:.2f}   top1 {v['top1_same_image']:.2f} "
              f"(chance {v['chance_top1_same_image']:.3f})")

    # ---- 3x3 matrix: mean same-image similarity, held-out, learned ----
    ids, maps, classes, names = split_data["test"]
    emb = embeddings_for("learned", maps, encoder, random_encoder)
    mat = np.eye(3)
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            if i != j:
                mat[i, j] = float(np.mean(np.sum(emb[a] * emb[b], axis=1)))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(METHODS)
    ax.set_yticks(range(3)); ax.set_yticklabels(METHODS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    ax.set_title(f"{args.model}: mean similarity, same image\n"
                 f"(held-out images, learned embedding)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{args.model}_correlation_matrix.png", dpi=150)
    plt.close(fig)

    # ---- t-SNE of every map ----
    all_emb, all_method, all_class, all_split = [], [], [], []
    for split in ("train", "test"):
        ids, maps, classes, names = split_data[split]
        e = embeddings_for("learned", maps, encoder, random_encoder)
        for m in METHODS:
            all_emb.append(e[m])
            all_method += [m] * len(ids)
            all_class += list(classes)
            all_split += [split] * len(ids)
    all_emb = np.concatenate(all_emb)
    all_method, all_class, all_split = map(np.array, (all_method, all_class, all_split))
    perplexity = max(5, min(30, len(all_emb) // 4))
    proj = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=42).fit_transform(all_emb)
    class_ids = {c: i for i, c in enumerate(sorted(set(all_class)))}
    class_num = np.array([class_ids[c] for c in all_class])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for m in METHODS:
        mask = all_method == m
        axes[0].scatter(proj[mask, 0], proj[mask, 1], label=m, s=16, alpha=0.7)
    axes[0].set_title("Colored by explanation method")
    axes[0].legend(fontsize=8)
    axes[1].scatter(proj[:, 0], proj[:, 1], c=class_num, cmap="tab20", s=16, alpha=0.8)
    axes[1].set_title("Colored by predicted class")
    fig.suptitle(f"{args.model}: t-SNE of explanation-map embeddings")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{args.model}_tsne.png", dpi=150)
    plt.close(fig)

    print(f"[analyze] wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
