"""Generates a small synthetic 'leaf disease' dataset so the pipeline can be
exercised end-to-end while the real PlantVillage download is in progress.

NOT biologically meaningful -- just colored blobs on a green background,
enough to give the CNNs something learnable (color + blob-position signal
per class) so training/explainability/robustness/correlation code can all
be verified to actually run and produce sane-shaped outputs.
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("data/raw/plantvillage")
CLASSES = {
    "Tomato___healthy": (60, 140, 60),      # uniform green, no lesion
    "Tomato___Late_blight": (90, 70, 40),   # brown lesion blobs
    "Tomato___Early_blight": (150, 120, 30),# yellow-brown lesion blobs
    "Apple___healthy": (50, 120, 50),
    "Apple___Black_rot": (40, 30, 30),      # dark lesion blobs
    "Corn___Common_rust": (170, 90, 40),    # rust-colored streaks
}
N_PER_CLASS = 60
IMG_SIZE = 128
random.seed(42)
np.random.seed(42)


def make_leaf(base_color, lesion_color, has_lesion, n_lesions):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (20, 60, 20))
    draw = ImageDraw.Draw(img)
    # leaf blob (ellipse) with per-pixel-ish color jitter via noise overlay
    margin = 8
    draw.ellipse([margin, margin, IMG_SIZE - margin, IMG_SIZE - margin],
                 fill=base_color)
    if has_lesion:
        for _ in range(n_lesions):
            cx = random.randint(30, IMG_SIZE - 30)
            cy = random.randint(30, IMG_SIZE - 30)
            r = random.randint(6, 16)
            jitter = tuple(min(255, max(0, c + random.randint(-15, 15)))
                            for c in lesion_color)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=jitter)
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-10, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    import sys
    if OUT.exists() and any(OUT.iterdir()) and "--force" not in sys.argv:
        raise SystemExit(
            f"{OUT} already has files (maybe the real dataset). This script "
            "writes fake images there. Move the real data away first, or "
            "pass --force.")
    OUT.mkdir(parents=True, exist_ok=True)
    for cname, color in CLASSES.items():
        cdir = OUT / cname
        cdir.mkdir(parents=True, exist_ok=True)
        healthy = "healthy" in cname
        for i in range(N_PER_CLASS):
            img = make_leaf(
                base_color=color if healthy else (50, 110, 50),
                lesion_color=color,
                has_lesion=not healthy,
                n_lesions=random.randint(2, 5),
            )
            img.save(cdir / f"{cname}_{i:03d}.jpg", quality=90)
        print(f"[synthetic] wrote {N_PER_CLASS} images -> {cdir}")


if __name__ == "__main__":
    main()
