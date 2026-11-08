"""Fig. 2 of the report: Grad-CAM of both networks on one test leaf.
Uses the Potato late blight test image whose overlap in outputs/comparison/iou_raw.csv
is closest to 0.67 (the value shown in the figure title)."""
import os, sys
os.environ["FORCE_CPU"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.explain.common import load_model, load_image_tensor
from src.explain.gradcam import compute_gradcam
from src.compare_explanations import heatmap_iou

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7})
df = pd.read_csv("outputs/comparison/iou_raw.csv")
sub = df[(df.true_class == "Potato___Late_blight") & df.both_correct].reset_index(drop=True)
row = sub.iloc[int((sub.iou - 0.67).abs().argmin())]
a, names = load_model("mobilenet_v2"); b, _ = load_model("resnet50")
x, rgb = load_image_tensor(row.path)
ha, pa = compute_gradcam(a, x); hb, pb = compute_gradcam(b, x)
iou = heatmap_iou(ha, hb, 0.25)
print("image", os.path.basename(row.path), "iou", round(iou, 3), "pred", names[pa], names[pb])
fig, ax = plt.subplots(1, 3, figsize=(3.45, 1.35))
for k, (im, t) in enumerate([(rgb, "Leaf (Potato late blight)"), (show_cam_on_image(rgb, ha, use_rgb=True), "MobileNetV2"), (show_cam_on_image(rgb, hb, use_rgb=True), "ResNet50")]):
    ax[k].imshow(im); ax[k].set_title(t, fontsize=6.5, pad=2); ax[k].axis("off")
fig.suptitle(f"IoU of the top 25% regions: {iou:.2f}", fontsize=6.5, y=0.02, va="bottom")
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.03)
fig.savefig("report/figures/gradcam_example.png", dpi=400)
