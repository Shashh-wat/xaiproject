"""Fig. 3 of the report, drawn from outputs/robustness/<model>_perturbation_summary.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8})
d = {m: pd.read_csv(f"outputs/robustness/{m}_perturbation_summary.csv") for m in ("mobilenet_v2", "resnet50")}
kinds = ["blur", "noise", "occlusion", "brightness"]
titles = {"blur": "Gaussian blur", "noise": "Gaussian noise", "occlusion": "Occlusion (area fraction)", "brightness": "Brightness shift"}
fig, ax = plt.subplots(2, 4, figsize=(7.2, 2.55), sharey="row")
style = {"mobilenet_v2": dict(color="black", marker="o", ls="-", label="MobileNetV2"), "resnet50": dict(color="0.45", marker="s", ls="--", label="ResNet50")}
for j, k in enumerate(kinds):
    for i, (col, yl) in enumerate((("flip_rate", "Predictions changed"), ("mean_iou", "Grad-CAM IoU"))):
        a = ax[i, j]
        for m, st in style.items():
            r = d[m][d[m].perturbation == k].sort_values("severity")
            a.plot(r.severity, r[col], ms=3.2, lw=1.1, **st)
        a.set_ylim(0, 1.02); a.grid(alpha=.3, lw=.5)
        if j == 0: a.set_ylabel(yl)
        if i == 0: a.set_title(titles[k], fontsize=8)
        if i == 1: a.set_xlabel("strength")
ax[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig("report/figures/robustness.png", dpi=300)
