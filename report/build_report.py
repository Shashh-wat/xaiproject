"""Builds report/report.html and report/report.pdf (IEEE conference format).

Tables are generated from the files under outputs/. Numbers that appear in the
prose are checked against those files with chk(); the build stops if one is off.

Run from the project root:   python3 report/build_report.py
"""
import json, os, subprocess, sys
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# --------------------------------------------------------------- data
models = ("mobilenet_v2", "resnet50")
rep = {m: json.load(open(f"outputs/eval/{m}_classification_report.json")) for m in models}
hist = {m: json.load(open(f"outputs/checkpoints/{m}_history.json")) for m in models}
rob = {m: pd.read_csv(f"outputs/robustness/{m}_perturbation_summary.csv") for m in models}
iou = pd.read_csv("outputs/comparison/iou_raw.csv")
iou_cls = pd.read_csv("outputs/comparison/iou_by_class.csv")
cv = json.load(open("outputs/correlate/analysis/mobilenet_v2_cross_validation.json"))
summ = json.load(open("outputs/correlate/analysis/mobilenet_v2_summary.json"))
enc_hist = json.load(open("outputs/correlate/checkpoints/encoder_mobilenet_v2_history.json"))
meta = pd.read_csv("outputs/correlate/metadata.csv")
counts = json.load(open("report/class_counts.json"))
raw = summ["raw_all_images"]["pairs"]

def chk(name, value, expected, tol):
    if abs(value - expected) > tol:
        sys.exit(f"CHECK FAILED {name}: file says {value}, text says {expected}")

NICE = {
    "Pepper__bell___Bacterial_spot": "Bell pepper, bacterial spot", "Pepper__bell___healthy": "Bell pepper, healthy",
    "Potato___Early_blight": "Potato, early blight", "Potato___Late_blight": "Potato, late blight", "Potato___healthy": "Potato, healthy",
    "Tomato_Bacterial_spot": "Tomato, bacterial spot", "Tomato_Early_blight": "Tomato, early blight", "Tomato_Late_blight": "Tomato, late blight",
    "Tomato_Leaf_Mold": "Tomato, leaf mold", "Tomato_Septoria_leaf_spot": "Tomato, Septoria leaf spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato, two-spotted spider mite", "Tomato__Target_Spot": "Tomato, target spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato, yellow leaf curl virus", "Tomato__Tomato_mosaic_virus": "Tomato, mosaic virus",
    "Tomato_healthy": "Tomato, healthy",
}
assert set(NICE) == set(counts), "class list changed"

# ---- numbers used in the prose, checked against the files
acc = {m: rep[m]["accuracy"] for m in models}
f1 = {m: rep[m]["macro_f1"] for m in models}
n_test = int(rep["mobilenet_v2"]["macro avg"]["support"])
chk("n_images", sum(counts.values()), 20638, 0)
chk("n_test", n_test, 3094, 0)
chk("acc mobilenet", acc["mobilenet_v2"], 0.918, 5e-4); chk("acc resnet", acc["resnet50"], 0.889, 5e-4)
chk("f1 mobilenet", f1["mobilenet_v2"], 0.902, 5e-4); chk("f1 resnet", f1["resnet50"], 0.883, 5e-4)
ci = {m: 1.96 * (acc[m] * (1 - acc[m]) / n_test) ** 0.5 for m in models}
chk("ci mobilenet", ci["mobilenet_v2"], 0.0097, 5e-4); chk("ci resnet", ci["resnet50"], 0.0111, 5e-4)
chk("max/min class", max(counts.values()) / min(counts.values()), 21.1, 0.05)
v = hist["resnet50"]["val_acc"]; chk("resnet val e1", v[0], 0.808, 5e-4); chk("resnet val e3", v[2], 0.893, 5e-4)
v = hist["mobilenet_v2"]["val_acc"]; chk("mobilenet best val", max(v), 0.917, 5e-4); assert v.index(max(v)) == 1
sup = [rep["mobilenet_v2"][k]["support"] for k in NICE]
rho = {m: spearmanr(sup, [rep[m][k]["recall"] for k in NICE]) for m in models}
chk("rho mobilenet", rho["mobilenet_v2"].correlation, -0.19, 0.005); chk("rho resnet", rho["resnet50"].correlation, -0.20, 0.005)
chk("p mobilenet", rho["mobilenet_v2"].pvalue, 0.50, 0.01); chk("p resnet", rho["resnet50"].pvalue, 0.47, 0.01)
chk("recall early blight M", rep["mobilenet_v2"]["Tomato_Early_blight"]["recall"], 0.74, 0.005)
chk("recall early blight R", rep["resnet50"]["Tomato_Early_blight"]["recall"], 0.62, 0.005)
chk("precision potato healthy M", rep["mobilenet_v2"]["Potato___healthy"]["precision"], 0.61, 0.005)
chk("iou n", len(iou), 75, 0); chk("iou mean", iou.iou.mean(), 0.435, 5e-4); chk("iou median", iou.iou.median(), 0.452, 5e-4)
chk("iou sd", iou.iou.std(), 0.193, 5e-4); chk("iou min", iou.iou.min(), 0.015, 5e-4); chk("iou max", iou.iou.max(), 0.81, 5e-3)
g = iou.groupby("both_correct").iou.agg(["mean", "count"])
chk("iou both correct", g.loc[True, "mean"], 0.456, 5e-4); chk("n both correct", g.loc[True, "count"], 67, 0)
chk("iou one wrong", g.loc[False, "mean"], 0.259, 5e-4); chk("n one wrong", g.loc[False, "count"], 8, 0)
lo, hi = iou_cls.sort_values("mean").iloc[0], iou_cls.sort_values("mean").iloc[-1]
assert lo.true_class == "Tomato_Bacterial_spot" and hi.true_class == "Potato___Late_blight"
chk("iou lo class", lo["mean"], 0.152, 5e-4); chk("iou hi class", hi["mean"], 0.588, 5e-4)
def rb(m, kind, sev, col): return float(rob[m][(rob[m].perturbation == kind) & (rob[m].severity == sev)][col].iloc[0])
mild = [("blur", 0.2), ("noise", 0.05), ("occlusion", 0.1), ("brightness", 0.3), ("brightness", -0.3)]
mild_iou = [rb(m, k, s, "mean_iou") for m in models for k, s in mild]
chk("mild iou min", min(mild_iou), 0.44, 0.005); chk("mild iou max", max(mild_iou), 0.55, 0.005)
mild_flip = [rb(m, k, s, "flip_rate") for m in models for k, s in mild]
chk("mild flip max", max(mild_flip), 0.30, 1e-9); chk("mild flip min", min(mild_flip), 0.05, 1e-9)
chk("occ 0.3 M", rb("mobilenet_v2", "occlusion", 0.3, "flip_rate"), 0.25, 1e-9); chk("occ 0.3 R", rb("resnet50", "occlusion", 0.3, "flip_rate"), 0.60, 1e-9)
chk("bright -0.6 iou R", rb("resnet50", "brightness", -0.6, "mean_iou"), 0.067, 5e-4)
chk("bright -0.6 iou M", rb("mobilenet_v2", "brightness", -0.6, "mean_iou"), 0.243, 5e-4)
chk("bright -0.6 flip R", rb("resnet50", "brightness", -0.6, "flip_rate"), 0.90, 1e-9)
strong = [rb(m, k, s, "mean_iou") for m in models for k, s in (("blur", 1.0), ("noise", 0.3), ("occlusion", 0.4), ("brightness", 0.6), ("brightness", -0.6))]
chk("strong iou min", min(strong), 0.067, 5e-4); chk("strong iou max", max(strong), 0.243, 5e-4)
chk("occ 0.1 M", rb("mobilenet_v2", "occlusion", 0.1, "flip_rate"), 0.10, 1e-9); chk("occ 0.2 M", rb("mobilenet_v2", "occlusion", 0.2, "flip_rate"), 0.30, 1e-9)
chk("occ 0.1 R", rb("resnet50", "occlusion", 0.1, "flip_rate"), 0.30, 1e-9); chk("occ 0.2 R", rb("resnet50", "occlusion", 0.2, "flip_rate"), 0.35, 1e-9)
chk("top3 class300", cv["configs"]["label=class,epochs=300"]["top3_same_image"]["mean"], 0.286, 5e-4); chk("top3 raw", cv["raw_maps"]["top3_same_image"]["mean"], 0.406, 5e-4)
chk("untrained sd", cv["untrained_encoder"]["top1_same_image"]["std"], 0.030, 5e-4)
first = meta[meta.method == "gradcam"]
chk("n maps images", len(first), 60, 0); chk("pred correct", (first.pred_class == first.true_class).mean(), 0.933, 5e-4)
chk("pred classes", first.pred_class.nunique(), 13, 0)
cfg = cv["configs"]
for name, lo_, hi_ in (("label=class,epochs=30", 0.092, 0.043), ("label=class,epochs=100", 0.117, 0.030), ("label=class,epochs=300", 0.075, 0.014),
                       ("label=image,epochs=30", 0.094, 0.024), ("label=image,epochs=100", 0.100, 0.027), ("label=image,epochs=300", 0.097, 0.025)):
    chk(name + " mean", cfg[name]["top1_same_image"]["mean"], lo_, 5e-4); chk(name + " sd", cfg[name]["top1_same_image"]["std"], hi_, 5e-4)
chk("untrained top1", cv["untrained_encoder"]["top1_same_image"]["mean"], 0.092, 5e-4)
chk("raw top1", cv["raw_maps"]["top1_same_image"]["mean"], 0.192, 5e-4); chk("raw top1 sd", cv["raw_maps"]["top1_same_image"]["std"], 0.032, 5e-4)
chk("chance top1", cfg["label=class,epochs=30"]["chance_top1_same_image"]["mean"], 0.083, 5e-4)
chk("chance top3", cfg["label=class,epochs=30"]["chance_top3_same_image"]["mean"], 0.25, 5e-4)
chk("chance class", cfg["label=class,epochs=30"]["chance_top1_same_class"]["mean"], 0.147, 5e-4)
chk("gc-lime same", raw["gradcam->lime"]["mean_sim_same_image"], 0.231, 5e-4); chk("gc-lime other", raw["gradcam->lime"]["mean_sim_other_image"], 0.091, 5e-4)
chk("gc-shap same", raw["gradcam->shap"]["mean_sim_same_image"], 0.139, 5e-4); chk("gc-shap other", raw["gradcam->shap"]["mean_sim_other_image"], 0.108, 5e-4)
chk("shap-lime same", raw["shap->lime"]["mean_sim_same_image"], 0.039, 5e-4); chk("shap-lime other", raw["shap->lime"]["mean_sim_other_image"], 0.031, 5e-4)
chk("gc->lime top1", raw["gradcam->lime"]["top1_same_image"], 0.083, 5e-4); chk("chance raw top1", raw["gradcam->lime"]["chance_top1_same_image"], 0.017, 5e-4)
cls_avg = sum(v["top1_same_class"] for v in raw.values()) / 6; chk("raw class avg", cls_avg, 0.114, 5e-4)
chk("raw class chance", raw["gradcam->lime"]["chance_top1_same_class"], 0.087, 5e-4)
chk("enc train loss first", enc_hist["train_loss"][0], 4.72, 5e-3); chk("enc train loss last", enc_hist["train_loss"][-1], 4.02, 5e-3)
chk("enc held first", enc_hist["held_out_loss"][0], 3.56, 5e-3); chk("enc held last", enc_hist["held_out_loss"][-1], 3.64, 5e-3)
tr = summ["train"]["learned"]; chk("enc on train top1", sum(x["top1_same_image"] for x in tr.values()) / 6, 0.05, 6e-3)

# --------------------------------------------------------------- html helpers
def f(x, n=3): return f"{x:.{n}f}"
def mn(x): return f"{x:.2f}".replace('-', '\u2212')
def table(num, title, head, rows, wide=False, note=None, align=None):
    align = align or ["l"] + ["r"] * (len(head) - 1)
    th = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(head, align))
    body = ""
    for r in rows:
        cls = ""
        if isinstance(r, tuple): r, cls = r
        body += f'<tr class="{cls}">' + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, align)) + "</tr>"
    n = f'<div class="tnote">{note}</div>' if note else ""
    return (f'<div class="tbl{" wide" if wide else ""}"><div class="tcap">TABLE {num}<br>{title.upper()}</div>'
            f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>{n}</div>')
def table_split(num, title, head, rows, split, note=None, align=None):
    """Wide table drawn as two halves side by side (rows[:split] and rows[split:])."""
    align = align or ["l"] + ["r"] * (len(head) - 1)
    th = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(head, align))
    def one(rs):
        body = "".join("<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(r, align)) + "</tr>" for r in rs)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"
    n = f'<div class="tnote">{note}</div>' if note else ""
    return (f'<div class="tbl wide"><div class="tcap">TABLE {num}<br>{title.upper()}</div>'
            f'<div class="twin">{one(rows[:split])}{one(rows[split:])}</div>{n}</div>')
def figure(num, src, caption, wide=False, width="100%"):
    return (f'<div class="fig{" wide" if wide else ""}"><img src="{src}" style="width:{width}"><div class="fcap">Fig. {num}. {caption}</div></div>')
def pm(d, k): return f'{d[k]["mean"]:.3f} ± {d[k]["std"]:.3f}'

# ---- tables
t1_rows = [[NICE[k], f"{counts[k]:,}", f"{int(rep['mobilenet_v2'][k]['support'])}", f"{rep['mobilenet_v2'][k]['recall']:.2f}", f"{rep['resnet50'][k]['recall']:.2f}"] for k in NICE]
t1_rows.append((["All 15 classes", f"{sum(counts.values()):,}", f"{n_test:,}", f"{rep['mobilenet_v2']['macro avg']['recall']:.2f}", f"{rep['resnet50']['macro avg']['recall']:.2f}"], "total"))
T1 = table("I", "Images per class, test-set size, and per-class recall", ["Class", "Images", "Test n", "MobileNetV2", "ResNet50"], t1_rows,
           note="Recall on the test split. Last row: macro-average recall. Class names are shortened from the folder names.")
def mrow(m, label, tp, total):
    r = rep[m]["macro avg"]; h = hist[m]
    return [label, f"{tp:,} / {total:,}", f"{max(h['val_acc']):.3f}", f"{acc[m]:.3f} ± {ci[m]:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{f1[m]:.3f}"]
T2 = table("II", "Classification results on the test split (3,094 images)",
           ["Model", "Trainable / all parameters", "Best val. acc.", "Test acc. (95% CI)", "Macro precision", "Macro recall", "Macro F1"],
           [mrow("mobilenet_v2", "MobileNetV2", 19215, 2243087), mrow("resnet50", "ResNet50", 30735, 23538767)], wide=True,
           note="Three epochs, backbone frozen. The interval is the binomial 95% interval for accuracy on 3,094 test images.")
rows3 = []
for kind, sevs, label in (("blur", (0.2, 0.4, 0.6, 0.8, 1.0), "Blur"), ("noise", (0.05, 0.1, 0.2, 0.3), "Noise"),
                          ("occlusion", (0.1, 0.2, 0.3, 0.4), "Occlusion"), ("brightness", (-0.6, -0.3, 0.3, 0.6), "Brightness")):
    for s in sevs:
        rows3.append([label, f"{s:g}", f"{rb('mobilenet_v2', kind, s, 'flip_rate'):.2f}", f"{rb('resnet50', kind, s, 'flip_rate'):.2f}",
                      f"{rb('mobilenet_v2', kind, s, 'mean_iou'):.2f}", f"{rb('resnet50', kind, s, 'mean_iou'):.2f}"])
T3 = table_split("III", "Effect of image changes (20 test images per network)",
           ["Change", "Strength", "Pred. changed, MobileNetV2", "Pred. changed, ResNet50", "IoU, MobileNetV2", "IoU, ResNet50"], rows3, 9, align=["l", "r", "r", "r", "r", "r"],
           note="Strength 0 is the unchanged image (0 changed, IoU 1). IoU is between the Grad-CAM map of the unchanged and changed image.")
P = {"gradcam->shap": "Grad-CAM → SHAP", "shap->gradcam": "SHAP → Grad-CAM", "gradcam->lime": "Grad-CAM → LIME", "lime->gradcam": "LIME → Grad-CAM", "shap->lime": "SHAP → LIME", "lime->shap": "LIME → SHAP"}
rows4 = [[P[k], f"{raw[k]['mean_sim_same_image']:.3f}", f"{raw[k]['mean_sim_other_image']:.3f}", f"{raw[k]['top1_same_image']:.3f}", f"{raw[k]['top3_same_image']:.3f}", f"{raw[k]['top1_same_class']:.3f}"] for k in P]
r0 = raw["gradcam->lime"]
rows4.append((["Chance", "", "", f"{r0['chance_top1_same_image']:.3f}", f"{r0['chance_top3_same_image']:.3f}", f"{r0['chance_top1_same_class']:.3f}"], "total"))
T4 = table("IV", "Plain correlation of the maps, all 60 images, no training",
           ["Query → target", "Pearson, same image", "Pearson, other images", "Top-1 same image", "Top-3 same image", "Top-1 same class"], rows4,
           note="Pearson columns: mean over the 60 images, and over all pairs of different images. Top-k: the nearest target maps among 60.")
cfg_names = {"label=class,epochs=30": "Class label, 30 epochs", "label=class,epochs=100": "Class label, 100 epochs", "label=class,epochs=300": "Class label, 300 epochs",
             "label=image,epochs=30": "Image label, 30 epochs", "label=image,epochs=100": "Image label, 100 epochs", "label=image,epochs=300": "Image label, 300 epochs"}
rows5 = [[cfg_names[k], pm(cfg[k], "top1_same_image"), pm(cfg[k], "top3_same_image"), pm(cfg[k], "top1_same_class")] for k in cfg_names]
rows5.append(["Untrained encoder", pm(cv["untrained_encoder"], "top1_same_image"), pm(cv["untrained_encoder"], "top3_same_image"), pm(cv["untrained_encoder"], "top1_same_class")])
rows5.append((["Plain correlation", pm(cv["raw_maps"], "top1_same_image"), pm(cv["raw_maps"], "top3_same_image"), pm(cv["raw_maps"], "top1_same_class")], "bold"))
rows5.append((["Chance", "0.083", "0.250", "0.147"], "total"))
T5 = table("V", "Held-out matching of explanation maps, 5-fold cross-validation",
           ["Embedding", "Top-1 same image", "Top-3 same image", "Top-1 same class"], rows5, wide=True,
           note="Mean ± standard deviation over the 5 folds. Each fold holds out 12 images. Averaged over the six query → target pairs.")

# ---- pipeline figure (inline SVG, grayscale)
def box(x, y, w, h, t, lines=(), dark=False):
    fill = "#e8e8e8" if dark else "#ffffff"
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="#222" stroke-width="1"/>'
    s += f'<text x="{x + w / 2}" y="{y + 15}" text-anchor="middle" font-size="9.5" font-weight="bold">{t}</text>'
    for i, l in enumerate(lines): s += f'<text x="{x + w / 2}" y="{y + 29 + i * 11}" text-anchor="middle" font-size="8">{l}</text>'
    return s
def arr(x1, y1, x2, y2): return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#222" stroke-width="1" marker-end="url(#a)"/>'
svg = '<svg viewBox="0 0 696 150" width="100%" font-family="Times New Roman, Times, serif"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#222"/></marker></defs>'
svg += box(4, 45, 84, 60, "Leaf image", ["224 × 224", "test split"])
svg += box(112, 45, 112, 60, "CNN classifier", ["MobileNetV2 or ResNet50", "frozen backbone,", "trained linear layer"], dark=True)
svg += box(256, 8, 100, 38, "Grad-CAM", ["last conv block"]) + box(256, 56, 100, 38, "SHAP", ["GradientExplainer"]) + box(256, 104, 100, 38, "LIME", ["superpixels"])
svg += box(388, 45, 112, 60, "Evidence maps", ["64 × 64 per method,", "for the predicted class"], dark=True)
svg += box(532, 10, 160, 60, "Plain correlation", ["Pearson correlation", "of two maps"]) + box(532, 80, 160, 62, "Siamese encoder", ["shared weights, SupCon loss", "(did not beat correlation)"])
svg += arr(88, 75, 110, 75) + arr(224, 75, 254, 27) + arr(224, 75, 254, 75) + arr(224, 75, 254, 123)
svg += arr(356, 27, 386, 66) + arr(356, 75, 386, 75) + arr(356, 123, 386, 84) + arr(500, 68, 530, 40) + arr(500, 82, 530, 108)
svg += "</svg>"
FIG1_HTML = f'<div class="fig wide">{svg}<div class="fcap">Fig. 1. Pipeline. One image goes through a classifier. Three methods explain the predicted class. Each explanation becomes a 64 × 64 map, and the maps are compared.</div></div>'

# --------------------------------------------------------------- text
ABSTRACT = (
    "A crop disease classifier that returns only a label gives a grower no reason to trust it. We train two convolutional networks, "
    "MobileNetV2 and ResNet50, on a 15-class, 20,638-image copy of the PlantVillage dataset (pepper, potato and tomato leaves) and explain "
    "their predictions with Grad-CAM, SHAP and LIME. The networks reach 91.8% and 88.9% test accuracy. The Grad-CAM maps of the two networks "
    "overlap with an intersection over union (IoU) of 0.435 on 75 test images, and of 0.259 on the 8 images where at least one network is wrong. "
    "When images are blurred, noised, occluded or darkened, the Grad-CAM map changes at strengths where most predictions do not: the mean IoU "
    "with the original map is about 0.5 at the mildest strength of each change. We then ask whether the three kinds of explanation can be linked. "
    "Each is reduced to a 64 × 64 map of evidence for the predicted class. On 60 images, the correlation between the maps of one image is 0.23 for "
    "Grad-CAM and LIME (0.09 between different images), 0.14 for Grad-CAM and SHAP (0.11), and 0.04 for SHAP and LIME (0.03). A Siamese encoder "
    "trained with a supervised contrastive loss did not improve on this plain correlation on unseen images: top-1 same-image matching was 0.075 "
    "to 0.117 for the encoder, 0.192 for correlation, and 0.083 by chance (5-fold cross-validation). We report this as a negative result at this "
    "data size and list the likely causes."
)
KEYWORDS = "explainable AI, Grad-CAM, SHAP, LIME, plant disease classification, Siamese network, supervised contrastive learning"

TEXT = f"""
<h1>I. Introduction</h1>
<p class="first">Convolutional neural networks trained on curated image repositories [1] classify crop leaf diseases with high accuracy on held-out images from the same repository [2].
A label alone gives a grower or an agronomist little reason to trust a prediction. Explanation methods add a reason by marking where in the image the evidence lies.</p>
<p>Three widely used methods do this in three different forms. Grad-CAM [3] returns a coarse heat map taken from the last convolutional layer. SHAP [5] returns a value for each pixel.
LIME [4] returns a weight for each superpixel, a group of similar neighbouring pixels. Because the forms differ, it is not obvious how far the three agree on the same prediction,
and a reader who sees three pictures has to judge by eye. If the map of one method could be matched reliably to the maps of the others, a cheap method could stand in for an expensive one,
and disagreement could be flagged automatically.</p>
<p>This paper measures that agreement and tests one way of learning it. The contributions are:</p>
<ol class="contrib">
<li>Two baseline classifiers on a 15-class copy of PlantVillage, with overall and per-class test results (Section V).</li>
<li>The overlap of Grad-CAM maps between the two networks, their change under four kinds of image change, and a weight-randomization sanity check [6] (Sections VI-A to VI-C).</li>
<li>A common 64 × 64 form for Grad-CAM, SHAP and LIME maps, and a held-out test of how well the maps of one method can be matched to those of another,
using plain correlation and a Siamese encoder [7] trained with a supervised contrastive loss [8] (Sections VI-D and VI-E).</li>
<li>A web application that runs the whole pipeline on an uploaded leaf image.</li>
</ol>
<p>Fig. 1 shows the pipeline. The encoder did not beat plain correlation; we report that result in full.</p>

<h1>II. Related Work</h1>
<p class="first"><b>Plant disease images.</b> Hughes and Salathé [1] released an open-access repository of over 50,000 curated images of healthy and infected crop leaves.
Mohanty et al. [2] trained a convolutional network on 54,306 images covering 14 crop species and 26 diseases and reported 99.35% accuracy on a held-out test set,
but 31.4% on images from online sources. The gap shows that accuracy on the curated images does not carry to other photographs.</p>
<p><b>Explanation methods.</b> Grad-CAM [3] uses the gradients flowing into the last convolutional layer to make a coarse heat map of the regions that drove a prediction.
LIME [4] explains one prediction of any classifier by fitting a simple interpretable model to the classifier's behaviour around that prediction.
SHAP [5] scores how much each input feature contributed to a prediction, using additive importance measures with a unique solution that has desirable properties.</p>
<p><b>Checking explanations.</b> Adebayo et al. [6] propose tests for saliency methods and report that some are independent of both the trained model and the data.
One of their tests re-initializes the weights of a network from the top layer to the bottom and checks that the explanation changes; we use a version of it in Section IV-E.</p>
<p><b>Learned similarity.</b> A Siamese network [7] has two identical sub-networks joined at their outputs, so that two inputs are compared in a shared space.
Supervised contrastive learning [8] uses class labels so that embeddings of the same class are pulled together and those of different classes are pushed apart.
We use both ideas to compare explanation maps.</p>
<p>We do not propose a new explanation method. We measure how three existing ones relate on the same predictions and test whether a learned embedding helps.</p>

<h1>III. Data</h1>
<p class="first">We use the Kaggle copy of PlantVillage [9]. It has 20,638 images in 15 classes: 2 of bell pepper, 3 of potato and 10 of tomato (Table I). The 54,306-image set used in [2] covers 14 crop species and 26 diseases; this copy covers only three crops. The images show single leaves on plain backgrounds (Fig. 2). Class sizes range from 152 to 3,208 images, a ratio of 21.</p>
<p>The images were split by class into 70% training, 15% validation and 15% test with a fixed seed (42): 14,447, 3,097 and 3,094 images. We did not check for near-duplicate images across the splits.
If there are any, test accuracy is higher than it would be on new photographs. All images were resized to 224 × 224 and normalized with the ImageNet mean and standard deviation.
Training images were also flipped horizontally at random, rotated by up to 15 degrees, and given brightness and contrast jitter of 0.2.</p>
{T1}

<h1>IV. Methods</h1>
<h2>A. Classifiers</h2>
<p class="first">We use MobileNetV2 [10] and ResNet50 [11] with ImageNet [12] weights. The backbone weights are frozen and the final layer is replaced by a new linear layer with 15 outputs
(1280 inputs for MobileNetV2, preceded by dropout 0.2; 2048 for ResNet50). Only that layer is trained: 19,215 of 2,243,087 parameters for MobileNetV2 and 30,735 of 23,538,767 for ResNet50.
Batch-normalization statistics in the frozen backbone still update, because the model runs in training mode. The loss is cross-entropy with class weights
w<sub>c</sub> = N / (K n<sub>c</sub>), where N is the number of training images, K = 15 and n<sub>c</sub> the images in class c. We use Adam [13] with learning rate 10<sup>−3</sup>,
weight decay 10<sup>−4</sup>, batch size 32 and 3 epochs, and keep the epoch with the best validation accuracy.</p>
<h2>B. Explanation methods</h2>
<p class="first">All three methods explain the class the network predicted. <i>Grad-CAM</i> uses the pytorch-grad-cam library [14] on the last block (<tt>features[-1]</tt> in MobileNetV2, <tt>layer4[-1]</tt> in ResNet50),
whose output is a 7 × 7 grid. Because the backbone is frozen, no weight needs a gradient, so the input tensor is marked as requiring one; without this the gradient hook receives nothing.
<i>SHAP</i> uses GradientExplainer (shap 0.49.1) with 30 background images drawn at random from the image folder and 50 samples; the values are summed over the colour channels.
<i>LIME</i> segments the image with quickshift (kernel size 4, maximum distance 200, ratio 0.2), classifies 150 perturbed copies in which hidden superpixels are set to black, fits a linear model, and gives each pixel the weight of its superpixel.</p>
<h2>C. Common form of the maps</h2>
<p class="first">Grad-CAM is non-negative after its ReLU. For SHAP and LIME we keep only positive values, so all three maps show evidence for the predicted class and evidence against it is dropped.
Each map is scaled to [0, 1] by its own minimum and maximum and resized to 64 × 64. A map with no positive value becomes all zeros.</p>
<h2>D. Overlap and stability under image changes</h2>
<p class="first">The overlap of two heat maps is the intersection over union of their top 25% regions,</p>
<div class="eq"><span>IoU(a, b) = |T<sub>a</sub> ∩ T<sub>b</sub>| / |T<sub>a</sub> ∪ T<sub>b</sub>|,</span><span>(1)</span></div>
<p class="noind">where T is the set of pixels whose value is at or above the 75th percentile of the map. We compare the Grad-CAM maps of the two networks on 75 test images (5 per class, seed 42).</p>
<p>For stability we take 20 random test images (seed 42, the same for both networks) and apply four changes at several strengths <i>s</i>: Gaussian blur (kernel size about 15<i>s</i>, odd and at least 3, σ = 5<i>s</i>);
additive Gaussian noise with standard deviation <i>s</i> on pixel values in [0, 1]; a grey square covering the fraction <i>s</i> of the image at a random position; and a brightness shift of <i>s</i> added to pixel values in [0, 1].
Noise and the occlusion position are drawn once per image and strength. For each change we record the share of images whose predicted class differs from that of the unchanged image,
and the mean IoU between the Grad-CAM map of the unchanged and of the changed image, each for the class predicted on that image.</p>
<h2>E. Sanity check</h2>
<p class="first">Following the idea of Adebayo et al. [6], we re-initialize the layers that have parameters, starting at the output and moving to the input, in five steps of 20% of those layers each,
and compute Grad-CAM after each step. We report the Spearman rank correlation between the maps of consecutive steps. We ran this for one image and for MobileNetV2 only. The re-initialization was not seeded, so a repeat run gives different values.</p>
<h2>F. Linking the maps</h2>
<p class="first"><i>Plain correlation.</i> Two 64 × 64 maps a and b, flattened, are compared with the Pearson correlation</p>
<div class="eq"><span>ρ(a, b) = Σ<sub>k</sub>(a<sub>k</sub> − ā)(b<sub>k</sub> − b̄) / √(Σ<sub>k</sub>(a<sub>k</sub> − ā)² Σ<sub>k</sub>(b<sub>k</sub> − b̄)²).</span><span>(2)</span></div>
<p class="noind"><i>Siamese encoder.</i> One small network turns a 64 × 64 map into a vector of length 64 with unit norm. It has three blocks of 3 × 3 convolution, batch normalization and ReLU
(1→16, 16→32 and 32→64 channels), with max pooling after the first two and global average pooling after the third, followed by two linear layers (64→64 with ReLU, then 64→64).
It has 31,840 parameters. The same weights process the maps of all three methods, and the network is not told which method made a map.
It is trained with the supervised contrastive loss [8]. With unit-norm embeddings z, temperature τ = 0.1, a batch with anchors <i>i</i>, positives P(i) and all other maps A(i),</p>
<div class="eq small"><span>ℓ<sub>i</sub> = −(1/|P(i)|) Σ<sub>p∈P(i)</sub> log [ exp(z<sub>i</sub>·z<sub>p</sub>/τ) / Σ<sub>a∈A(i)</sub> exp(z<sub>i</sub>·z<sub>a</sub>/τ) ],</span><span>(3)</span></div>
<p class="noind">and the loss is the mean of ℓ<sub>i</sub> over anchors with at least one positive. We tried two labels. With the <i>class label</i>, maps of images with the same predicted class are positives, whichever method made them.
With the <i>image label</i>, only the three maps of one image are positives, which is the usual Siamese pairing. Training used Adam (10<sup>−3</sup>), batch size 64 and seed 0.</p>
<p><i>Evaluation.</i> We took 60 random test images (seed 42) and cached their three maps from MobileNetV2 (56 of the 60 were classified correctly; 13 classes were predicted).
The images were split into 5 folds by image, so the three maps of an image are never divided between training and test. For each fold an encoder was trained on 48 images and tested on the 12 held-out ones.
For every ordered pair of methods (query → target) and every held-out image, we ranked the target maps of the 12 images by similarity to the query map and recorded whether the nearest
one belongs to the same image (top-1), whether the same image is among the three nearest (top-3), and whether the nearest map has the same predicted class. Chance is computed from the same data:
1/12 for top-1, 3/12 for top-3, and the share of held-out images with the query's class for same-class. We report the mean over the six ordered pairs, then the mean and standard deviation over the 5 folds.
Two baselines use the same procedure: an untrained encoder with random weights, and plain correlation, Eq. (2). Plain correlation needs no training, so we also measured it on all 60 images at once (chance 1/60).
The six encoder settings (two labels, 30, 100 and 300 epochs) were all run and all are reported.</p>
<h2>G. Implementation</h2>
<p class="first">Everything ran on one laptop (Apple M2, 8 GB memory). Classifier training and the Grad-CAM runs used the Apple GPU through PyTorch 2.12.0 and torchvision 0.27.0; SHAP, LIME and the encoder runs used the CPU.
Caching the maps of 60 images took about 18 minutes. A Python package and a FastAPI web application serve the same pipeline for one uploaded image in about 17 seconds on the CPU.
The source code and the numbers behind every table are at github.com/Shashh-wat/xaiproject.</p>

<h1>V. Results: Classification</h1>
<p class="first">MobileNetV2 reached {acc['mobilenet_v2']:.3f} test accuracy and ResNet50 {acc['resnet50']:.3f} (Table II). The 95% intervals, ±{ci['mobilenet_v2'] * 100:.1f} and ±{ci['resnet50'] * 100:.1f} points, do not overlap.
ResNet50 was still improving when training stopped: its validation accuracy was 0.808, 0.869 and 0.893 over the three epochs, while MobileNetV2 peaked at 0.917 in epoch 2.
The comparison therefore says little about the architectures themselves; it compares two short runs of a trained last layer.</p>
{T2}
<p>Recall is high for most classes and lowest for tomato early blight (0.74 for MobileNetV2, 0.62 for ResNet50; Table I). MobileNetV2 has precision of only 0.61 for healthy potato, the smallest class (23 test images),
although its recall there is 1.00. Class size and recall are not clearly related: the Spearman correlation between test-set class size and recall over the 15 classes is {mn(rho['mobilenet_v2'].correlation)} for MobileNetV2
(p = {rho['mobilenet_v2'].pvalue:.2f}) and {mn(rho['resnet50'].correlation)} for ResNet50 (p = {rho['resnet50'].pvalue:.2f}). With 15 classes this cannot rule out a weak relation.</p>

<h1>VI. Results: Explanations</h1>
<h2>A. Grad-CAM across the two networks</h2>
<p class="first">Over the 75 test images the mean IoU between the Grad-CAM maps of the two networks is 0.435 (median 0.452, standard deviation 0.193, range 0.015 to 0.81). Fig. 2 shows one leaf where both networks
look at the lesion and the overlap is 0.67. On the 67 images that both networks classify correctly the mean is 0.456; on the 8 images where at least one is wrong it is 0.259. Eight images are too few to
treat the second figure as a result. By class, the lowest mean is for tomato bacterial spot (0.152) and the highest for potato late blight (0.588), with 5 images per class.
The two networks agree on part of the evidence, not on all of it.</p>
{figure(2, "figures/gradcam_example.png", "Grad-CAM of both networks on one test leaf.")}
<h2>B. Stability under image changes</h2>
<p class="first">Table III and Fig. 3 give all values. Blur and noise raise the share of changed predictions steadily with strength. At the mildest non-zero strength of each change, 5% to 30% of predictions change,
yet the mean IoU of the Grad-CAM map with the original is already between 0.44 and 0.55; at the strongest strengths it is between 0.07 and 0.24. So a mild change moves the heat map in most images
even when the predicted class stays the same. The two networks behave alike with two exceptions. ResNet50 changes its prediction more often under occlusion (0.30, 0.35 and 0.60 at 0.1, 0.2 and 0.3 of the area, against 0.10, 0.30 and 0.25),
and at the darkest setting its Grad-CAM map has almost no overlap with the original (IoU 0.07, against 0.24 for MobileNetV2), with 0.90 of its predictions changed. With 20 images, each share moves in steps of 0.05 and differences below about 0.2 should not be read as real.
IoU of top-25% regions of a coarse map is sensitive to small shifts, and we have no reference for how much shift a user would accept.</p>
{figure(3, "figures/robustness.png", "Share of predictions that change, and Grad-CAM IoU with the unchanged image, against the strength of each image change (20 test images per network).", wide=True)}
{T3}
<h2>C. Sanity check</h2>
<p class="first">Fig. 4 shows the maps for one image. The rank correlation between the maps of consecutive steps is 0.014, 0.366, 0.340, −0.293 and 0.040. The map already differs from the trained one
after the output-side 20% of the layers are re-initialized, which is what the test expects if the map depends on the learned weights. The values are not monotone and come from one image and one network,
so the check does not show more than that.</p>
{figure(4, "../docs/figures/sanity_mobilenet_v2.png", "Grad-CAM for one image as more layers are re-initialized (output side first): trained network, then 20% to 100%.", wide=True)}
<h2>D. Do the three methods agree?</h2>
<p class="first">Table IV gives plain correlation of the 64 × 64 maps over the 60 images. The maps of one image are correlated more than maps of different images for every pair, by very different amounts.
Grad-CAM and LIME are clearly linked: 0.231 for the same image against 0.091 for different images, and the nearest LIME map to a Grad-CAM map belongs to the same image 8.3% of the time against 1.7% by chance.
Grad-CAM and SHAP are linked weakly (0.139 against 0.108). SHAP and LIME hardly at all (0.039 against 0.031). Even for the best pair, the right image is first in 8 of 100 cases.
The nearest map has the same predicted class in {cls_avg * 100:.1f}% of cases on average, against 8.7% by chance, so the maps identify the image more than the disease.</p>
{T4}
<h2>E. A learned link</h2>
<p class="first">Table V gives the cross-validated results. None of the six encoder settings is clearly better than the untrained encoder. Top-1 same-image matching is 0.075 to 0.117 for the trained encoders against 0.092 ± 0.030 untrained,
and the differences are smaller than the fold-to-fold spread. All six are below plain correlation (0.192 ± 0.032), and the best encoder result in top-3 (0.286) is below correlation (0.406). On the training images themselves the encoder
reaches a top-1 same-image rate of only 0.05, so it is under-fitted and not over-fitted: the training loss falls from 4.72 to 4.02 in 100 epochs and the held-out loss moves from 3.56 to 3.64.</p>
{T5}
<p>Fig. 5 shows why. The embeddings of the maps group by method, not by class and not by image. The network learned to tell Grad-CAM, SHAP and LIME maps apart,
which is the opposite of the intended use. On one demonstration image the cosine similarities of the encoder's embeddings were between 0.83 and 0.94 for all three pairs, while the correlations of the same maps were between 0.13 and 0.45.</p>
{figure(5, "../docs/figures/correlation_tsne_mobilenet_v2.png", "t-SNE of the embeddings of all 180 maps from the encoder trained with the class label (fold 4). Left: coloured by method. Right: coloured by predicted class.", wide=True)}

<h1>VII. Discussion</h1>
<p class="first"><b>What the results support.</b> Both networks give usable baselines. Their Grad-CAM maps overlap only partly (0.435) and move under mild image changes (about 0.5), so a single map should not be read as a fixed reason for a prediction.
Among the three explanation methods, Grad-CAM and LIME point at similar places on the same image, Grad-CAM and SHAP slightly, SHAP and LIME almost not at all. A likely reason, which we did not test, is scale:
Grad-CAM comes from a 7 × 7 grid and LIME from superpixels, so both are coarse, while the SHAP maps vary from pixel to pixel.</p>
<p><b>What they do not support.</b> They do not show that a learned encoder links the methods. At this data size it did not beat plain correlation, and it learned to separate the methods instead.
Likely causes, none of them tested, are the small training set (48 images per fold), the noise in the SHAP maps (50 samples and 30 background images), and the fact that telling methods apart is easier than finding what they share.
A fair test needs a few hundred images, a less noisy SHAP, and the same procedure for ResNet50.</p>
<p><b>Threats to validity.</b> The samples are small: 60 images for linking, 20 for stability, 75 for overlap and one for the sanity check, with one split and one seed for each classifier.
SHAP is random: on one image the Grad-CAM–SHAP correlation was 0.17 in one run and 0.19 in another. The classifiers are three-epoch baselines with a frozen backbone.
The photographs are on plain backgrounds, and [2] reports a large fall in accuracy on other images; we did not test field photographs. Keeping only positive evidence at 64 × 64 discards evidence against the class and fine detail.
An earlier run used SHAP maps that were wrong: with the installed shap version the values come in a different array layout, and our first code read them as horizontal stripes.
We found this because the encoder was at chance and the maps looked wrong. The code now handles both layouts and checks the shape. All results in this paper use the corrected maps.</p>

<h1>VIII. Conclusion</h1>
<p class="first">We trained two CNN baselines on a 15-class PlantVillage copy (test accuracy 0.918 and 0.889), explained them with Grad-CAM, SHAP and LIME, and measured how the explanations relate.
The Grad-CAM maps of the two networks overlap by 0.435 and change visibly under mild image changes. Grad-CAM and LIME maps of the same prediction are correlated (0.23 against 0.09 for different images),
Grad-CAM and SHAP weakly, SHAP and LIME barely. A Siamese encoder with a supervised contrastive loss did not improve on plain correlation on unseen images. Next steps are more images, a less noisy SHAP,
ResNet50 maps, and photographs taken outside the plain-background setting.</p>
"""

REFS = [
    'D. P. Hughes and M. Salathé, “An open access repository of images on plant health to enable the development of mobile disease diagnostics,” arXiv:1511.08060, 2015.',
    'S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based plant disease detection,” arXiv:1604.03169, 2016.',
    'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in <i>Proc. IEEE Int. Conf. Comput. Vis. (ICCV)</i>, 2017; arXiv:1610.02391.',
    'M. T. Ribeiro, S. Singh, and C. Guestrin, “‘Why should I trust you?’: Explaining the predictions of any classifier,” arXiv:1602.04938, 2016.',
    'S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in <i>Adv. Neural Inf. Process. Syst. (NIPS)</i>, 2017; arXiv:1705.07874.',
    'J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” arXiv:1810.03292, 2018.',
    'J. Bromley, I. Guyon, Y. LeCun, E. Säckinger, and R. Shah, “Signature verification using a ‘Siamese’ time delay neural network,” in <i>Adv. Neural Inf. Process. Syst. (NIPS)</i>, vol. 6, 1993, pp. 737–744.',
    'P. Khosla <i>et al.</i>, “Supervised contrastive learning,” arXiv:2004.11362, 2020.',
    'emmarex, “PlantVillage dataset,” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/emmarex/plantdisease (accessed Sep. 2026).',
    'M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2018, pp. 4510–4520.',
    'K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” arXiv:1512.03385, 2015.',
    'J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2009, pp. 248–255.',
    'D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2015.',
    'J. Gildenblat and contributors, “PyTorch library for CAM methods,” GitHub, 2021. [Online]. Available: https://github.com/jacobgil/pytorch-grad-cam',
]
refs_html = "".join(f'<div class="ref"><span class="rn">[{i}]</span> {r}</div>' for i, r in enumerate(REFS, 1))

CSS = """
@page { size: 8.5in 11in; margin: 0.75in 0.625in 1in 0.625in; }
* { box-sizing: border-box; }
html { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.2; color: #000; }
body { margin: 0; }
.title { column-span: all; hyphens: none; text-align: center; font-size: 22pt; line-height: 1.12; margin: 0 0 10pt; font-weight: normal; }
.byline { column-span: all; text-align: center; font-size: 11pt; margin-bottom: 3pt; }
.byline2 { column-span: all; text-align: center; font-size: 9pt; margin-bottom: 12pt; font-style: italic; }
.cols { column-count: 2; column-gap: 0.25in; text-align: justify; hyphens: auto; }
.abs { font-weight: bold; font-size: 9pt; line-height: 1.17; text-align: justify; margin-bottom: 4pt; }
.abs i { font-weight: bold; }
.kw { font-weight: bold; font-size: 9pt; margin-bottom: 6pt; text-align: justify; }
h1 { font-size: 10pt; font-weight: normal; text-transform: uppercase; text-align: center; margin: 9pt 0 4pt; break-after: avoid; break-after: avoid-column; }
h2 { font-size: 10pt; font-style: italic; font-weight: normal; text-align: left; margin: 6pt 0 2pt; break-after: avoid; }
p { margin: 0; text-indent: 0.17in; }
p.first, p.noind { text-indent: 0; }
ol.contrib { margin: 2pt 0 2pt 0; padding-left: 0.22in; }
ol.contrib li { margin: 0; padding-left: 0.02in; }
tt { font-family: "Courier New", monospace; font-size: 8.5pt; }
.eq { display: flex; justify-content: space-between; align-items: baseline; margin: 4pt 0; text-align: left; font-style: italic; break-inside: avoid; }
.eq span:first-child { flex: 1; text-align: center; }
.eq.small { font-size: 8.5pt; }
.fig, .tbl { break-inside: avoid; margin: 6pt 0; text-align: center; }
.fig.wide, .tbl.wide { column-span: all; margin: 8pt 0; }
.fig img { display: block; margin: 0 auto; }
.fcap { font-size: 8pt; text-align: justify; margin-top: 3pt; }
.tcap { font-size: 8pt; text-align: center; margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.15; }
thead th { border-top: 1px solid #000; border-bottom: 1px solid #000; padding: 2pt 3pt; font-weight: bold; vertical-align: bottom; }
tbody td { padding: 1pt 3pt; vertical-align: top; }
tbody tr:last-child td { border-bottom: 1px solid #000; }
tr.total td { border-top: 0.5px solid #000; }
tr.bold td { font-weight: bold; }
td.l, th.l { text-align: left; } td.l { white-space: nowrap; } td.r, th.r { text-align: right; }
.twin { display: flex; gap: 0.25in; align-items: flex-start; }
.twin table { width: calc(50% - 0.125in); }
td.r { white-space: nowrap; }
.tnote { font-size: 7.5pt; text-align: justify; margin-top: 2pt; }
.refs { font-size: 8pt; text-align: justify; hyphens: auto; }
.ref { padding-left: 20pt; text-indent: -20pt; margin-bottom: 1.5pt; }
.rn { display: inline-block; width: 17pt; text-indent: 0; }
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Explainable Crop Disease Detection</title><style>{CSS}</style></head><body>
<div class="cols">
<div class="title">Explainable Crop Disease Detection with CNNs: <span style="white-space:nowrap">Grad-CAM</span>, SHAP and LIME, and a Test of Whether Their Explanations Can Be Linked</div>
<div class="byline" style="font-size:12pt">Shashwat Singh (23bce1517)</div>
<div class="byline" style="font-size:10pt;margin-bottom:0">BCSE418L Explainable Artificial Intelligence, Digital Assignment</div>
<div class="byline2">Source code: github.com/Shashh-wat/xaiproject</div>
{FIG1_HTML}
<div class="abs"><i>Abstract</i>—{ABSTRACT}</div>
<div class="kw"><i>Index Terms</i>—{KEYWORDS}.</div>
{TEXT}
<h1 style="margin-top:10pt">References</h1>
<div class="refs">{refs_html}</div>
</div></body></html>"""
open("report/report.html", "w").write(html)
print("html written")

out = os.path.abspath("report/report.pdf")
r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                    "file://" + os.path.abspath("report/report.html")], capture_output=True, text=True, timeout=120)
print("pdf written" if os.path.exists(out) else r.stderr[-400:])
