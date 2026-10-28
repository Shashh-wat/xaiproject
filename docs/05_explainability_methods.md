# 5. Explainability Methods

Three methods, each answering a different question about the same
prediction:

| Method | Question it answers | Granularity | Code |
|---|---|---|---|
| **Grad-CAM** | "Which spatial regions of the image drove this class score?" | Coarse heatmap (last-conv resolution, upsampled) | `src/explain/gradcam.py` |
| **SHAP** | "How much did each pixel/region contribute, additively, positively or negatively?" | Pixel/superpixel-level, signed attribution | `src/explain/shap_explain.py` |
| **LIME** | "If I perturb superpixels, which ones does the model's decision depend on?" | Superpixel-level, local surrogate model | `src/explain/lime_explain.py` |

Grad-CAM is the primary method (it's the CNN-native one and the fastest to
run at scale for the comparison in step 6); SHAP and LIME are the "any new
method" / comparison partners the rubric asks for.

## Grad-CAM (`src/explain/gradcam.py`)

Uses the `pytorch-grad-cam` library. Hooks the target conv layer
(`model.get_target_layer()`), computes gradients of the predicted class
score w.r.t. that layer's activations, weights the activation maps by their
gradients, ReLUs, and upsamples to input resolution.

```bash
python -m src.explain.gradcam --model mobilenet_v2 \
    --image data/raw/plantvillage/Tomato___Late_blight/some_image.JPG \
    --out outputs/explanations/gradcam_example.png
```

Outputs a side-by-side: original image | heatmap overlay | predicted class +
confidence.

## SHAP (`src/explain/shap_explain.py`)

Uses `shap.GradientExplainer` (works directly on the PyTorch model, faster
than `KernelExplainer` for images) with a small background sample (~50
training images) as the reference distribution. Outputs a signed
red/blue overlay — red = pushed toward the predicted class, blue = pushed
away.

```bash
python -m src.explain.shap_explain --model mobilenet_v2 \
    --image data/raw/plantvillage/Tomato___Late_blight/some_image.JPG
```

## LIME (`src/explain/lime_explain.py`)

Uses `lime.lime_image`. Segments the image into superpixels (via
`skimage.segmentation.quickshift`), perturbs them (mean-color-out), fits a
local linear surrogate, and returns the top-k superpixels that support the
predicted class.

```bash
python -m src.explain.lime_explain --model mobilenet_v2 \
    --image data/raw/plantvillage/Tomato___Late_blight/some_image.JPG
```

## What to report in the paper

- One figure per method (3 total) on the same example image, so the reader
  can visually compare granularity/agreement
- 2–3 sentences per method: what it shows for this specific example, and any
  disagreement between methods (e.g. Grad-CAM highlighting the whole leaf
  while SHAP isolates just the lesion edge)
