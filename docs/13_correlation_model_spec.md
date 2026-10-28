# 13. Correlation model: technical specification

Code: `src/correlate/`. Idea and results: [`12_linking_xai_methods.md`](12_linking_xai_methods.md).
Run commands: [`11_explanation_correlation.md`](11_explanation_correlation.md).

## Purpose

Grad-CAM, SHAP and LIME each return a map for one prediction, in a different
form. The model was meant to put the three kinds of map into one vector
space, so a map from one method can be compared with, and matched to, maps
from the other two. It was built and tested. It did not beat plain
correlation of the maps (see [`12_linking_xai_methods.md`](12_linking_xai_methods.md)),
so the frontend uses plain correlation. This document describes both.

## Pipeline

```
image ──► classifier (MobileNetV2 or ResNet50) ──► predicted class c
   │
   ├─► Grad-CAM ─┐
   ├─► SHAP ─────┼─► 3 maps, each 64×64 in [0,1], evidence for class c
   └─► LIME ─────┘
                      │
                      ▼
           shared encoder (same weights for all 3)
                      │
                      ▼
          3 unit vectors of length 64 ──► cosine similarity, matching
```

One encoder is trained per classifier. Maps from MobileNetV2 and ResNet50
are never mixed.

## 1. Map extraction (`maps.py`)

All three maps are for the class the classifier predicted, not the true
class. All three keep only evidence for that class.

| Method | Settings | Raw output | Reduction to one map |
|---|---|---|---|
| Grad-CAM | `pytorch-grad-cam` `GradCAM`; target layer `features[-1]` (MobileNetV2) or `layer4[-1]` (ResNet50); target = predicted class | H×W, non-negative | min-max to [0,1] |
| SHAP | `shap.GradientExplainer`; 30 background images drawn at random from the whole image folder (seed 42); `nsamples=50`; `ranked_outputs=1` | 3×H×W signed | sum over channels, keep values above 0, min-max to [0,1] |
| LIME | `lime_image`; superpixels from `quickshift(kernel_size=4, max_dist=200, ratio=0.2)`; `top_labels=1`; `hide_color=0`; `num_samples=150` for the cached maps and the API | weight per superpixel | give each pixel its superpixel's weight, keep values above 0, min-max to [0,1] |

Then each map is resized to 64×64 (`skimage.transform.resize`, anti-aliased).
A map with no positive value becomes all zeros.

The classifier is trained with a frozen backbone. Grad-CAM needs
`requires_grad=True` on the input tensor for that case
(`src/explain/gradcam.py`).

## 2. Encoder (`encoder.py`)

Input: `[N, 1, 64, 64]`. Output: `[N, 64]`, each row length 1.

```
Conv 1→16, 3×3, pad 1 · BatchNorm · ReLU · MaxPool 2
Conv 16→32, 3×3, pad 1 · BatchNorm · ReLU · MaxPool 2
Conv 32→64, 3×3, pad 1 · BatchNorm · ReLU · global average pool
Linear 64→64 · ReLU · Linear 64→64 · L2 normalize
```

31,840 parameters. The encoder is not told which method made a map. The
three methods go through the same weights, which is the Siamese part.

## 3. Loss (`losses.py`)

Supervised contrastive loss (SupCon, Khosla et al. 2020), temperature
τ = 0.1. For anchor `i` with label `y_i`, `P(i)` is every other map in the
batch with the same label, `A(i)` is every other map in the batch:

```
L = mean over anchors with |P(i)| > 0 of
      − (1 / |P(i)|) · Σ_{p ∈ P(i)}  log [ exp(z_i·z_p / τ) / Σ_{a ∈ A(i)} exp(z_i·z_a / τ) ]
```

Label `y` = the class the classifier predicted for the image. So maps of
the same predicted class are pulled together, whichever method made them,
and maps of different classes are pushed apart. The three maps of one image
always share a label, so they are always positives of each other.

## 4. Training (`train_contrastive.py`)

| Item | Value |
|---|---|
| Data | cached maps from `build_explanation_dataset.py` (60 test-set images of the classifier, 3 maps each) |
| Split | by image. Images with `sample_id % 5 == 4` are held out (20%). All three maps of an image stay on the same side |
| Optimizer | Adam, learning rate 1e-3 |
| Label | `class` (default): the predicted class. `image`: the image id, so only the three maps of one image are positives. Set with `--label` |
| Epochs, batch | 100, 64 (last partial batch dropped when there are more than 64 training maps) |
| Seed | 0 (`torch.manual_seed`) |
| Logged | training loss and held-out loss every 10 epochs |

The held-out images are chosen from the classifier's test split, so the
classifier has not trained on them either.

`train_contrastive.py` holds out fold 4 and saves the encoder the API loads.
`cross_validate.py` repeats this for all 5 folds (fold k holds out images with
`sample_id % 5 == k`) and for every label and epoch setting given on the
command line. It reports every setting.

## 5. Evaluation (`link.py`, `analyze.py`)

Computed on the held-out images only (training images are also reported, for
the gap). For every ordered pair of methods (query a → target b), and every
held-out image `i`: rank the target maps of all held-out images by cosine
similarity to the query map of image `i`.

| Metric | Meaning |
|---|---|
| top-1 same image | the nearest target map comes from image `i` |
| top-3 same image | image `i` is among the 3 nearest |
| top-1 same class | the nearest target map has the same predicted class as image `i` |
| mean similarity, same image / other image | average cosine of the matching pair vs. non-matching pairs |
| chance | 1/N, 3/N, and the share of images that share image `i`'s class, from the same data |

N is the number of held-out images. Reported per pair (6 ordered pairs) and
averaged.

Two baselines use the same procedure:

- **Untrained encoder:** same architecture, random weights.
- **Raw maps:** no encoder. Each map is flattened and mean-centred, so the
  dot product is the Pearson correlation of the two maps.

The trained encoder is only worth using if it beats both. In the run it did
not (doc 12).

Two more outputs:

- **Raw maps on all cached images.** Raw correlation needs no training, so it
  is also measured on all 60 images at once, per method pair
  (`summary.json`, key `raw_all_images`). The pool is larger, so chance is
  1/60.
- **Cross-validation** (`cross_validate.py`): mean and standard deviation over
  the 5 folds, for each setting, for the untrained encoder and for raw maps
  (`<model>_cross_validation.json`). Use this, not the single fold, to judge
  an encoder.

## 6. Outputs

`outputs/correlate/` (not committed):

| Path | Content |
|---|---|
| `maps/*.npy`, `metadata.csv` | cached maps and their image, class and method |
| `checkpoints/encoder_<model>.pt` | trained encoder |
| `checkpoints/encoder_<model>_history.json` | training and held-out loss per epoch |
| `analysis/<model>_summary.json` | held-out fold numbers, and raw-map numbers on all images |
| `analysis/<model>_cross_validation.json` | 5-fold results for every setting, the untrained encoder and raw maps |
| `analysis/<model>_per_image_correlation.csv` | similarity per image and method pair |
| `analysis/<model>_correlation_matrix.png`, `<model>_tsne.png` | figures (the matrix shows encoder similarities, which are high for every pair; read it with doc 12) |

Copies of the figures used in the docs are in `docs/figures/`.

## 7. API (`src/api/main.py`)

`POST /correlate?model=<name>` with an image file. Runs the three methods
(about 20 seconds to a minute on CPU) and returns:

| Field | Content |
|---|---|
| `prediction`, `confidence` | classifier output |
| `pearson` | Pearson correlation of the maps for `gradcam-shap`, `gradcam-lime`, `shap-lime` |
| `similarity` | cosine similarity of the trained encoder's embeddings for the same pairs; `null` if no encoder exists for the model. Not shown in the frontend: on the demo image it was 0.83 to 0.94 for every pair |
| `reference` | mean Pearson correlation for the same image and for different images over the cached images (from `summary.json`) |
| `nearest` | for this image's Grad-CAM map, the 3 closest cached SHAP maps and 3 closest cached LIME maps by Pearson correlation (image name, predicted class, correlation) |
| `image_png_base64`, `overlays_png_base64` | the image and the three maps drawn over it. The colour scale is stretched so the brightest 1% of pixels reach the top; the returned numbers use the maps unstretched |

`GET /results` returns the numbers shown in the frontend's Results tab,
including the cross-validation file.

## 8. Limits

- Small data: 60 images, 12 held out. One model, one data split, one seed.
  Differences of a few points are noise at this size.
- Maps are only compared after being reduced to 64×64 and scaled to [0,1].
  Absolute size of a contribution is lost.
- Evidence against the class is dropped for SHAP and LIME.
- LIME maps are blocky because of the superpixels. The encoder may use that
  as a cue for the method. The t-SNE figure (coloured by method) is the
  check.
- The label is the predicted class. With wrong predictions the encoder
  learns to link explanations of the wrong class.
- SHAP uses a random background of 30 images and samples 50 times inside.
  A different background or a second run gives a different map (Grad-CAM–SHAP
  correlation on one image: 0.17 and 0.19 in two runs).
- The embeddings group by method, not by class (t-SNE in doc 12), and on the
  demo image all pairs had cosine similarity 0.83 to 0.94. The encoder
  learned to tell methods apart.
- Only MobileNetV2 was run. The code takes `--model resnet50` but no ResNet50
  maps were built.
