# 4. Baseline Models

Two architecturally different backbones, both fine-tuned from ImageNet
weights via transfer learning (fast to train on a laptop GPU or even CPU for
a few epochs, given PlantVillage's lab-condition images make this an easy
task):

| | Model A | Model B |
|---|---|---|
| Backbone | MobileNetV2 | ResNet50 |
| Depth/family | Shallow, depthwise-separable convs, mobile-oriented | Deep, residual connections |
| Params (ImageNet head removed) | ~2.2M | ~23M |
| Role in the comparison | "Does a smaller/shallower model produce noisier or more diffuse Grad-CAM heatmaps?" | "Does depth buy more spatially precise explanations, or does it overfit to background texture?" |

Both are built in `src/models/factory.py::build_model(name, num_classes)`,
which:
- loads `torchvision.models.mobilenet_v2` / `resnet50` with
  `weights="IMAGENET1K_V1"`
- freezes the backbone by default (`--freeze_backbone`, on by default for a
  fast first pass), replaces the classifier head with a `Linear` layer sized
  to `num_classes`
- exposes `.get_target_layer()` used by Grad-CAM (last conv block for each
  architecture — `features[-1]` for MobileNetV2, `layer4[-1]` for ResNet50)

## Training

```bash
python -m src.train --model mobilenet_v2 --epochs 10
python -m src.train --model resnet50 --epochs 10
```

- Loss: cross-entropy (class-weighted, since the dataset is imbalanced —
  weights computed from the EDA class counts)
- Optimizer: Adam, lr=1e-3 on the new head (and 1e-5 on unfrozen backbone
  layers if you pass `--unfreeze_last_n 2` for a second, higher-accuracy
  pass)
- Checkpoints saved to `outputs/checkpoints/<model_name>.pt`
- Best-val-accuracy checkpoint kept; training curve logged to
  `outputs/checkpoints/<model_name>_history.json`

## Evaluation

```bash
python -m src.evaluate --model mobilenet_v2
python -m src.evaluate --model resnet50
```

Reports accuracy, macro-F1, per-class precision/recall, and a confusion
matrix (`outputs/eval/<model_name>_confusion_matrix.png`) — the per-class
breakdown is what feeds the fairness angle in step 7.

## What to report in the paper

- A table: model, params, train/val/test accuracy, macro-F1, training time
- The confusion matrix for each model (2 figures)
- 1–2 sentences on which classes are hardest for each model (usually visually
  similar diseases across related crops, e.g. different blight types)
