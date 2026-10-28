# 11. Explanation correlation: how to run it

Idea and results: [`12_linking_xai_methods.md`](12_linking_xai_methods.md).
Design details: [`13_correlation_model_spec.md`](13_correlation_model_spec.md).

You need a trained classifier first (`python -m src.train --model <name>`).
Run each step for one model at a time. Heavy steps use a lot of memory; on an
8 GB laptop close other apps and use `FORCE_CPU=1`.

```bash
# 1. Cache Grad-CAM, SHAP and LIME maps for 60 test images (about 20 min on CPU)
FORCE_CPU=1 python -m src.correlate.build_explanation_dataset \
    --model mobilenet_v2 --n_images 60 --lime_samples 150

# 2. Train the encoder (under a minute)
python -m src.correlate.train_contrastive --model mobilenet_v2 --epochs 100

# 3. Measure it on the held-out images; writes numbers and figures
python -m src.correlate.analyze --model mobilenet_v2

# 4. Judge it properly: 5-fold cross-validation, every setting reported (about 10 min)
FORCE_CPU=1 python -m src.correlate.cross_validate --model mobilenet_v2 \
    --labels class image --epochs 30 100 300
```

Outputs are in `outputs/correlate/` (see section 6 of the spec).

To use it on one new image, start the app (`uvicorn src.api.main:app --port 8000`),
open <http://localhost:8000>, and use the Explain tab. Or call
`POST /correlate?model=mobilenet_v2` with an image file.

Re-running step 1 replaces the cached maps and the images chosen. Run steps 2
to 4 again after it.

The Explain tab works without step 2 (it uses plain correlation). Steps 1 and 3
are needed for the "typical" values and the nearest-map lists.
