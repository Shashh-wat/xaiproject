"""Streamlit dashboard for interactive inspection: upload a leaf image,
pick a model, see the prediction + Grad-CAM (always) and optional SHAP/LIME
overlays. Also supports a side-by-side model-comparison mode.

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `streamlit run src/dashboard/app.py` to resolve `src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import streamlit as st
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import CFG
from src.compare_explanations import heatmap_iou
from src.explain.common import load_model, predict
from src.explain.gradcam import compute_gradcam
from src.data.dataset import build_transforms, denormalize

st.set_page_config(page_title="Explainable Crop Disease Detection",
                    layout="wide")
st.title("🌿 Explainable Crop Disease Detection")
st.caption("BCSE418L XAI assignment — Agriculture / Crop Disease Detection "
           "(CNN + Grad-CAM/SHAP/LIME)")


@st.cache_resource
def get_model(name: str):
    return load_model(name)


def checkpoint_missing(name: str) -> bool:
    return not (CFG.checkpoints_dir / f"{name}.pt").exists()


def image_to_tensor(pil_img: Image.Image):
    transform = build_transforms(train=False)
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0)
    rgb_float = np.array(denormalize(tensor[0])).transpose(1, 2, 0)
    return tensor, rgb_float


mode = st.sidebar.radio(
    "Mode", ["Single model", "Compare models", "Explanation Correlation"])
uploaded = st.sidebar.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a leaf image from the sidebar to get started. "
             "Sample images live under data/raw/plantvillage/<class>/ "
             "once you've run `python -m src.data.download`.")
    st.stop()

pil_img = Image.open(uploaded)
tensor, rgb_float = image_to_tensor(pil_img)

if mode == "Single model":
    model_name = st.sidebar.selectbox("Model", ["mobilenet_v2", "resnet50"])
    if checkpoint_missing(model_name):
        st.error(f"No checkpoint found for {model_name}. Train it first: "
                 f"`python -m src.train --model {model_name}`")
        st.stop()

    model, class_names = get_model(model_name)
    topk = predict(model, tensor, class_names, topk=3)
    heatmap, pred_idx = compute_gradcam(model, tensor)
    overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_float, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

    st.subheader("Prediction")
    for cls, prob in topk:
        st.write(f"**{cls}** — {prob:.1%}")
        st.progress(prob)

    with st.expander("Show SHAP explanation (slower)"):
        if st.button("Run SHAP"):
            from src.explain.shap_explain import compute_shap, sample_background
            with st.spinner("Sampling background + computing SHAP values ..."):
                background = sample_background(n=30)
                attribution, shap_pred_idx = compute_shap(model, tensor, background)
            vmax = np.percentile(np.abs(attribution), 99) + 1e-8
            fig_col1, fig_col2 = st.columns(2)
            with fig_col2:
                st.write(f"SHAP explains: **{class_names[shap_pred_idx]}**")
                st.image(rgb_float, caption="base image", use_container_width=True)
            st.caption("Red = supports predicted class, Blue = opposes it")

    with st.expander("Show LIME explanation (slower)"):
        if st.button("Run LIME"):
            from src.explain.lime_explain import compute_lime
            from skimage.segmentation import mark_boundaries
            with st.spinner("Running LIME (perturbation sampling) ..."):
                temp, mask, lime_pred_idx = compute_lime(
                    model, rgb_float, class_names, num_samples=300)
            boundary_img = mark_boundaries(temp / 255.0, mask)
            st.write(f"LIME explains: **{class_names[lime_pred_idx]}**")
            st.image(boundary_img, caption="LIME superpixels",
                     use_container_width=True)

elif mode == "Explanation Correlation":
    st.subheader("Do Grad-CAM, SHAP, and LIME actually agree?")
    st.caption(
        "Instead of raw pixel overlap, this embeds all three explanation "
        "maps with a Siamese/SupCon-trained encoder (`src/correlate/`) that "
        "learns to place explanations of the same predicted class close "
        "together, regardless of which method produced them. Cosine "
        "similarity between two methods' embeddings for *this* image is "
        "their learned agreement score."
    )
    model_name = st.sidebar.selectbox("Model", ["mobilenet_v2", "resnet50"],
                                       key="corr_model")
    if checkpoint_missing(model_name):
        st.error(f"No checkpoint found for {model_name}. Train it first: "
                 f"`python -m src.train --model {model_name}`")
        st.stop()

    from src.correlate.train_contrastive import CKPT_DIR as CORR_CKPT_DIR
    encoder_path = CORR_CKPT_DIR / f"encoder_{model_name}.pt"
    if not encoder_path.exists():
        st.error(
            f"No trained correlation encoder for {model_name} at "
            f"{encoder_path}. Build the cache and train it first:\n\n"
            f"```\npython -m src.correlate.build_explanation_dataset "
            f"--model {model_name}\npython -m src.correlate.train_contrastive "
            f"--model {model_name}\n```"
        )
        st.stop()

    if st.button("Compute explanation correlation"):
        from src.correlate.score_single import score_image

        model, class_names = get_model(model_name)
        with st.spinner("Computing Grad-CAM, SHAP, and LIME maps, then "
                         "embedding them ..."):
            maps, pred_idx, similarities = score_image(
                model, tensor, rgb_float, class_names, model_name,
                lime_samples=300,
            )

        st.write(f"Prediction: **{class_names[pred_idx]}**")

        cols = st.columns(3)
        for col, method in zip(cols, ["gradcam", "shap", "lime"]):
            with col:
                st.image(maps[method], caption=method, use_container_width=True,
                         clamp=True)

        st.subheader("Learned method agreement (cosine similarity)")
        mcols = st.columns(3)
        for col, (pair, sim) in zip(mcols, similarities.items()):
            with col:
                st.metric(pair, f"{sim:.2f}")
        low_pair = min(similarities, key=similarities.get)
        if similarities[low_pair] < 0.3:
            st.warning(f"Low agreement between {low_pair} on this image — "
                       "the methods are pointing at different evidence for "
                       "the same prediction.")

else:
    if checkpoint_missing("mobilenet_v2") or checkpoint_missing("resnet50"):
        st.error("Both model checkpoints are required for comparison mode. "
                 "Train them with `python -m src.train --model mobilenet_v2` "
                 "and `python -m src.train --model resnet50`.")
        st.stop()

    model_a, class_names = get_model("mobilenet_v2")
    model_b, _ = get_model("resnet50")

    heat_a, pred_a = compute_gradcam(model_a, tensor)
    heat_b, pred_b = compute_gradcam(model_b, tensor)
    iou = heatmap_iou(heat_a, heat_b, top_pct=0.25)

    overlay_a = show_cam_on_image(rgb_float, heat_a, use_rgb=True)
    overlay_b = show_cam_on_image(rgb_float, heat_b, use_rgb=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(rgb_float, caption="Original", use_container_width=True)
    with col2:
        st.image(overlay_a, caption=f"MobileNetV2 -> {class_names[pred_a]}",
                 use_container_width=True)
    with col3:
        st.image(overlay_b, caption=f"ResNet50 -> {class_names[pred_b]}",
                 use_container_width=True)

    st.metric("Grad-CAM heatmap agreement (IoU, top 25% region)",
              f"{iou:.2f}")
    if pred_a != pred_b:
        st.warning("The two models disagree on the predicted class.")
