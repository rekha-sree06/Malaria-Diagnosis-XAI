"""run_xai_validated.py

Mathematically Rigorous and Validated Explainable AI (XAI) Suite for Malaria Diagnosis.

This module provides verified implementations of Grad-CAM and LIME for the
Xception architecture trained on thin blood smear cell images (Model 1).

Key Scientific & Engineering Principles:
-----------------------------------------
1. Decoupled Two-Part Gradient Sub-Models:
   Builds a dedicated feature extractor (inputs -> target conv layer) and an
   independent classifier head (target conv layer -> linear logit).
   The target convolutional activation tensor is explicitly watched on the
   GradientTape BEFORE downstream logit operations are evaluated, ensuring
   unbroken, exact gradient connectivity without sigmoid saturation.
2. Verified Pre-Sigmoid Logit Layer Construction:
   Instantiates the linear Dense layer without weights in constructor, builds it
   within the functional graph, and sets weights cleanly via set_weights().
   Numerically verifies that tf.sigmoid(logit) matches the original model output
   using a numerically stabilized TensorFlow kernel.
3. Class Directionality Formulation:
   Properly handles binary classification semantics:
   - Class 0 = Parasitized: score = -z  (maximizing P(Parasitized) = sigmoid(-z))
   - Class 1 = Uninfected:  score = +z  (maximizing P(Uninfected)  = sigmoid(+z))
4. Identical Preprocessing Pipeline:
   Uses the exact TensorFlow PNG decoding and bilinear resizing (224x224, float32,
   [0.0, 1.0]) used during training in model1.py.
5. Direct LIME Attribution Mask Construction:
   Directly extracts positive (weight > 0) and negative (weight < 0) feature weights
   from `explanation.local_exp[target_class]` and maps them onto `explanation.segments`.
   Avoids unsupported API keywords and generates three distinct visualizations:
   positive-only (green), negative-only (red), and combined overlays.
6. Robust LIME Repeatability Evaluation:
   Compares spatial pixel footprints (selected vs unselected pixels) separately
   for positive and negative attributions. Handles empty attribution masks safely.
   Documents that fixed segmentation isolates surrogate sampling variability, while
   varying segmentation introduces both segmentation partition shifts and surrogate sampling variability.
7. Strengthened Grad-CAM Gradient Validation:
   Verifies gradient finiteness, computes gradient L2 norm, reports non-zero status,
   raises clear errors on NaN/Inf, and flags gradient vanishing explicitly.
8. Defensive Probability Validation:
   Checks array dimensionality, finite values, range [0.0, 1.0], and partition of
   unity before allowing downstream indexing.
9. Non-Destructive In-Memory Randomization:
   Performs partial classifier-head randomization on an isolated in-memory clone.
   Verifies that every original model weight remains bitwise identical via exact
   pre/post array comparisons.
10. Comprehensive Individual & Summary Artifact Outputs:
    Saves Grad-CAM Parasitized, Grad-CAM Uninfected, LIME positive, LIME negative,
    LIME combined, and grid summary visualizations as separate files exclusively inside
    `results_validated/` using collision-safe unique filenames.
11. Controlled Experiment Workflow:
    Provides an explicit, callable pipeline function (`run_validated_xai_pipeline`)
    without executing anything automatically upon import.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from skimage.segmentation import mark_boundaries, quickshift, slic
import tensorflow as tf

from lime import lime_image


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclasses.dataclass
class PreprocessingConfig:
    """Input dimensions and scaling configuration."""
    img_height: int = 224
    img_width: int = 224
    channels: int = 3
    scale_min: float = 0.0
    scale_max: float = 1.0
    target_layer_name: str = "block14_sepconv2_act"
    pooling_layer_name: str = "global_avg_pool"
    penultimate_layer_name: str = "dense_1024"
    dropout_layer_name: str = "dropout"
    output_layer_name: str = "output"


@dataclasses.dataclass
class GradCamConfig:
    """Grad-CAM hyperparameter configuration."""
    colormap: int = cv2.COLORMAP_JET
    alpha: float = 0.4
    eps: float = 1e-12


@dataclasses.dataclass
class LimeConfig:
    """LIME explanation hyperparameter configuration."""
    num_samples: int = 1000
    num_features: int = 8
    random_seed: int = 42
    segmentation_algorithm: str = "quickshift"  # 'quickshift' or 'slic'
    kernel_size: float = 2.0
    max_dist: float = 10.0
    ratio: float = 0.2
    n_segments: int = 50
    compactness: float = 10.0
    hide_color_mode: str = "mean"  # 'mean', 'black', or 'rgb'
    custom_hide_color: Optional[Tuple[float, float, float]] = None


@dataclasses.dataclass
class ValidationConfig:
    """Validation protocol parameters."""
    random_seeds: List[int] = dataclasses.field(default_factory=lambda: [42, 101, 2024])
    top_k_percent: float = 0.10
    isolate_surrogate_sampling: bool = True  # If True, fix segmentation grid across seeds to isolate surrogate variance


# ==============================================================================
# Preprocessing & Model Loading Utilities
# ==============================================================================

def load_and_preprocess_image(
    image_path: str,
    config: PreprocessingConfig = PreprocessingConfig()
) -> np.ndarray:
    """Load and preprocess an RGB image using the exact TensorFlow pipeline from model1.py.

    Matches model1.py lines 102-107:
      - tf.io.read_file
      - tf.image.decode_png(image, channels=3)
      - tf.image.resize(image, [224, 224], method=BILINEAR)
      - tf.cast(image, tf.float32) / 255.0

    Args:
        image_path: Path to the PNG image.
        config: Preprocessing specifications.

    Returns:
        np.ndarray of shape (1, 224, 224, 3) in float32 scaled to [0.0, 1.0].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    raw_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_png(raw_bytes, channels=config.channels)
    img = tf.image.resize(img, [config.img_height, config.img_width], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, axis=0).numpy()


def build_gradcam_submodels(
    model: tf.keras.Model,
    config: PreprocessingConfig = PreprocessingConfig()
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build decoupled feature extractor and classifier head sub-models for Grad-CAM.

    To ensure the GradientTape watches convolutional activations BEFORE downstream
    operations are evaluated, we construct two separate sub-models:
      1. feature_extractor: model.inputs -> target_conv_layer.output
      2. classifier_head: target_conv_layer.output -> [pre_sigmoid_logit, sigmoid_prob]

    In classifier_head, the linear logit is constructed by instantiating an
    un-activated Dense layer without weights in the constructor, building it
    within the functional graph, and setting its weights cleanly via set_weights().

    Args:
        model: Loaded tf.keras.Model.
        config: Layer name and preprocessing configuration.

    Returns:
        Tuple of (feature_extractor_model, classifier_head_model).
    """
    target_conv_layer = model.get_layer(config.target_layer_name)
    pooling_layer = model.get_layer(config.pooling_layer_name)
    penultimate_layer = model.get_layer(config.penultimate_layer_name)
    output_layer = model.get_layer(config.output_layer_name)

    # 1. Feature Extractor Sub-Model
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=target_conv_layer.output,
        name="gradcam_feature_extractor"
    )

    # 2. Classifier Head Sub-Model (operates on convolutional feature maps)
    conv_feature_input = tf.keras.Input(
        shape=target_conv_layer.output.shape[1:],
        name="conv_feature_input"
    )
    x = pooling_layer(conv_feature_input)
    x = penultimate_layer(x)

    # Instantiate linear Dense layer WITHOUT passing weights to constructor
    linear_dense = tf.keras.layers.Dense(
        units=output_layer.units,
        activation=None,
        use_bias=output_layer.use_bias,
        name="pre_sigmoid_linear_dense"
    )
    linear_logit = linear_dense(x)

    # Assign weights cleanly after the layer is built in the functional graph
    linear_dense.set_weights(output_layer.get_weights())

    # Add sigmoid activation layer to verify output probability
    sigmoid_prob = tf.keras.layers.Activation("sigmoid", name="head_sigmoid_prob")(linear_logit)

    classifier_head = tf.keras.Model(
        inputs=conv_feature_input,
        outputs=[linear_logit, sigmoid_prob],
        name="gradcam_classifier_head"
    )

    return feature_extractor, classifier_head


def verify_logit_extraction(
    model: tf.keras.Model,
    feature_extractor: tf.keras.Model,
    classifier_head: tf.keras.Model,
    sample_tensor: np.ndarray,
    atol: float = 1e-5
) -> Dict[str, Any]:
    """Numerically verify that reconstructed logit and probability match the original model.

    Uses a numerically stabilized TensorFlow sigmoid kernel to eliminate exponential
    overflow risks.
    """
    orig_prob = model(sample_tensor, training=False).numpy()
    conv_feats = feature_extractor(sample_tensor, training=False)
    logit, reconstructed_prob = classifier_head(conv_feats, training=False)

    logit_tf = tf.convert_to_tensor(logit)
    rec_prob_np = reconstructed_prob.numpy()

    # Numerically stable TensorFlow sigmoid calculation
    tf_stabilized_sigmoid = tf.sigmoid(logit_tf).numpy()
    max_prob_diff = float(np.max(np.abs(rec_prob_np - orig_prob)))
    max_tf_sigmoid_diff = float(np.max(np.abs(tf_stabilized_sigmoid - orig_prob)))

    is_valid = bool(max_prob_diff < atol and max_tf_sigmoid_diff < atol)
    if not is_valid:
        raise RuntimeError(
            f"Logit reconstruction verification failed! Max discrepancy: {max_prob_diff:.6e}"
        )

    return {
        "is_verified": is_valid,
        "max_prob_discrepancy": max_prob_diff,
        "max_tf_sigmoid_discrepancy": max_tf_sigmoid_diff,
        "sample_raw_logit": float(logit.numpy()[0][0]),
        "sample_prob_uninfected": float(orig_prob[0][0]),
        "sample_prob_parasitized": float(1.0 - orig_prob[0][0])
    }


# ==============================================================================
# Grad-CAM Engine
# ==============================================================================

def compute_gradcam_heatmap(
    feature_extractor: tf.keras.Model,
    classifier_head: tf.keras.Model,
    img_tensor: np.ndarray,
    target_class: int,
    config: GradCamConfig = GradCamConfig()
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute mathematically verified Grad-CAM for target class using pre-sigmoid logits.

    The convolutional activations are evaluated first and watched on the GradientTape
    BEFORE the downstream operations in classifier_head are evaluated.

    Class Directionality:
      - Class 0 (Parasitized): target_score = -z
      - Class 1 (Uninfected):  target_score = +z

    Gradient Validation:
      - Checks `grads is not None`.
      - Checks gradient finiteness (`tf.reduce_all(tf.math.is_finite(grads))`).
      - Computes gradient L2 norm and reports non-zero status.
      - Explicitly flags gradient vanishing when norm <= config.eps.

    Args:
        feature_extractor: Model extracting conv layer output.
        classifier_head: Model mapping conv features -> [linear_logit, prob].
        img_tensor: Batch array of shape (1, 224, 224, 3).
        target_class: 0 for Parasitized, 1 for Uninfected.
        config: Grad-CAM hyperparameter configuration.

    Returns:
        Tuple of (resized_cam [224, 224], diagnostics_dict).
    """
    if target_class not in (0, 1):
        raise ValueError(f"Invalid target_class {target_class}. Must be 0 (Parasitized) or 1 (Uninfected).")

    # 1. Forward pass through feature extractor
    conv_outputs = feature_extractor(img_tensor, training=False)

    # 2. Watch conv_outputs on GradientTape BEFORE classifier_head forward pass
    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)
        logit, prob = classifier_head(conv_outputs, training=False)

        # Directional class score:
        # P(Uninfected) = sigmoid(z). P(Parasitized) = sigmoid(-z).
        # Maximizing Parasitized corresponds to logit = -z.
        if target_class == 0:
            target_score = -logit[:, 0]
        else:
            target_score = logit[:, 0]

    # 3. Compute exact gradients of directional logit with respect to conv_outputs
    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM computation failed: Gradients could not be computed (tape is disconnected).")

    # 4. Strict gradient validation: finiteness, norm, and vanishing checks
    is_finite_tensor = tf.reduce_all(tf.math.is_finite(grads))
    is_finite = bool(is_finite_tensor.numpy())
    if not is_finite:
        raise ValueError("Grad-CAM computation failed: Gradients contain non-finite values (NaN or Inf).")

    grad_norm = float(tf.norm(grads).numpy())
    is_gradient_nonzero = bool(grad_norm > config.eps)
    is_gradient_vanishing = not is_gradient_nonzero

    if is_gradient_vanishing:
        cam_warning = (
            f"Gradient vanishing detected: L2 norm ({grad_norm:.6e}) <= eps ({config.eps:.1e}). "
            "Feature maps exert negligible linear influence on the target class score; heatmap is flat zero."
        )
    else:
        cam_warning = None

    # 5. Channel importance weights via Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

    # 6. Weighted combination of feature maps: L = ReLU( sum_k alpha_k * A^k )
    feature_maps = conv_outputs[0]  # shape: (H_feat, W_feat, channels)
    cam = feature_maps @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0.0)  # Rectification (ReLU)

    max_val = float(tf.reduce_max(cam).numpy())
    min_val = float(tf.reduce_min(cam).numpy())

    # Safe normalization avoiding division by zero
    if not np.isfinite(max_val) or max_val <= config.eps:
        normalized_cam = np.zeros(cam.shape, dtype=np.float32)
        is_degenerate = True
    else:
        normalized_cam = (cam / max_val).numpy()
        is_degenerate = False

    # Spatially aligned bilinear resize to match input resolution (224, 224)
    resized_cam = cv2.resize(
        normalized_cam,
        (img_tensor.shape[2], img_tensor.shape[1]),
        interpolation=cv2.INTER_LINEAR
    )

    diagnostics = {
        "target_class": target_class,
        "class_name": "Parasitized" if target_class == 0 else "Uninfected",
        "raw_logit_z": float(logit.numpy()[0][0]),
        "p_uninfected": float(prob.numpy()[0][0]),
        "p_parasitized": float(1.0 - prob.numpy()[0][0]),
        "gradient_l2_norm": grad_norm,
        "is_gradient_finite": is_finite,
        "is_gradient_nonzero": is_gradient_nonzero,
        "is_gradient_vanishing": is_gradient_vanishing,
        "cam_max_raw": max_val,
        "cam_min_raw": min_val,
        "is_degenerate": is_degenerate,
        "warning": cam_warning
    }

    return resized_cam, diagnostics


def create_gradcam_overlay(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Safely overlay a normalized [0, 1] heatmap onto an RGB image.

    Corrects OpenCV's default BGR colormap output to RGB.
    """
    if original_rgb.max() <= 1.0:
        base_img = (original_rgb * 255.0).astype(np.uint8)
    else:
        base_img = original_rgb.astype(np.uint8)

    uint8_heatmap = np.uint8(255.0 * np.clip(heatmap, 0.0, 1.0))
    bgr_colored = cv2.applyColorMap(uint8_heatmap, colormap)
    rgb_colored = cv2.cvtColor(bgr_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(base_img, 1.0 - alpha, rgb_colored, alpha, 0)
    return overlay


# ==============================================================================
# LIME Engine
# ==============================================================================

def make_lime_predict_fn(
    model: tf.keras.Model,
    config: PreprocessingConfig = PreprocessingConfig()
) -> Callable[[List[np.ndarray]], np.ndarray]:
    """Create a verified predict_fn for LIME returning shape (N, 2).

    Column 0: P(Parasitized) = 1.0 - p
    Column 1: P(Uninfected)  = p
    """
    def predict_fn(images: List[np.ndarray] | np.ndarray) -> np.ndarray:
        batch = []
        for img in images:
            if img.shape[0] != config.img_height or img.shape[1] != config.img_width:
                resized = cv2.resize(img, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
            else:
                resized = img

            arr = np.asarray(resized, dtype=np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            batch.append(arr)

        batch_tensor = np.stack(batch, axis=0)
        p_uninfected = model.predict(batch_tensor, verbose=0)  # shape (N, 1)
        p_parasitized = 1.0 - p_uninfected
        return np.hstack([p_parasitized, p_uninfected])  # shape (N, 2)

    return predict_fn


def get_compatible_segmentation_fn(config: LimeConfig) -> Callable[[np.ndarray], np.ndarray]:
    """Generate a reproducible superpixel segmentation function compatible across skimage versions.

    Dynamically inspects parameters to support both 'rng' (skimage >= 0.20) and
    'random_seed' (skimage < 0.20).
    """
    if config.segmentation_algorithm == "quickshift":
        sig = inspect.signature(quickshift)
        kwargs: Dict[str, Any] = {
            "kernel_size": config.kernel_size,
            "max_dist": config.max_dist,
            "ratio": config.ratio,
        }
        if "rng" in sig.parameters:
            kwargs["rng"] = config.random_seed
        elif "random_seed" in sig.parameters:
            kwargs["random_seed"] = config.random_seed

        return lambda img: quickshift(img, **kwargs)

    elif config.segmentation_algorithm == "slic":
        sig = inspect.signature(slic)
        kwargs: Dict[str, Any] = {
            "n_segments": config.n_segments,
            "compactness": config.compactness,
            "start_label": 0
        }
        if "rng" in sig.parameters:
            kwargs["rng"] = config.random_seed
        elif "random_seed" in sig.parameters:
            kwargs["random_seed"] = config.random_seed

        return lambda img: slic(img, **kwargs)

    else:
        raise ValueError(f"Unsupported segmentation algorithm: {config.segmentation_algorithm}")


def compute_lime_explanation(
    model: tf.keras.Model,
    img_array: np.ndarray,
    target_class: Optional[int] = None,
    lime_config: LimeConfig = LimeConfig(),
    prep_config: PreprocessingConfig = PreprocessingConfig(),
    custom_segmentation_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Compute mathematically verified LIME explanation with positive and negative attributions.

    Args:
        model: Base trained model.
        img_array: Single image array (224, 224, 3) in [0, 1].
        target_class: 0 for Parasitized, 1 for Uninfected. If None, uses predicted class.
        lime_config: Hyperparameter settings for LIME.
        prep_config: Input specifications.
        custom_segmentation_fn: Optional pre-defined segmentation function to hold superpixels constant.

    Returns:
        Tuple of (LIME ImageExplanation object, diagnostics_dict).
    """
    predict_fn = make_lime_predict_fn(model, prep_config)

    # In-distribution occlusion baseline
    if lime_config.hide_color_mode == "mean":
        hide_color = tuple(np.mean(img_array, axis=(0, 1)))
    elif lime_config.hide_color_mode == "black":
        hide_color = (0.0, 0.0, 0.0)
    elif lime_config.hide_color_mode == "rgb" and lime_config.custom_hide_color is not None:
        hide_color = lime_config.custom_hide_color
    else:
        hide_color = tuple(np.mean(img_array, axis=(0, 1)))

    explainer = lime_image.LimeImageExplainer(random_state=lime_config.random_seed)
    if custom_segmentation_fn is not None:
        segmenter = custom_segmentation_fn
    else:
        segmenter = get_compatible_segmentation_fn(lime_config)

    # Obtain model prediction
    probs = predict_fn(np.expand_dims(img_array, axis=0))[0]
    predicted_class = int(np.argmax(probs))

    if target_class is None:
        chosen_class = predicted_class
    else:
        chosen_class = target_class

    explanation = explainer.explain_instance(
        image=img_array,
        classifier_fn=predict_fn,
        labels=(chosen_class,),
        hide_color=hide_color,
        num_samples=lime_config.num_samples,
        segmentation_fn=segmenter,
        random_seed=lime_config.random_seed
    )

    r2_score = explanation.score.get(chosen_class, float("nan"))
    intercept = explanation.intercept.get(chosen_class, float("nan"))

    diagnostics = {
        "explained_class": chosen_class,
        "class_name": "Parasitized" if chosen_class == 0 else "Uninfected",
        "predicted_class": predicted_class,
        "p_parasitized": float(probs[0]),
        "p_uninfected": float(probs[1]),
        "surrogate_r2": float(r2_score),
        "surrogate_intercept": float(intercept),
        "hide_color_used": hide_color,
        "num_samples": lime_config.num_samples,
        "random_seed": lime_config.random_seed
    }

    return explanation, diagnostics


def extract_lime_visualizations(
    explanation: Any,
    target_class: int,
    num_features: int = 5,
    alpha: float = 0.4
) -> Dict[str, Any]:
    """Extract three distinct LIME visualizations directly from local_exp and segments.

    Highlights selected superpixel regions with semi-transparent color overlays
    while preserving the underlying RGB cell image details:
      - Positive attribution regions (weights > 0): Green overlay
      - Negative attribution regions (weights < 0): Red overlay
      - Combined visualization: Green for positive, Red for negative

    Class Ordering:
      Class 0 = Parasitized
      Class 1 = Uninfected

    Args:
        explanation: LIME ImageExplanation object.
        target_class: Target class index (0 or 1).
        num_features: Number of top superpixels to highlight.
        alpha: Overlay blending weight in [0.0, 1.0]. Defaults to 0.4.

    Returns:
        Dictionary containing positive_only, negative_only, and combined visualization arrays,
        along with spatial masks and selected feature IDs.
    """
    if target_class not in explanation.local_exp:
        raise KeyError(
            f"Target class {target_class} not found in explanation.local_exp. "
            f"Available classes: {list(explanation.local_exp.keys())}"
        )

    exp = explanation.local_exp[target_class]
    segments = explanation.segments
    image = explanation.image

    # 1. Positive superpixels: weights > 0 (evidence supporting target class)
    pos_tuples = sorted([x for x in exp if x[1] > 0], key=lambda x: x[1], reverse=True)[:num_features]
    pos_features = [x[0] for x in pos_tuples]

    # 2. Negative superpixels: weights < 0 (evidence opposing target class)
    neg_tuples = sorted([x for x in exp if x[1] < 0], key=lambda x: abs(x[1]), reverse=True)[:num_features]
    neg_features = [x[0] for x in neg_tuples]

    # 3. Construct exact spatial boolean masks directly from segments
    pos_mask = np.zeros(segments.shape, dtype=bool)
    for f in pos_features:
        pos_mask[segments == f] = True

    neg_mask = np.zeros(segments.shape, dtype=bool)
    for f in neg_features:
        neg_mask[segments == f] = True

    # 4. Generate highlighted overlays preserving the original RGB image underneath
    base_image = np.array(image, dtype=np.float32, copy=True)
    if base_image.max() > 1.0:
        base_image = base_image / 255.0
    base_image = np.clip(base_image, 0.0, 1.0)

    green_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Positive support in Green
    pos_vis = base_image.copy()
    if np.any(pos_mask):
        pos_vis[pos_mask] = (1.0 - alpha) * pos_vis[pos_mask] + alpha * green_color
    pos_vis = np.clip(pos_vis, 0.0, 1.0)

    # Negative support in Red
    neg_vis = base_image.copy()
    if np.any(neg_mask):
        neg_vis[neg_mask] = (1.0 - alpha) * neg_vis[neg_mask] + alpha * red_color
    neg_vis = np.clip(neg_vis, 0.0, 1.0)

    # Combined support: positive in Green, negative in Red
    combined_vis = base_image.copy()
    if np.any(pos_mask):
        combined_vis[pos_mask] = (1.0 - alpha) * combined_vis[pos_mask] + alpha * green_color
    if np.any(neg_mask):
        combined_vis[neg_mask] = (1.0 - alpha) * combined_vis[neg_mask] + alpha * red_color
    combined_vis = np.clip(combined_vis, 0.0, 1.0)

    return {
        "positive_only": pos_vis,
        "negative_only": neg_vis,
        "combined": combined_vis,
        "positive_mask": pos_mask,
        "negative_mask": neg_mask,
        "top_positive_features": pos_features,
        "top_negative_features": neg_features
    }


# ==============================================================================
# Validation & Diagnostic Suite (Descriptive Reporting)
# ==============================================================================

def validate_probability_formatting(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    test_batch: np.ndarray,
    atol: float = 1e-5
) -> Dict[str, Any]:
    """Test 1: Defensively verify predict_fn output shape, range, and partition of unity.

    Validates:
      1. Output is a 2D numpy array with shape (N, 2).
      2. All entries are finite (no NaN or Inf).
      3. All probabilities reside in [0.0, 1.0].
      4. Row sums equal 1.0 within tolerance atol.

    Raises:
        TypeError: If output is not np.ndarray.
        ValueError: If shape, finiteness, range, or partition of unity is violated.
    """
    probs = predict_fn(test_batch)

    # 1. Output type and dimensionality checks
    if not isinstance(probs, np.ndarray):
        raise TypeError(f"predict_fn must return np.ndarray, got {type(probs)}")

    if probs.ndim != 2:
        raise ValueError(
            f"predict_fn output must be 2-dimensional (N, 2), got ndim={probs.ndim} with shape {probs.shape}"
        )

    if probs.shape[1] != 2:
        raise ValueError(
            f"predict_fn must output exactly 2 class columns [P(Parasitized), P(Uninfected)], "
            f"got {probs.shape[1]} columns (shape={probs.shape})"
        )

    if probs.shape[0] != test_batch.shape[0]:
        raise ValueError(
            f"predict_fn batch size mismatch: input had {test_batch.shape[0]} samples, "
            f"output had {probs.shape[0]} samples"
        )

    # 2. Finite values check
    if not np.all(np.isfinite(probs)):
        raise ValueError("predict_fn output contains non-finite values (NaN or Inf)")

    # 3. Probability range check
    min_val, max_val = float(probs.min()), float(probs.max())
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(
            f"predict_fn probabilities out of valid range [0.0, 1.0]: min={min_val:.6e}, max={max_val:.6e}"
        )

    # 4. Partition of unity check (sum to 1.0 per row)
    row_sums = probs.sum(axis=1)
    max_sum_dev = float(np.max(np.abs(row_sums - 1.0)))
    if max_sum_dev > atol:
        raise ValueError(
            f"predict_fn probabilities do not sum to 1.0 per row: max deviation={max_sum_dev:.6e} (tolerance={atol})"
        )

    return {
        "check": "Probability Formatting (Strict Validation)",
        "shape": list(probs.shape),
        "is_shape_valid": True,
        "is_range_valid": True,
        "is_sum_to_one_valid": True,
        "is_finite": True,
        "max_row_sum_deviation": max_sum_dev,
        "is_compliant": True
    }


def evaluate_gradcam_class_contrast(
    feature_extractor: tf.keras.Model,
    classifier_head: tf.keras.Model,
    img_tensor: np.ndarray,
    config: ValidationConfig = ValidationConfig()
) -> Dict[str, Any]:
    """Test 2: Target-Class Contrastivity Descriptive Evaluation.

    Computes Grad-CAM for both Class 0 and Class 1 on the same image.
    Reports Spearman correlation and top-k Jaccard overlap descriptively.
    In binary classification, mutually exclusive classes ideally exhibit distinct
    spatial attributions; higher Jaccard overlap indicates lower class specificity.
    """
    cam_parasitized, diag_p = compute_gradcam_heatmap(
        feature_extractor, classifier_head, img_tensor, target_class=0
    )
    cam_uninfected, diag_u = compute_gradcam_heatmap(
        feature_extractor, classifier_head, img_tensor, target_class=1
    )

    flat_p = cam_parasitized.flatten()
    flat_u = cam_uninfected.flatten()

    rho, p_val = spearmanr(flat_p, flat_u)

    k = int(len(flat_p) * config.top_k_percent)
    top_p = np.zeros_like(flat_p, dtype=bool)
    top_u = np.zeros_like(flat_u, dtype=bool)
    top_p[np.argsort(flat_p)[-k:]] = True
    top_u[np.argsort(flat_u)[-k:]] = True

    intersection = np.logical_and(top_p, top_u).sum()
    union = np.logical_or(top_p, top_u).sum()
    jaccard_overlap = float(intersection / (union + 1e-12))

    return {
        "check": "Grad-CAM Target-Class Contrast (Descriptive)",
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "spearman_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "top_k_percent": config.top_k_percent,
        "top_k_jaccard_overlap": jaccard_overlap,
        "diag_parasitized": diag_p,
        "diag_uninfected": diag_u
    }


def evaluate_lime_local_fidelity(
    surrogate_r2: float
) -> Dict[str, Any]:
    """Test 3: LIME Local Surrogate Fidelity Reporting.

    Reports the ridge regression R^2 score descriptively. R^2 closer to 1.0
    indicates that the surrogate linear model closely approximates the deep model
    locally within the perturbed neighborhood.
    """
    return {
        "check": "LIME Local Fidelity (Descriptive)",
        "surrogate_r2": float(surrogate_r2),
        "interpretation_guide": (
            "R^2 measures local linear fit quality on perturbed samples. "
            "Higher values reflect higher local surrogate fidelity."
        )
    }


def _compute_spatial_jaccard_distribution(masks_list: List[np.ndarray]) -> Dict[str, Any]:
    """Compute pairwise spatial Jaccard similarities, handling empty masks safely.

    Convention for Empty Masks:
      When both masks A and B contain zero active pixels, they represent identical absence
      of attribution. Conventionally, IoU(empty, empty) is defined as 1.0 (perfect agreement on null attribution).
      To prevent a score of 1.0 from obscuring an absence of signal, `empty_mask_pair_count` explicitly
      tracks how many such pairs occurred.

    If fewer than 2 masks are available, pairwise comparison is impossible; returns status 'not_evaluable'.
    """
    n = len(masks_list)
    if n < 2:
        return {
            "status": "not_evaluable",
            "reason": f"Fewer than 2 masks available (n={n}); pairwise comparison impossible.",
            "pairwise_values": [],
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "total_pair_count": 0,
            "empty_mask_pair_count": 0,
            "non_empty_pair_count": 0,
        }

    pairwise_jaccard = []
    empty_pair_count = 0
    non_empty_pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(masks_list[i], masks_list[j]).sum()
            union = np.logical_or(masks_list[i], masks_list[j]).sum()
            if union == 0:
                # Both masks are completely empty (concordant absence of attribution)
                iou = 1.0
                empty_pair_count += 1
            else:
                iou = float(inter / union)
                non_empty_pair_count += 1
            pairwise_jaccard.append(iou)

    return {
        "status": "evaluable",
        "pairwise_values": pairwise_jaccard,
        "mean": float(np.mean(pairwise_jaccard)),
        "std": float(np.std(pairwise_jaccard)),
        "min": float(np.min(pairwise_jaccard)),
        "max": float(np.max(pairwise_jaccard)),
        "total_pair_count": len(pairwise_jaccard),
        "empty_mask_pair_count": empty_pair_count,
        "non_empty_pair_count": non_empty_pair_count,
        "empty_mask_convention": (
            "When both masks have zero active pixels, IoU is defined as 1.0 (concordant null attribution). "
            "Refer to empty_mask_pair_count to verify whether high scores reflect genuine spatial overlap "
            "or mutual absence of attribution."
        )
    }


def evaluate_lime_repeatability(
    model: tf.keras.Model,
    img_array: np.ndarray,
    target_class: int,
    base_lime_config: LimeConfig = LimeConfig(),
    val_config: ValidationConfig = ValidationConfig()
) -> Dict[str, Any]:
    """Test 4: LIME Stochastic Repeatability Evaluation across multiple random seeds.

    Evaluates spatial pixel-level Jaccard Index (IoU) separately for positive-support
    and negative-support masks. Validates that at least two distinct random seeds are provided.

    Methodological Principle:
      Fixed segmentation isolates surrogate sampling variability, while varying
      segmentation introduces both segmentation partition shifts and surrogate sampling variability.
    """
    unique_seeds = sorted(list(set(val_config.random_seeds)))
    if len(unique_seeds) < 2:
        return {
            "check": "LIME Repeatability Across Seeds (Descriptive)",
            "status": "not_evaluable",
            "reason": (
                f"Repeatability evaluation requires at least 2 distinct random seeds, "
                f"but received {len(unique_seeds)} distinct seed(s): {val_config.random_seeds}."
            ),
            "seeds_tested": unique_seeds,
            "positive_support_repeatability": {
                "status": "not_evaluable",
                "pairwise_values": [],
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "total_pair_count": 0,
                "empty_mask_pair_count": 0,
                "non_empty_pair_count": 0,
            },
            "negative_support_repeatability": {
                "status": "not_evaluable",
                "pairwise_values": [],
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "total_pair_count": 0,
                "empty_mask_pair_count": 0,
                "non_empty_pair_count": 0,
            },
            "r2_scores_across_seeds": [],
            "mean_r2": None,
            "std_r2": None,
            "methodological_documentation": (
                "Fixed segmentation isolates surrogate sampling variability, while varying "
                "segmentation introduces both segmentation partition shifts and surrogate sampling variability."
            )
        }

    pos_spatial_masks = []
    neg_spatial_masks = []
    r2_scores = []

    # If isolate_surrogate_sampling is True, pre-compute segmentation partition once
    if val_config.isolate_surrogate_sampling:
        segmenter = get_compatible_segmentation_fn(base_lime_config)
        fixed_segments = segmenter(img_array)
        custom_segmenter = lambda img: fixed_segments
        eval_scope = "Isolated LIME surrogate sampling variance (constant segmentation partition)"
    else:
        custom_segmenter = None
        eval_scope = "Compound segmentation boundary shift and surrogate sampling variance"

    for seed in unique_seeds:
        cfg = copy.deepcopy(base_lime_config)
        cfg.random_seed = seed
        explanation, diag = compute_lime_explanation(
            model=model,
            img_array=img_array,
            target_class=target_class,
            lime_config=cfg,
            custom_segmentation_fn=custom_segmenter
        )

        vis_dict = extract_lime_visualizations(
            explanation=explanation,
            target_class=target_class,
            num_features=cfg.num_features
        )
        pos_spatial_masks.append(vis_dict["positive_mask"])
        neg_spatial_masks.append(vis_dict["negative_mask"])
        r2_scores.append(diag["surrogate_r2"])

    pos_stats = _compute_spatial_jaccard_distribution(pos_spatial_masks)
    neg_stats = _compute_spatial_jaccard_distribution(neg_spatial_masks)

    return {
        "check": "LIME Repeatability Across Seeds (Descriptive)",
        "status": "evaluable",
        "evaluation_scope": eval_scope,
        "seeds_tested": unique_seeds,
        "positive_support_repeatability": pos_stats,
        "negative_support_repeatability": neg_stats,
        "r2_scores_across_seeds": r2_scores,
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)) if len(r2_scores) > 1 else 0.0,
        "methodological_documentation": (
            "Fixed segmentation isolates surrogate sampling variability, while varying "
            "segmentation introduces both segmentation partition shifts and surrogate sampling variability."
        )
    }


def evaluate_classifier_head_randomization_sanity(
    original_model: tf.keras.Model,
    img_tensor: np.ndarray,
    target_class: int,
    prep_config: PreprocessingConfig = PreprocessingConfig()
) -> Dict[str, Any]:
    """Test 5: Partial Classifier-Head Parameter Randomization Check (Adebayo et al., 2018).

    Scope:
      Partial Classifier-Head Randomization (randomizes only the dense head: `dense_1024` and `output`).
      Note: This is NOT a complete cascading layer-by-layer randomization across the convolutional backbone.

    Safety:
      Makes independent copies of every original model weight before cloning.
      Randomizes ONLY an in-memory clone.
      After testing, performs an exact np.array_equal comparison across all original weights
      to guarantee the original model was completely untouched.
    """
    # 1. Capture exact independent copy of all original weights
    saved_orig_weights = [np.copy(w) for w in original_model.get_weights()]

    # 2. Original Grad-CAM
    feat_orig, head_orig = build_gradcam_submodels(original_model, prep_config)
    orig_cam, _ = compute_gradcam_heatmap(feat_orig, head_orig, img_tensor, target_class=target_class)

    # 3. Create isolated in-memory clone
    randomized_model = tf.keras.models.clone_model(original_model)
    randomized_model.set_weights(saved_orig_weights)

    # 4. Randomize ONLY classifier head layers on the clone
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    for layer_name in [prep_config.penultimate_layer_name, prep_config.output_layer_name]:
        layer = randomized_model.get_layer(layer_name)
        new_weights = [initializer(shape=w.shape).numpy() for w in layer.get_weights()]
        layer.set_weights(new_weights)

    # 5. Exact post-test identity check on the original model
    post_test_orig_weights = original_model.get_weights()
    is_original_untouched = (
        len(saved_orig_weights) == len(post_test_orig_weights)
        and all(np.array_equal(w_pre, w_post) for w_pre, w_post in zip(saved_orig_weights, post_test_orig_weights))
    )

    if not is_original_untouched:
        raise RuntimeError("FATAL SAFETY VIOLATION: Original model weights were mutated during randomization test!")

    feat_rand, head_rand = build_gradcam_submodels(randomized_model, prep_config)
    rand_cam, _ = compute_gradcam_heatmap(feat_rand, head_rand, img_tensor, target_class=target_class)

    flat_orig = orig_cam.flatten()
    flat_rand = rand_cam.flatten()
    rho, _ = spearmanr(flat_orig, flat_rand)
    mean_abs_diff = float(np.mean(np.abs(flat_orig - flat_rand)))

    return {
        "check": "Partial Classifier-Head Randomization (Descriptive)",
        "scope": "Classifier-Head Randomization (dense_1024 and output layers only)",
        "is_original_model_untouched": is_original_untouched,
        "spearman_rho_original_vs_randomized": float(rho) if not np.isnan(rho) else 0.0,
        "mean_absolute_difference": mean_abs_diff,
        "interpretation_guide": (
            "A parameter-sensitive explanation will show diminished correlation with "
            "randomized network weights. High correlation indicates invariance to head parameters."
        )
    }


# ==============================================================================
# Output Safety & Visual Reporting
# ==============================================================================

RESERVED_OUTPUT_FILENAMES = {
    "gradcam_explanation.png",
    "lime_explanation.png",
    "shap_explanation.png",
    "shap_model2.png",
    "shap_signed.png",
    "sample_img.png"
}


def get_unique_output_path(base_name: str, ext: str = "png", output_dir: str = "results_validated") -> str:
    """Generate a unique filepath strictly preventing overwriting of existing assets.

    Ensures destination folder exists and never conflicts with reserved baseline files.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_ext = ext.lstrip(".")

    # If user provided a reserved filename, force a non-conflicting prefix
    if f"{base_name}.{clean_ext}" in RESERVED_OUTPUT_FILENAMES:
        base_name = f"validated_{base_name}"

    candidate = os.path.join(output_dir, f"{base_name}.{clean_ext}")
    if not os.path.exists(candidate):
        return candidate

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_ts = os.path.join(output_dir, f"{base_name}_{timestamp}.{clean_ext}")
    if not os.path.exists(candidate_ts):
        return candidate_ts

    counter = 1
    while True:
        candidate_indexed = os.path.join(output_dir, f"{base_name}_{timestamp}_{counter}.{clean_ext}")
        if not os.path.exists(candidate_indexed):
            return candidate_indexed
        counter += 1


def save_image_artifact(image_array: np.ndarray, base_name: str, output_dir: str = "results_validated") -> str:
    """Save an RGB float [0, 1] or uint8 image to a unique, non-conflicting PNG file."""
    safe_path = get_unique_output_path(base_name, ext="png", output_dir=output_dir)
    if image_array.dtype != np.uint8 and image_array.max() <= 1.0:
        uint8_img = np.uint8(np.clip(image_array * 255.0, 0, 255))
    else:
        uint8_img = np.uint8(np.clip(image_array, 0, 255))

    plt.imsave(safe_path, uint8_img)
    return safe_path


def generate_validation_summary_plot(
    img_array: np.ndarray,
    cam_parasitized: np.ndarray,
    cam_uninfected: np.ndarray,
    lime_pos_vis: np.ndarray,
    lime_neg_vis: np.ndarray,
    lime_comb_vis: np.ndarray,
    p_uninfected: float,
    output_filename: str = "validated_xai_summary.png",
    output_dir: str = "results_validated"
) -> str:
    """Render a comprehensive diagnostic grid showing verified explanations for both classes."""
    safe_output_path = get_unique_output_path(
        base_name=os.path.splitext(output_filename)[0],
        ext="png",
        output_dir=output_dir
    )

    p_parasitized = 1.0 - p_uninfected
    predicted_label = "Parasitized" if p_parasitized >= 0.5 else "Uninfected"
    pred_confidence = max(p_parasitized, p_uninfected)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        f"Verified XAI Suite (Model 1: Xception)\n"
        f"Predicted: {predicted_label} (p={pred_confidence:.4f}) | "
        f"P(Parasitized)={p_parasitized:.4f}, P(Uninfected)={p_uninfected:.4f}",
        fontsize=14,
        fontweight="bold"
    )

    # 1. Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Input Image (RGB [0, 1])")
    axes[0, 0].axis("off")

    # 2. Grad-CAM Parasitized (Target 0: logit = -z)
    overlay_p = create_gradcam_overlay(img_array, cam_parasitized)
    axes[0, 1].imshow(overlay_p)
    axes[0, 1].set_title("Grad-CAM: Target = Parasitized\n(Logit = -z)")
    axes[0, 1].axis("off")

    # 3. Grad-CAM Uninfected (Target 1: logit = +z)
    overlay_u = create_gradcam_overlay(img_array, cam_uninfected)
    axes[0, 2].imshow(overlay_u)
    axes[0, 2].set_title("Grad-CAM: Target = Uninfected\n(Logit = +z)")
    axes[0, 2].axis("off")

    # 4. LIME Positive Support (Green)
    axes[1, 0].imshow(lime_pos_vis)
    axes[1, 0].set_title(f"LIME: Positive Support (Green)\nfor {predicted_label}")
    axes[1, 0].axis("off")

    # 5. LIME Negative Support (Red)
    axes[1, 1].imshow(lime_neg_vis)
    axes[1, 1].set_title(f"LIME: Negative Support (Red)\nagainst {predicted_label}")
    axes[1, 1].axis("off")

    # 6. LIME Combined Bidirectional
    axes[1, 2].imshow(lime_comb_vis)
    axes[1, 2].set_title(f"LIME: Combined Support\n(Green: For, Red: Against)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(safe_output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return safe_output_path


# ==============================================================================
# Controlled Experiment Execution Function
# ==============================================================================

def run_validated_xai_pipeline(
    model_path: str,
    image_path: str,
    output_dir: str = "results_validated",
    target_class: Optional[int] = None,
    prep_config: PreprocessingConfig = PreprocessingConfig(),
    gradcam_config: GradCamConfig = GradCamConfig(),
    lime_config: LimeConfig = LimeConfig(),
    val_config: ValidationConfig = ValidationConfig()
) -> Dict[str, Any]:
    """Execute the full validated XAI pipeline on a single image.

    This function is strictly opt-in and will NOT execute automatically on import.
    It:
      1. Loads the saved model checkpoint without mutating it.
      2. Preprocesses the image matching model1.py training.
      3. Verifies numerical logit extraction.
      4. Validates probability formatting defensively.
      5. Computes verified Grad-CAM for both classes.
      6. Computes verified LIME positive, negative, and combined attributions.
      7. Runs all descriptive validation diagnostics (including LIME repeatability).
      8. Saves individual visualization files and summary grid exclusively into output_dir.

    Args:
        model_path: Path to existing .keras model checkpoint.
        image_path: Path to PNG cell image.
        output_dir: Directory where non-conflicting results will be stored.
        target_class: Optional explicit class (0 or 1). If None, uses predicted class.
        prep_config: Preprocessing settings.
        gradcam_config: Grad-CAM settings.
        lime_config: LIME settings.
        val_config: Validation protocol settings.

    Returns:
        Structured dictionary containing diagnostics, prediction probabilities,
        and paths to all generated visual artifacts.
    """
    print(f"[XAI-Pipeline] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    print(f"[XAI-Pipeline] Preprocessing image from: {image_path}")
    img_tensor = load_and_preprocess_image(image_path, prep_config)
    img_array = img_tensor[0]

    # 1. Decoupled sub-models and logit identity check
    feat_extractor, classifier_head = build_gradcam_submodels(model, prep_config)
    logit_check = verify_logit_extraction(model, feat_extractor, classifier_head, img_tensor)
    print(f"[XAI-Pipeline] Logit extraction verified: max prob discrepancy = {logit_check['max_prob_discrepancy']:.6e}")

    # 2. Defensive probability formatting check
    predict_fn = make_lime_predict_fn(model, prep_config)
    prob_check = validate_probability_formatting(predict_fn, img_tensor)

    # 3. Compute Grad-CAM for both classes
    cam_parasitized, diag_cam_p = compute_gradcam_heatmap(
        feat_extractor, classifier_head, img_tensor, target_class=0, config=gradcam_config
    )
    cam_uninfected, diag_cam_u = compute_gradcam_heatmap(
        feat_extractor, classifier_head, img_tensor, target_class=1, config=gradcam_config
    )

    overlay_cam_p = create_gradcam_overlay(img_array, cam_parasitized, alpha=gradcam_config.alpha, colormap=gradcam_config.colormap)
    overlay_cam_u = create_gradcam_overlay(img_array, cam_uninfected, alpha=gradcam_config.alpha, colormap=gradcam_config.colormap)

    # 4. Grad-CAM Class Contrast Evaluation
    contrast_eval = evaluate_gradcam_class_contrast(
        feat_extractor, classifier_head, img_tensor, val_config
    )

    # 5. Compute LIME Explanations
    p_uninfected = diag_cam_p["p_uninfected"]
    predicted_class = 0 if (1.0 - p_uninfected) >= 0.5 else 1
    chosen_lime_class = target_class if target_class is not None else predicted_class

    lime_explanation, lime_diag = compute_lime_explanation(
        model=model,
        img_array=img_array,
        target_class=chosen_lime_class,
        lime_config=lime_config,
        prep_config=prep_config
    )

    lime_visualizations = extract_lime_visualizations(
        explanation=lime_explanation,
        target_class=chosen_lime_class,
        num_features=lime_config.num_features
    )

    # 6. LIME Local Fidelity Evaluation
    lime_fidelity = evaluate_lime_local_fidelity(lime_diag["surrogate_r2"])

    # 7. LIME Repeatability Evaluation across seeds (both positive and negative masks)
    print("[XAI-Pipeline] Evaluating LIME repeatability across seeds...")
    lime_repeatability = evaluate_lime_repeatability(
        model=model,
        img_array=img_array,
        target_class=chosen_lime_class,
        base_lime_config=lime_config,
        val_config=val_config
    )

    # 8. Partial Classifier-Head Randomization Sanity Check
    rand_sanity = evaluate_classifier_head_randomization_sanity(
        original_model=model,
        img_tensor=img_tensor,
        target_class=chosen_lime_class,
        prep_config=prep_config
    )

    # 9. Save Individual Visualization Files
    path_cam_p = save_image_artifact(overlay_cam_p, "gradcam_target_parasitized", output_dir)
    path_cam_u = save_image_artifact(overlay_cam_u, "gradcam_target_uninfected", output_dir)
    path_lime_pos = save_image_artifact(lime_visualizations["positive_only"], "lime_positive_support", output_dir)
    path_lime_neg = save_image_artifact(lime_visualizations["negative_only"], "lime_negative_support", output_dir)
    path_lime_comb = save_image_artifact(lime_visualizations["combined"], "lime_combined_support", output_dir)

    # 10. Render Grid Summary Plot
    summary_plot_path = generate_validation_summary_plot(
        img_array=img_array,
        cam_parasitized=cam_parasitized,
        cam_uninfected=cam_uninfected,
        lime_pos_vis=lime_visualizations["positive_only"],
        lime_neg_vis=lime_visualizations["negative_only"],
        lime_comb_vis=lime_visualizations["combined"],
        p_uninfected=p_uninfected,
        output_filename="validated_xai_summary.png",
        output_dir=output_dir
    )

    print(f"[XAI-Pipeline] Output artifacts saved exclusively to: {output_dir}")

    return {
        "image_path": image_path,
        "model_path": model_path,
        "output_directory": output_dir,
        "prediction": {
            "predicted_class": predicted_class,
            "class_name": "Parasitized" if predicted_class == 0 else "Uninfected",
            "p_parasitized": 1.0 - p_uninfected,
            "p_uninfected": p_uninfected,
            "raw_logit_z": diag_cam_p["raw_logit_z"]
        },
        "diagnostics": {
            "logit_verification": logit_check,
            "probability_compliance": prob_check,
            "gradcam_class_contrast": contrast_eval,
            "lime_fidelity": lime_fidelity,
            "lime_repeatability": lime_repeatability,
            "classifier_head_randomization_sanity": rand_sanity
        },
        "output_artifacts": {
            "gradcam_parasitized": path_cam_p,
            "gradcam_uninfected": path_cam_u,
            "lime_positive": path_lime_pos,
            "lime_negative": path_lime_neg,
            "lime_combined": path_lime_comb,
            "summary_grid": summary_plot_path
        },
        "scientific_disclaimer": (
            "Mathematical correctness and model faithfulness do NOT imply biological localization. "
            "Localization claims require independent pathologist-annotated ground-truth masks."
        )
    }


# ==============================================================================
# Standalone CLI Entry Point Guard
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validated XAI Suite for Malaria Diagnosis (Model 1: Xception)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an individual blood smear cell image PNG."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_models/model1_stage2_best.keras",
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_validated",
        help="Directory to save non-conflicting visual outputs."
    )
    parser.add_argument(
        "--target_class",
        type=int,
        choices=[0, 1],
        default=None,
        help="Target class to explain (0: Parasitized, 1: Uninfected). Defaults to model prediction."
    )

    args = parser.parse_args()

    if args.image_path is None:
        print("=" * 80)
        print("run_xai_validated.py - Validated XAI Diagnostic Suite")
        print("=" * 80)
        print("Module loaded safely. No automatic operations were performed.")
        print()
        print("To run the validated pipeline on an image, pass --image_path:")
        print("  python run_xai_validated.py --image_path sample_img.png")
        print()
        print("Or import and call from Python:")
        print("  from run_xai_validated import run_validated_xai_pipeline")
        print("  results = run_validated_xai_pipeline(model_path='...', image_path='...')")
        print("=" * 80)
    else:
        results = run_validated_xai_pipeline(
            model_path=args.model_path,
            image_path=args.image_path,
            output_dir=args.output_dir,
            target_class=args.target_class
        )
