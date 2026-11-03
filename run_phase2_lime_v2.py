import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import quickshift
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("saved_models/model1_stage2_best.keras")
MANIFEST_PATH = Path("results_validated/phase2_pilot/image_manifest.csv")
OUTPUT_DIR = Path("results_validated/phase2_pilot/lime_complete")

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_SAMPLES = 1000
NUM_FEATURES = 8
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "Parasitized",
    1: "Uninfected"
}


# ============================================================
# SAFETY CHECKS
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(MANIFEST_PATH)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}. "
        "Refusing to overwrite."
    )

with MANIFEST_PATH.open(newline="") as f:
    manifest = list(csv.DictReader(f))

if len(manifest) != 20:
    raise ValueError(f"Expected 20 manifest images, found {len(manifest)}")

for row in manifest:
    if not Path(row["image_path"]).is_file():
        raise FileNotFoundError(row["image_path"])


# ============================================================
# IMAGE PREPROCESSING — MATCH model1.py
# ============================================================

def load_image(path):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_png(raw, channels=3)
    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH],
        method="bilinear"
    )
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


# ============================================================
# MODEL AND LIME PREDICTION FUNCTION
# ============================================================

print("Loading frozen Xception checkpoint...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

if tuple(model.input_shape[1:]) != (224, 224, 3):
    raise ValueError(f"Unexpected model input shape: {model.input_shape}")

if model.output_shape[-1] != 1:
    raise ValueError(f"Unexpected model output shape: {model.output_shape}")

print("Model loaded:", model.name)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


def predict_fn(images):
    batch = np.asarray(images, dtype=np.float32)

    if batch.ndim != 4 or batch.shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected LIME input shape: {batch.shape}")

    if not np.isfinite(batch).all():
        raise ValueError("LIME input contains NaN or infinity")

    if batch.max() > 1.0:
        batch = batch / 255.0

    batch = np.clip(batch, 0.0, 1.0)

    p_uninfected = model(batch, training=False).numpy().reshape(-1)
    p_parasitized = 1.0 - p_uninfected

    return np.column_stack([p_parasitized, p_uninfected])


# ============================================================
# LIME VISUALIZATION EXTRACTION
# ============================================================

def save_rgb(array, path):
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


def extract_visualizations(explanation, target_class):
    exp = explanation.local_exp[target_class]
    segments = explanation.segments
    image = np.clip(explanation.image.astype(np.float32), 0.0, 1.0)

    positive = sorted(
        [(int(i), float(w)) for i, w in exp if w > 0],
        key=lambda x: x[1],
        reverse=True
    )[:NUM_FEATURES]

    negative = sorted(
        [(int(i), float(w)) for i, w in exp if w < 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:NUM_FEATURES]

    pos_mask = np.isin(segments, [i for i, _ in positive])
    neg_mask = np.isin(segments, [i for i, _ in negative])

    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.4

    pos_vis = image.copy()
    neg_vis = image.copy()
    combined = image.copy()

    pos_vis[pos_mask] = (
        0.6 * pos_vis[pos_mask] + alpha * green
    )
    neg_vis[neg_mask] = (
        0.6 * neg_vis[neg_mask] + alpha * red
    )

    combined[pos_mask] = (
        0.6 * combined[pos_mask] + alpha * green
    )
    combined[neg_mask] = (
        0.6 * combined[neg_mask] + alpha * red
    )

    return {
        "positive": np.clip(pos_vis, 0, 1),
        "negative": np.clip(neg_vis, 0, 1),
        "combined": np.clip(combined, 0, 1),
        "positive_features": positive,
        "negative_features": negative,
        "positive_area_fraction": float(pos_mask.mean()),
        "negative_area_fraction": float(neg_mask.mean())
    }


# ============================================================
# RUN PHASE 2 LIME
# ============================================================

OUTPUT_DIR.mkdir(parents=True)

explainer = lime_image.LimeImageExplainer(
    random_state=RANDOM_SEED
)

summary_rows = []

for image_index, row in enumerate(manifest, start=1):
    image_path = Path(row["image_path"])
    image = load_image(image_path)

    # Generate ONE segmentation and reuse it for both classes.
    segments = quickshift(
        image,
        kernel_size=2.0,
        max_dist=10,
        ratio=0.2,
        rng=RANDOM_SEED,
        channel_axis=-1
    )

    segment_count = int(len(np.unique(segments)))

    if segment_count < 2:
        raise RuntimeError(
            f"Insufficient superpixels for {image_path.name}"
        )

    probabilities = predict_fn(image[np.newaxis, ...])[0]
    predicted_class = int(np.argmax(probabilities))

    image_dir = OUTPUT_DIR / f"image_{image_index:02d}"
    image_dir.mkdir()

    save_rgb(image, image_dir / "original_image.png")

    print(
        f"\n[{image_index}/20] {image_path.name} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | "
        f"Superpixels: {segment_count}"
    )

    for target_class in (0, 1):
        class_name = CLASS_NAMES[target_class]
        class_dir = image_dir / f"class_{target_class}_{class_name.lower()}"
        class_dir.mkdir()

        # Fixed segmentation isolates surrogate sampling variability.
        fixed_segmenter = lambda img, s=segments: s

        start_time = time.perf_counter()

        explanation = explainer.explain_instance(
            image=image,
            classifier_fn=predict_fn,
            labels=(target_class,),
            hide_color=tuple(np.mean(image, axis=(0, 1))),
            num_samples=NUM_SAMPLES,
            segmentation_fn=fixed_segmenter,
            random_seed=RANDOM_SEED
        )

        runtime = time.perf_counter() - start_time

        visual = extract_visualizations(explanation, target_class)

        save_rgb(
            visual["positive"],
            class_dir / "lime_positive.png"
        )
        save_rgb(
            visual["negative"],
            class_dir / "lime_negative.png"
        )
        save_rgb(
            visual["combined"],
            class_dir / "lime_combined.png"
        )

        score_value = explanation.score
        intercept_value = explanation.intercept

        surrogate_r2 = float(
            score_value.get(target_class, float("nan"))
            if isinstance(score_value, dict)
            else score_value
        )
        intercept = float(
            intercept_value.get(target_class, float("nan"))
            if isinstance(intercept_value, dict)
            else intercept_value
        )

        summary_rows.append({
            "image_index": image_index,
            "image_path": str(image_path),
            "true_label": row["true_label"],
            "predicted_label": predicted_class,
            "predicted_class_name": CLASS_NAMES[predicted_class],
            "explained_class": target_class,
            "explained_class_name": class_name,
            "p_parasitized": float(probabilities[0]),
            "p_uninfected": float(probabilities[1]),
            "surrogate_r2": surrogate_r2,
            "surrogate_intercept": intercept,
            "runtime_seconds": runtime,
            "num_samples": NUM_SAMPLES,
            "random_seed": RANDOM_SEED,
            "segmentation": "quickshift",
            "num_superpixels": segment_count,
            "positive_features": json.dumps(visual["positive_features"]),
            "negative_features": json.dumps(visual["negative_features"]),
            "positive_area_fraction": visual["positive_area_fraction"],
            "negative_area_fraction": visual["negative_area_fraction"]
        })

        print(
            f"  {class_name}: R2={surrogate_r2:.4f}, "
            f"runtime={runtime:.1f}s"
        )

# Write summary only after all explanations complete.
summary_path = OUTPUT_DIR / "lime_summary.csv"

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("\n====================================")
print("PHASE 2 LIME COMPLETED")
print("Images:", len(manifest))
print("Explanations:", len(summary_rows))
print("Summary:", summary_path)
print("Output:", OUTPUT_DIR.resolve())
print("====================================")
