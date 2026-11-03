import csv
import gc
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from matplotlib import cm

BASE = Path("results_validated/phase2_pilot")
SHAP_DIR = BASE / "shap_gradient_pilot"
MANIFEST = BASE / "image_manifest.csv"

MODEL_PATH = Path("saved_models/model1_stage2_best.keras")
OUT = BASE / "shap_gradient_batch_ns200"

NSAMPLES = 200
ALPHA = 0.5
BASE_COLOR = 0.11
SPAN = 0.12
SIGMA = 1.5

HEIGHT, WIDTH = 224, 224

# Create a completely new output directory.
OUT.mkdir(parents=True, exist_ok=False)

def load_image(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(
        image, [HEIGHT, WIDTH], method="bilinear"
    )
    return (tf.cast(image, tf.float32) / 255.0).numpy()

# Read fixed pilot manifest.
with MANIFEST.open(newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 20:
    raise RuntimeError(f"Expected 20 manifest images, found {len(rows)}")

# First image already has a separate 200-sample SHAP pilot.
remaining_rows = rows[1:]

# Reuse the exact existing background images.
background = np.load(SHAP_DIR / "background_images.npy")

print("Background shape:", background.shape)
print("Images to process:", len(remaining_rows))

# Load frozen model.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

background_predictions = model.predict(
    background, verbose=0
).reshape(-1)

baseline = float(background_predictions.mean())

print("Mean background P(Uninfected):", baseline)
print("Loading GradientExplainer...")

explainer = shap.GradientExplainer(model, background)

summary_path = OUT / "shap_batch_summary.csv"

columns = [
    "image_number",
    "image_name",
    "true_class",
    "predicted_class",
    "p_uninfected",
    "p_parasitized",
    "correct",
    "shap_sum",
    "prediction_delta",
    "additivity_residual",
    "runtime_seconds",
    "status"
]

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for index, row in enumerate(remaining_rows, start=2):
        image_path = Path(row["image_path"])
        image_name = image_path.name
        true_class = image_path.parent.name

        print(f"\n[{index}/20] Processing {image_name}")

        image_out = OUT / f"image_{index:02d}"
        image_out.mkdir(parents=True, exist_ok=False)

        try:
            sample = load_image(image_path)[None, ...]

            prediction = float(
                model.predict(sample, verbose=0)[0, 0]
            )

            predicted_class = (
                "Uninfected" if prediction >= 0.5
                else "Parasitized"
            )

            # Explain the model's sigmoid output P(Uninfected).
            start = time.time()
            values = explainer.shap_values(
                sample, nsamples=NSAMPLES
            )
            runtime = time.time() - start

            if isinstance(values, list):
                values = values[0]

            values = np.asarray(values)

            if values.shape != (1, 224, 224, 3, 1):
                raise ValueError(
                    f"Unexpected SHAP shape: {values.shape}"
                )

            if not np.isfinite(values).all():
                raise ValueError("Non-finite SHAP values")

            # Save raw explanation and original image.
            np.save(
                image_out / "shap_values.npy", values
            )
            np.save(
                image_out / "original_image.npy", sample
            )

            # Model 2-style absolute attribution visualization.
            shap_map = np.mean(
                np.abs(values[0, :, :, :, 0]), axis=-1
            )

            shap_map = (
                (shap_map - shap_map.min()) /
                (shap_map.max() - shap_map.min() + 1e-12)
            )

            smooth_map = gaussian_filter(
                shap_map, sigma=SIGMA
            )
            smooth_map /= smooth_map.max() + 1e-12

            heat = cm.jet(
                BASE_COLOR + SPAN * smooth_map
            )[..., :3]

            overlay = np.clip(
                (1 - ALPHA) * sample[0] + ALPHA * heat,
                0, 1
            )

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            axes[0].imshow(sample[0])
            axes[0].axis("off")
            axes[0].set_title(
                f"Original\nTrue: {true_class}"
            )

            axes[1].imshow(overlay)
            axes[1].axis("off")
            axes[1].set_title(
                f"SHAP\nPred: {predicted_class}"
            )

            plt.tight_layout()
            plt.savefig(
                image_out / "shap_visualization.png",
                dpi=200,
                bbox_inches="tight"
            )
            plt.close(fig)

            shap_sum = float(
                values[0, :, :, :, 0].sum()
            )

            delta = prediction - baseline

            result = {
                "image_number": index,
                "image_name": image_name,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "p_uninfected": prediction,
                "p_parasitized": 1 - prediction,
                "correct": true_class == predicted_class,
                "shap_sum": shap_sum,
                "prediction_delta": delta,
                "additivity_residual": delta - shap_sum,
                "runtime_seconds": round(runtime, 2),
                "status": "success"
            }

            writer.writerow(result)
            f.flush()

            print("Prediction:", predicted_class)
            print("P(Uninfected):", prediction)
            print("SHAP sum:", shap_sum)
            print("Residual:", delta - shap_sum)
            print("Runtime:", round(runtime, 2), "seconds")

        except Exception as error:
            plt.close("all")
            writer.writerow({
                "image_number": index,
                "image_name": image_name,
                "true_class": true_class,
                "status": f"failed: {error}"
            })
            f.flush()
            print("FAILED:", error)

        gc.collect()

    print("\n========== BATCH SUMMARY ==========")

print("Completed rows:", len(remaining_rows))
print("Summary:", summary_path)
print("Output directory:", OUT)
