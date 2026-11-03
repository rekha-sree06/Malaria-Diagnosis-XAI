import csv
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import shap

MODEL_PATH = Path("saved_models/model1_stage2_best.keras")
MANIFEST = Path("results_validated/phase2_pilot/image_manifest.csv")
DATA_DIR = Path.home() / "malaria_phase2_data/cell_images"
OUT = Path("results_validated/phase2_pilot/shap_gradient_pilot")

HEIGHT, WIDTH = 224, 224
NSAMPLES = 50
SEED = 42

OUT.mkdir(parents=True, exist_ok=False)

# Read fixed pilot manifest.
with MANIFEST.open(newline="") as f:
    pilot_rows = list(csv.DictReader(f))

pilot_paths = {str(Path(r["image_path"])) for r in pilot_rows}
sample_path = Path(pilot_rows[0]["image_path"])

def load_image(path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(
        image, [HEIGHT, WIDTH], method="bilinear"
    )
    return (tf.cast(image, tf.float32) / 255.0).numpy()

# Select 5 reference images per class, excluding pilot images.
rng = random.Random(SEED)
background_paths = []

for class_name in ["Parasitized", "Uninfected"]:
    candidates = sorted(
        p for p in (DATA_DIR / class_name).glob("*.png")
        if str(p) not in pilot_paths
    )
    if len(candidates) < 5:
        raise RuntimeError(f"Not enough background images: {class_name}")
    background_paths.extend(rng.sample(candidates, 5))

background = np.stack([load_image(p) for p in background_paths])
sample = load_image(sample_path)[None, ...]

print("Background shape:", background.shape)
print("Explained image:", sample_path.name)
print("Explained image shape:", sample.shape)

print("Loading frozen Xception model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)

prediction = float(model.predict(sample, verbose=0)[0, 0])
print("P(Uninfected):", prediction)
print("P(Parasitized):", 1.0 - prediction)

print("Creating GradientExplainer...")
explainer = shap.GradientExplainer(model, background)

print(f"Computing SHAP values ({NSAMPLES} samples)...")
start = time.time()
values = explainer.shap_values(sample, nsamples=NSAMPLES)
runtime = time.time() - start

if isinstance(values, list):
    print("Number of output arrays:", len(values))
    values = values[0]

values = np.asarray(values)
print("SHAP output shape:", values.shape)
print("SHAP output finite:", np.isfinite(values).all())
print("SHAP runtime:", round(runtime, 2), "seconds")

np.save(OUT / "gradient_shap_values.npy", values)
np.save(OUT / "background_images.npy", background)

print("\nGradientExplainer pilot completed.")
print("Output:", OUT)
