from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE = Path("results_validated/phase2_pilot")
OUT = BASE / "faithfulness_deletion"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = Path("saved_models/model1_stage2_best.keras")
MANIFEST = BASE / "image_manifest.csv"
N = 14
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEED = 42

model = tf.keras.models.load_model(CHECKPOINT, compile=False)
manifest = pd.read_csv(MANIFEST)
rng = np.random.default_rng(SEED)

def load_image(path):
    img = tf.io.read_file(str(path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224], method="bilinear")
    return img.numpy().astype(np.float32) / 255.0

def patch_order(score_map):
    scores = []
    for r in range(N):
        for c in range(N):
            patch = score_map[
                r * 16:(r + 1) * 16,
                c * 16:(c + 1) * 16
            ]
            scores.append(float(np.mean(patch)))
    return np.argsort(scores)[::-1]

def mask_patches(image, order, fraction):
    result = image.copy()
    count = int(round(N * N * fraction))
    for idx in order[:count]:
        r, c = divmod(int(idx), N)
        result[r * 16:(r + 1) * 16,
               c * 16:(c + 1) * 16, :] = 0.5
    return result

def predicted_probability(image, predicted_label):
    p_uninfected = float(model(image[None, ...], training=False).numpy().reshape(-1)[0])
    return p_uninfected if predicted_label == 1 else 1.0 - p_uninfected

rows = []

for _, item in manifest.iterrows():
    image_number = int(item.get("image_index", item.get("index", _ + 1)))
    image_path = Path(item["image_path"])
    image_name = image_path.name
    folder = BASE / "gradcam_complete" / f"image_{image_number:02d}"

    image = load_image(image_path)
    p_uninfected = float(model(image[None, ...], training=False).numpy().reshape(-1)[0])
    predicted_label = int(p_uninfected >= 0.5)
    original_prob = p_uninfected if predicted_label == 1 else 1.0 - p_uninfected

    class_name = "uninfected" if predicted_label == 1 else "parasitized"
    grad_path = folder / f"{class_name}_heatmap.png"
    grad = np.asarray(Image.open(grad_path).convert("L"), dtype=np.float32) / 255.0

    if image_number == 1:
        shap_path = BASE / "shap_gradient_diagnostic_ns200/gradient_shap_values_ns200.npy"
    else:
        shap_path = BASE / "shap_gradient_batch_ns200" / f"image_{image_number:02d}" / "shap_values.npy"

    shap_values = np.load(shap_path)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 5:
        shap_values = shap_values[0, ..., 0]
    elif shap_values.ndim == 4:
        shap_values = shap_values[0]
    if shap_values.shape != (224, 224, 3):
        raise ValueError(f"Unexpected SHAP shape for {image_name}: {shap_values.shape}")

    # Model output is P(Uninfected): positive supports label 1,
    # negative supports label 0. Rank only contributions supporting prediction.
    shap_signed = shap_values.mean(axis=-1)
    shap_support = shap_signed if predicted_label == 1 else -shap_signed
    grad_order = patch_order(grad)
    shap_order = patch_order(np.maximum(shap_support, 0))
    random_order = rng.permutation(N * N)

    for method, order in [
        ("Grad-CAM", grad_order),
        ("SHAP", shap_order),
        ("Random", random_order),
    ]:
        for fraction in FRACTIONS:
            masked = mask_patches(image, order, fraction)
            prob = predicted_probability(masked, predicted_label)
            rows.append({
                "image_number": image_number,
                "image_name": image_name,
                "true_class": item.get("class_name", image_path.parent.name),
                "predicted_class": "Uninfected" if predicted_label else "Parasitized",
                "method": method,
                "deleted_fraction": fraction,
                "predicted_class_probability": prob,
                "original_predicted_class_probability": original_prob,
                "probability_drop": original_prob - prob,
            })

    print(f"Completed {image_number}/20: {image_name}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "deletion_results.csv", index=False)

summary = (
    df.groupby(["method", "deleted_fraction"])["probability_drop"]
      .agg(["mean", "std", "count"])
      .reset_index()
)
summary.to_csv(OUT / "deletion_summary.csv", index=False)

print("\n========== DELETION SUMMARY ==========")
print(summary.to_string(index=False))
print("\nSaved:", OUT)
