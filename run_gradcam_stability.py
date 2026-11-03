from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

BASE = Path("results_validated/phase2_pilot")
OUT = BASE / "stability_gradcam_pilot"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = Path("saved_models/model1_stage2_best.keras")
MANIFEST = pd.read_csv(BASE / "image_manifest.csv")
SELECTED = [1, 6, 11, 16, 20]

model = tf.keras.models.load_model(CHECKPOINT, compile=False)
conv_layer = model.get_layer("block14_sepconv2_act")
output_layer = model.layers[-1]

grad_model = tf.keras.Model(
    model.inputs,
    [conv_layer.output, output_layer.input]
)

def preprocess(pil_image):
    arr = np.asarray(pil_image.convert("RGB"))
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    tensor = tf.image.resize(tensor, [224, 224], method="bilinear")
    return tf.expand_dims(tensor / 255.0, 0)

def get_cam(pil_image, predicted_label):
    x = preprocess(pil_image)

    with tf.GradientTape() as tape:
        activations, features = grad_model(x, training=False)
        z = tf.matmul(features, output_layer.kernel) + output_layer.bias
        score = tf.reduce_sum(z if predicted_label == 1 else -z)

    grads = tape.gradient(score, activations)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * activations, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    cam = tf.image.resize(cam[..., None], [224, 224]).numpy()[..., 0]

    maximum = float(cam.max())
    if maximum <= 1e-12:
        return None
    return cam / maximum

rows = []

for image_number in SELECTED:
    row = MANIFEST.iloc[image_number - 1]
    image_path = Path(row["image_path"])
    original = Image.open(image_path).convert("RGB")

    original_tensor = preprocess(original)
    p_uninfected = float(model(original_tensor, training=False).numpy()[0, 0])
    predicted_label = int(p_uninfected >= 0.5)
    class_name = "uninfected" if predicted_label else "parasitized"

    variants = {
        "original": original,
        "brightness_plus10": ImageEnhance.Brightness(original).enhance(1.10),
        "contrast_plus10": ImageEnhance.Contrast(original).enhance(1.10),
    }

    original_cam = get_cam(original, predicted_label)
    if original_cam is None:
        print(f"Skipping image {image_number}: original CAM degenerate")
        continue

    image_dir = OUT / f"image_{image_number:02d}"
    image_dir.mkdir()

    for variant_name, variant_image in variants.items():
        cam = get_cam(variant_image, predicted_label)

        if cam is None:
            rows.append({
                "image_number": image_number,
                "image_name": image_path.name,
                "variant": variant_name,
                "predicted_class": class_name,
                "ssim": np.nan,
                "spearman": np.nan,
                "status": "degenerate_cam"
            })
            continue

        Image.fromarray(np.uint8(cam * 255)).save(
            image_dir / f"{variant_name}_heatmap.png"
        )

        if variant_name == "original":
            ssim, rho = 1.0, 1.0
        else:
            ssim = structural_similarity(
                original_cam, cam, data_range=1.0
            )
            rho = spearmanr(
                original_cam.ravel(), cam.ravel()
            ).statistic

        rows.append({
            "image_number": image_number,
            "image_name": image_path.name,
            "variant": variant_name,
            "predicted_class": class_name,
            "ssim": float(ssim),
            "spearman": float(rho),
            "status": "success"
        })

    print(f"Completed image {image_number}: {image_path.name}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "stability_results.csv", index=False)

print("\n========== STABILITY SUMMARY ==========")
print(
    df[df["variant"] != "original"]
    .groupby("variant")[["ssim", "spearman"]]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)
print("\nSaved:", OUT)
