
import csv
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("saved_models/model1_stage2_best.keras")
MANIFEST = Path("results_validated/phase2_pilot/image_manifest.csv")
OUT = Path("results_validated/phase2_pilot/gradcam")
LAST_CONV = "block14_sepconv2_act"

if not MODEL_PATH.exists() or not MANIFEST.exists():
    raise FileNotFoundError("Checkpoint or manifest missing.")

if any(OUT.iterdir()):
    raise RuntimeError("Output directory is not empty. Stopping.")

with MANIFEST.open() as f:
    rows = list(csv.DictReader(f))

if len(rows) != 20:
    raise ValueError(f"Expected 20 images, found {len(rows)}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
output_layer = model.get_layer("output")
conv_layer = model.get_layer(LAST_CONV)

grad_model = tf.keras.Model(
    model.input,
    [conv_layer.output, output_layer.input]
)

def generate_cam(gradients, activations):
    if gradients is None:
        raise RuntimeError("Gradient disconnected.")

    g = gradients.numpy()[0]
    a = activations.numpy()[0]

    if not np.isfinite(g).all():
        raise RuntimeError("Non-finite gradients.")

    weights = np.mean(g, axis=(0, 1))
    cam = np.maximum(np.sum(a * weights, axis=-1), 0)

    maximum = float(cam.max())
    if maximum <= 1e-8 or not np.isfinite(maximum):
        raise RuntimeError("Degenerate CAM.")

    cam /= maximum
    return cv2.resize(cam, (224, 224))

def save_cam(cam, base, folder, label):
    gray = np.uint8(cam * 255)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base, 0.6, colored, 0.4, 0)

    Image.fromarray(gray).save(folder / f"{label}_grayscale.png")
    Image.fromarray(colored).save(folder / f"{label}_heatmap.png")
    Image.fromarray(overlay).save(folder / f"{label}_overlay.png")

results = []

for index, row in enumerate(rows, start=1):
    image_path = Path(row["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    with tf.GradientTape(persistent=True) as tape:
        activations, features = grad_model(image, training=False)
        z = tf.matmul(features, output_layer.kernel) + output_layer.bias
        score_p = -tf.reduce_sum(z)
        score_u = tf.reduce_sum(z)

    grad_p = tape.gradient(score_p, activations)
    grad_u = tape.gradient(score_u, activations)
    del tape

    cam_p = generate_cam(grad_p, activations)
    cam_u = generate_cam(grad_u, activations)

    p_uninfected = float(tf.sigmoid(z).numpy()[0, 0])
    p_parasitized = 1.0 - p_uninfected
    true_label = int(row["true_label"])
    predicted_label = 1 if p_uninfected >= 0.5 else 0

    folder = OUT / f"image_{index:02d}"
    folder.mkdir(parents=False)

    original = Image.open(image_path).convert("RGB")
    base = np.array(original.resize((224, 224)))

    original.resize((224, 224)).save(folder / "original_image.png")
    save_cam(cam_p, base, folder, "parasitized")
    save_cam(cam_u, base, folder, "uninfected")

    results.append({
        "index": index,
        "image_path": str(image_path),
        "true_label": true_label,
        "predicted_label": predicted_label,
        "p_parasitized": p_parasitized,
        "p_uninfected": p_uninfected,
        "parasitized_cam_valid": True,
        "uninfected_cam_valid": True
    })

    print(f"[{index}/20] Completed: {image_path.name}")

summary = OUT / "gradcam_summary.csv"
with summary.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nGRAD-CAM BATCH COMPLETED")
print("Images:", len(results))
print("CAMs generated:", len(results) * 2)
print("Summary:", summary)
print("Output:", OUT.resolve())
