import os
import gc
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import shap

DATA_DIR = "data/cell_images"
MODEL_PATH = "saved_models/model2.keras"
OUTPUT_PATH = "shap_model2.png"

SEED = 42
N_BACKGROUND = 50          # paper: n = 50 random images
ALPHA = 0.5                # display settings (same as model 1)
BASE, SPAN, SIGMA = 0.11, 0.12, 1.5

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("Loading saved Model 2...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded. No training will be performed.")

classes = sorted(
    name for name in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, name))
)

def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    return tf.keras.preprocessing.image.img_to_array(img) / 255.0

# --- explained image: same choice as before (2nd Parasitized file) ---
sample_folder = os.path.join(DATA_DIR, classes[0])
sample_files = sorted(f for f in os.listdir(sample_folder) if f.lower().endswith(".png"))
sample_path = os.path.join(sample_folder, sample_files[1])
sample = load_image(sample_path)[None, ...].astype(np.float32)

# --- background: 50 random images from the dataset, excluding the sample ---
all_paths = []
for c in classes:
    folder = os.path.join(DATA_DIR, c)
    all_paths += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.lower().endswith(".png")]
candidates = [p for p in all_paths if p != sample_path]
bg_paths = random.Random(SEED).sample(candidates, N_BACKGROUND)
background = np.stack([load_image(p) for p in bg_paths]).astype(np.float32)

# --- prediction (sigmoid output = P(classes[1])) ---
p_class1 = float(model.predict(sample, verbose=0)[0][0])
pred_idx = int(p_class1 >= 0.5)
p_pred = p_class1 if pred_idx == 1 else 1.0 - p_class1
true_name, pred_name = classes[0], classes[pred_idx]
correct = (true_name == pred_name)
tag = ("TP" if true_name == "Parasitized" else "TN") if correct else \
      ("FN" if true_name == "Parasitized" else "FP")

print("Explained image:", sample_path)
print(f"P({classes[1]}) = {p_class1:.4f} -> predicted {pred_name} (p={p_pred:.2f}), {tag}")
print("Generating SHAP values with GradientExplainer...")

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(sample)   # default nsamples (200)

if isinstance(shap_values, list):
    shap_values = shap_values[0]
values = np.asarray(shap_values)
if values.ndim == 5 and values.shape[-1] == 1:   # SHAP 0.51: (1,224,224,3,1)
    values = values[..., 0]
if values.shape != (1, 224, 224, 3):
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")
print("SHAP values shape:", values.shape)

# --- paper-style map: absolute mean over RGB, normalize ---
shap_map = np.mean(np.abs(values[0]), axis=-1)
shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-12)

# display-only: light smoothing + restricted colormap range (as in model 1)
smooth_map = gaussian_filter(shap_map, sigma=SIGMA)
smooth_map = smooth_map / (smooth_map.max() + 1e-12)
heat = cm.jet(BASE + SPAN * smooth_map)[..., :3]
overlay = np.clip((1 - ALPHA) * sample[0] + ALPHA * heat, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample[0]); axes[0].axis("off")
axes[0].set_title(f"Original ({tag})\nClass: {pred_name} (p={p_pred:.2f})")
axes[1].imshow(overlay); axes[1].axis("off")
axes[1].set_title("SHAP (GradientExplainer)")
plt.tight_layout()
try:
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
except OSError:
    fallback = os.path.expanduser("~/shap_model2.png")
    plt.savefig(fallback, dpi=200, bbox_inches="tight")
    OUTPUT_PATH = fallback
plt.close(fig)

print("SHAP image saved to:", OUTPUT_PATH)

del explainer, shap_values, model
gc.collect()
