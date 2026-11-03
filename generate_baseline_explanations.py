from pathlib import Path
import pandas as pd

BASE = Path("results_validated/phase2_pilot")
INPUT = BASE / "cross_method_comparison/cross_method_comparison.csv"
OUT = BASE / "explanation_evaluation"

df = pd.read_csv(INPUT)

if not OUT.exists():
    raise FileNotFoundError("Create the evaluation folder first.")

def generate(row):
    predicted = "Uninfected" if row["predicted_label"] == 1 else "Parasitized"
    probability = (
        row["p_uninfected"]
        if row["predicted_label"] == 1
        else row["p_parasitized"]
    )

    cam_status = (
        row["uninfected_cam_status"]
        if row["predicted_label"] == 1
        else row["parasitized_cam_status"]
    )

    lime_r2 = float(row["lime_surrogate_r2"])
    shap_sum = float(row["shap_sum"])

    direction = (
        "toward Uninfected"
        if shap_sum > 0
        else "toward Parasitized"
        if shap_sum < 0
        else "neutral"
    )

    return {
        "image_name": row["image_name"],
        "predicted_class": predicted,
        "predicted_probability": float(probability),
        "explanation": (
            f"The model classified the image as {predicted} "
            f"with confidence {float(probability):.4%}. "
            f"Grad-CAM status: {cam_status}. "
            f"LIME surrogate R²: {lime_r2:.4f}. "
            f"SHAP aggregate attribution: {shap_sum:+.4f}, "
            f"with direction {direction}."
        )
    }

records = [generate(row) for _, row in df.iterrows()]
result = pd.DataFrame(records)

csv_path = OUT / "baseline_explanations.csv"
report_path = OUT / "baseline_explanations.md"

if csv_path.exists() or report_path.exists():
    raise FileExistsError("Baseline output already exists. No files overwritten.")

result.to_csv(csv_path, index=False)

with report_path.open("w") as f:
    f.write("# Conventional Baseline Explanations\n\n")
    f.write("Generated from the same 20-image pilot and measured XAI outputs.\n\n")
    for i, record in enumerate(records, 1):
        f.write(f"## Image {i}: {record['image_name']}\n\n")
        f.write(record["explanation"] + "\n\n---\n\n")

print("Baseline explanations generated:", len(result))
print("Saved:", csv_path)
print("Saved:", report_path)
