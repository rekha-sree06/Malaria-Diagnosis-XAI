from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("results_validated/phase2_pilot")
INPUT = BASE / "cross_method_comparison/cross_method_comparison.csv"
OUT = BASE / "evidence_gated_explanations_v3"

if OUT.exists():
    raise FileExistsError(f"Output folder already exists: {OUT}")

df = pd.read_csv(INPUT)
OUT.mkdir(parents=True)

def make_explanation(row):
    predicted = "Uninfected" if row["predicted_label"] == 1 else "Parasitized"
    probability = float(row["p_uninfected"] if row["predicted_label"] == 1
                        else row["p_parasitized"])

    # Select the Grad-CAM status for the predicted class.
    cam_status = (
        row["uninfected_cam_status"]
        if row["predicted_label"] == 1
        else row["parasitized_cam_status"]
    )

    cam_available = cam_status in ("valid", "valid_previous_run")

    if cam_available:
        if cam_status == "valid_previous_run":
            cam_text = (
                "A Grad-CAM map from a previous run is available for the "
                "predicted class. It indicates image regions that influenced "
                "the model's prediction, but was not independently regenerated "
                "in the consolidated batch. It does not establish parasite location."
            )
        else:
            cam_text = (
                "A Grad-CAM map was generated successfully for the predicted "
                "class. It indicates image regions that influenced the model's "
                "prediction; it does not establish parasite location."
            )
    else:
        cam_text = (
            "The predicted-class Grad-CAM map was degenerate, so it is "
            "excluded from this explanation."
        )

    # LIME: report the measured local surrogate fit without inventing a threshold.
    lime_r2 = float(row["lime_surrogate_r2"])
    lime_positive = float(row["lime_positive_area_fraction"])
    lime_negative = float(row["lime_negative_area_fraction"])

    lime_text = (
        f"LIME produced a local surrogate R² of {lime_r2:.3f}. "
        f"Positive-contribution superpixels cover {lime_positive:.1%} "
        f"of the image, while negative-contribution superpixels cover "
        f"{lime_negative:.1%}. The R² describes local surrogate fit, "
        f"not biological correctness."
    )

    # SHAP sum is oriented to the model output P(Uninfected).
    shap_sum = float(row["shap_sum"])
    residual = float(row["additivity_residual"])
    shap_status = str(row["shap_status"])

    if shap_status.startswith("success"):
        if shap_sum > 0:
            shap_direction = "toward the Uninfected output"
        elif shap_sum < 0:
            shap_direction = "toward the Parasitized output"
        else:
            shap_direction = "with no net direction"

        shap_text = (
            f"The sum of SHAP attributions across the image is {shap_sum:+.4f}, "
            f"indicating a net direction {shap_direction} for the model output "
            f"P(Uninfected). This aggregate does not describe individual pixel "
            f"contributions. The additivity residual is {residual:+.4f}. "
        )
        if abs(residual) > 0.05:
            shap_text += (
                "The absolute additivity residual exceeds the provisional "
                "0.05 reporting threshold. Therefore, the aggregate SHAP "
                "direction is provisional and must not be treated as confirmed "
                "directional evidence for the model output. The 0.05 threshold "
                "is a project-specific reporting heuristic, not a universally "
                "validated cutoff."
            )
        else:
            shap_text += (
                "The residual is within the provisional 0.05 reporting threshold. "
                "This does not establish exact additivity, biological correctness, "
                "or clinical validity. The 0.05 threshold is a project-specific "
                "reporting heuristic, not a universally validated cutoff."
            )
    else:
        shap_text = "SHAP evidence is unavailable or unsuccessful."

    limitations = [
        "Visual attribution does not prove parasite presence or localization.",
        "No independent expert localization masks were available.",
        "LIME R² measures surrogate fit, not biological correctness.",
        "The pilot contains only 20 exploratory images."
    ]

    if not cam_available:
        limitations.append("Predicted-class Grad-CAM was excluded.")
    if cam_status == "valid_previous_run":
        limitations.append(
            "Grad-CAM provenance includes maps retained from a previous run."
        )

    explanation = (
        f"The model predicted {predicted} with a probability of "
        f"{probability:.4%} for that class.\n\n"
        f"Grad-CAM: {cam_text}\n\n"
        f"LIME: {lime_text}\n\n"
        f"SHAP: {shap_text}\n\n"
        f"Limitations: {' '.join(limitations)}"
    )

    return {
        "image_name": row["image_name"],
        "true_class": row["class_name"],
        "predicted_class": predicted,
        "predicted_class_probability": probability,
        "gradcam_status": cam_status,
        "gradcam_included": cam_available,
        "lime_r2": lime_r2,
        "lime_positive_area_fraction": lime_positive,
        "lime_negative_area_fraction": lime_negative,
        "shap_sum": shap_sum,
        "shap_additivity_residual": residual,
        "shap_status": shap_status,
        "explanation": explanation
    }

records = [make_explanation(row) for _, row in df.iterrows()]
result = pd.DataFrame(records)

result.to_csv(OUT / "evidence_explanations.csv", index=False)

with (OUT / "explanations_report.md").open("w") as f:
    f.write("# Evidence-Gated Explanations for Malaria Detection\n\n")
    f.write(
        "These explanations describe model behaviour, not clinical diagnoses "
        "or verified parasite localization.\n\n"
    )
    for i, record in enumerate(records, start=1):
        f.write(f"## Image {i}: {record['image_name']}\n\n")
        f.write(record["explanation"])
        f.write("\n\n---\n\n")

summary = (
    f"Images processed: {len(result)}\n"
    f"Grad-CAM included: {int(result['gradcam_included'].sum())}\n"
    f"Grad-CAM excluded: {int((~result['gradcam_included']).sum())}\n"
    f"SHAP successful: {int(result['shap_status'].str.startswith('success').sum())}\n"
    f"Mean LIME R²: {result['lime_r2'].mean():.4f}\n"
    f"Localization verified: No\n"
    f"Independent expert masks available: No\n"
)
(OUT / "evidence_summary.txt").write_text(summary)

print(summary)
print("Saved outputs to:", OUT)
