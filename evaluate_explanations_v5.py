from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE = Path("results_validated/phase2_pilot")
EVAL = BASE / "explanation_evaluation"

SOURCE = BASE / "cross_method_comparison/cross_method_comparison.csv"
BASELINE = EVAL / "baseline_explanations.csv"
PROPOSED = BASE / "evidence_gated_explanations_v3/evidence_explanations.csv"

DETAIL_OUT = EVAL / "automated_evaluation_details_v5.csv"
SUMMARY_OUT = EVAL / "automated_evaluation_summary_v5.csv"

for path in [SOURCE, BASELINE, PROPOSED]:
    if not path.exists():
        raise FileNotFoundError(path)

source = pd.read_csv(SOURCE)
baseline = pd.read_csv(BASELINE)
proposed = pd.read_csv(PROPOSED)

# Join by image name and verify that all three files contain the same images.
if not (
    set(source.image_name) == set(baseline.image_name)
    == set(proposed.image_name)
):
    raise ValueError("Image names do not match across the three CSV files.")

source = source.set_index("image_name")
baseline = baseline.set_index("image_name")
proposed = proposed.set_index("image_name")

def number_after(pattern, text, flags=re.I):
    match = re.search(pattern, str(text), flags)
    return float(match.group(1)) if match else None

def close(actual, expected, tolerance):
    return (
        actual is not None
        and pd.notna(expected)
        and abs(actual - float(expected)) <= tolerance
    )

def predicted_cam_status(row):
    # predicted_label: 0 = Parasitized, 1 = Uninfected
    if int(row["predicted_label"]) == 0:
        return str(row["parasitized_cam_status"])
    return str(row["uninfected_cam_status"])

def cam_status_covered(text, expected):
    text = str(text).lower()
    expected = str(expected).lower()

    if expected == "valid_previous_run":
        return (
            "valid_previous_run" in text
            or "previous run" in text
        )
    if expected == "valid":
        return (
            "valid" in text
            or "generated successfully" in text
        )
    if expected in {"invalid", "degenerate", "failed"}:
        return any(word in text for word in [
            "invalid", "degenerate", "failed", "excluded"
        ])
    return expected in text

def shap_qualification_covered(text, residual):
    lower = str(text).lower()

    if abs(float(residual)) > 0.05:
        return (
            "provisional" in lower
            and "must not be treated as confirmed" in lower
        )

    return (
        "does not establish exact additivity" in lower
        or "does not establish exact additivity" in lower
    )


def evaluate_one(image, approach, explanation, predicted_class,
                 predicted_probability):
    src = source.loc[image]
    text = str(explanation)

    # Method-specific sections prevent unrelated numbers from being
    # accidentally counted as evidence for another method.
    lime_match = re.search(
        r"LIME\s*:(.*?)(?=\n\s*(?:SHAP|Limitations)\s*:|\Z)",
        text, re.I | re.S
    )
    shap_match = re.search(
        r"SHAP\s*:(.*?)(?=\n\s*Limitations\s*:|\Z)",
        text, re.I | re.S
    )

    lime_text = lime_match.group(1) if lime_match else text
    shap_text = shap_match.group(1) if shap_match else text

    # Baseline explanations are single paragraphs without section headings.
    # Restrict matching to the named metric rather than scanning all numbers.
    if not lime_match:
        lime_r2 = number_after(
            r"LIME\s+surrogate\s+R²\s*:\s*([-+]?\d*\.?\d+)", text
        )
        lime_pos = None
        lime_neg = None
    else:
        lime_r2 = number_after(
            r"surrogate\s+R²\s+of\s+([-+]?\d*\.?\d+)", lime_text
        )
        lime_pos = number_after(
            r"Positive-contribution superpixels cover\s+"
            r"([-+]?\d*\.?\d+)\s*%", lime_text
        )
        lime_neg = number_after(
            r"negative-contribution superpixels cover\s+"
            r"([-+]?\d*\.?\d+)\s*%", lime_text
        )

    if not shap_match:
        shap_sum = number_after(
            r"SHAP aggregate attribution\s*:\s*([-+]?\d*\.?\d+)",
            text
        )
        shap_residual = None
    else:
        shap_sum = number_after(
            r"sum of SHAP attributions.*?\bis\s+"
            r"([-+]?\d*\.?\d+)", shap_text
        )
        shap_residual = number_after(
            r"additivity residual is\s+([-+]?\d*\.?\d+)",
            shap_text
        )

    # Probability is reported as a percentage in both explanation formats.
    probability_percent = number_after(
        r"(?:confidence|probability)\s+(?:of\s+)?"
        r"([-+]?\d*\.?\d+)\s*%", text
    )
    probability_value = (
        probability_percent / 100
        if probability_percent is not None else None
    )

    expected_class = str(src["class_name"])
    expected_cam = predicted_cam_status(src)

    checks = {
        "predicted_class": (
            str(predicted_class).strip().lower()
            == expected_class.strip().lower()
        ),
        "predicted_probability": close(
            probability_value, predicted_probability, 0.00001
        ),
        "gradcam_status_provenance": cam_status_covered(
            text, expected_cam
        ),
        "lime_r2": close(
            lime_r2, src["lime_surrogate_r2"], 0.0011
        ),
        "lime_positive_area": close(
            lime_pos / 100 if lime_pos is not None else None,
            src["lime_positive_area_fraction"], 0.0006
        ),
        "lime_negative_area": close(
            lime_neg / 100 if lime_neg is not None else None,
            src["lime_negative_area_fraction"], 0.0006
        ),
        "shap_sum": close(
            shap_sum, src["shap_sum"], 0.00006
        ),
        "shap_additivity_residual": close(
            shap_residual, src["additivity_residual"], 0.00006
        ),
        "shap_residual_qualification": shap_qualification_covered(
            text, src["additivity_residual"]
        ),
    }

    lower = text.lower()

    # Same five disclosure criteria are applied to both approaches.
    limitations = {
        "localization_boundary": any(phrase in lower for phrase in [
            "does not establish parasite location",
            "does not prove parasite presence or localization",
            "does not establish parasite presence or localization",
            "does not prove parasite location",
        ]),
        "expert_masks_unavailable": any(phrase in lower for phrase in [
            "no independent expert localization masks were available",
            "expert localization masks were not available",
            "no expert localization masks were available",
        ]),
        "lime_r2_boundary": (
            "lime" in lower
            and "biological correctness" in lower
            and ("surrogate fit" in lower or "surrogate r²" in lower)
        ),
        "pilot_scope": (
            "20" in lower
            and ("pilot" in lower or "exploratory images" in lower)
        ),
        "gradcam_provenance_disclosure": cam_status_covered(
            text, expected_cam
        ),
    }

    row = {
        "image_name": image,
        "approach": approach,
        "evidence_items_covered": sum(checks.values()),
        "applicable_evidence_items": len(checks),
        "evidence_coverage": sum(checks.values()) / len(checks),
        "limitations_disclosed": sum(limitations.values()),
        "applicable_limitations": len(limitations),
        "limitation_coverage": (
            sum(limitations.values()) / len(limitations)
        ),
    }

    row.update({f"evidence_{k}": v for k, v in checks.items()})
    row.update({f"limitation_{k}": v for k, v in limitations.items()})
    return row

records = []

for image in sorted(source.index):
    b = baseline.loc[image]
    p = proposed.loc[image]

    records.append(evaluate_one(
        image=image,
        approach="Baseline",
        explanation=b["explanation"],
        predicted_class=b["predicted_class"],
        predicted_probability=b["predicted_probability"],
    ))

    records.append(evaluate_one(
        image=image,
        approach="Evidence-gated",
        explanation=p["explanation"],
        predicted_class=p["predicted_class"],
        predicted_probability=p["predicted_class_probability"],
    ))

details = pd.DataFrame(records)

summary = (
    details.groupby("approach")
    .agg(
        images=("image_name", "nunique"),
        mean_evidence_coverage=("evidence_coverage", "mean"),
        mean_limitation_coverage=("limitation_coverage", "mean"),
        mean_evidence_items=("evidence_items_covered", "mean"),
        mean_applicable_evidence=("applicable_evidence_items", "mean"),
        mean_limitations_disclosed=("limitations_disclosed", "mean"),
        mean_applicable_limitations=("applicable_limitations", "mean"),
    )
    .reset_index()
)

details.to_csv(DETAIL_OUT, index=False)
summary.to_csv(SUMMARY_OUT, index=False)

print("\n===== FAIR EVALUATION V5 =====")
print(summary.to_string(index=False))

print("\n===== EVIDENCE CHECKS BY APPROACH =====")
evidence_cols = [
    c for c in details.columns if c.startswith("evidence_")
]
print(
    details.groupby("approach")[evidence_cols]
    .mean()
    .T.to_string()
)

print("\n===== LIMITATION CHECKS BY APPROACH =====")
limitation_cols = [
    c for c in details.columns if c.startswith("limitation_")
]
print(
    details.groupby("approach")[limitation_cols]
    .mean()
    .T.to_string()
)

print("\nSaved:")
print(DETAIL_OUT)
print(SUMMARY_OUT)
print("\nNote: This is automated text/field coverage, not expert factuality,")
print("clinical validity, explanation faithfulness, or human evaluation.")
