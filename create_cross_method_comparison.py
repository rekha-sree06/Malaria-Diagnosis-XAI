from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("results_validated/phase2_pilot")
OUT = BASE / "cross_method_comparison"
OUT.mkdir(parents=True, exist_ok=True)

grad = pd.read_csv(BASE / "gradcam_complete/gradcam_summary_consolidated.csv")
lime = pd.read_csv(BASE / "lime_complete/lime_summary.csv")
shap_batch = pd.read_csv(BASE / "shap_gradient_batch_ns200/shap_batch_summary.csv")
inference = pd.read_csv(BASE / "inference_results.csv")

# LIME: retain the explanation for the class predicted by the model.
lime_pred = lime[lime["explained_class"] == lime["predicted_label"]].copy()
lime_pred = lime_pred.drop_duplicates(subset="image_index")

# SHAP: use the 200-sample diagnostic array for image 1.
values = np.load(BASE / "shap_gradient_diagnostic_ns200/gradient_shap_values_ns200.npy")
values = np.asarray(values)
if values.ndim == 5 and values.shape[-1] == 1:
    values = values[..., 0]
if values.ndim != 4:
    raise ValueError(f"Unexpected image-1 SHAP shape: {values.shape}")

image1_shap_sum = float(values[0].sum())

first_image_name = grad.iloc[0]["image_path"].split("/")[-1]
first_inf = inference[inference["image_path"].astype(str).str.endswith(first_image_name)]
if len(first_inf) != 1:
    raise ValueError(f"Could not uniquely match image 1 in inference CSV: {len(first_inf)} matches")

first_prediction = float(first_inf.iloc[0]["p_uninfected"])
first_delta = first_prediction - 0.4968311
first_residual = first_delta - image1_shap_sum

first_shap = {
    "image_number": 1,
    "image_name": first_image_name,
    "shap_sum": image1_shap_sum,
    "prediction_delta": first_delta,
    "additivity_residual": first_residual,
    "status": "success_ns200"
}

shap_all = pd.concat(
    [pd.DataFrame([first_shap]), shap_batch],
    ignore_index=True,
    sort=False
)

# Standardize join keys using image filename.
grad["image_name"] = grad["image_path"].astype(str).str.split("/").str[-1]
lime_pred["image_name"] = lime_pred["image_path"].astype(str).str.split("/").str[-1]

comparison = (
    grad.merge(
        lime_pred[["image_name", "surrogate_r2", "positive_area_fraction",
                   "negative_area_fraction", "runtime_seconds"]],
        on="image_name", how="left", validate="one_to_one",
        suffixes=("", "_lime")
    )
    .merge(
        shap_all[["image_name", "shap_sum", "prediction_delta",
                  "additivity_residual", "status"]],
        on="image_name", how="left", validate="one_to_one"
    )
)

comparison = comparison.rename(columns={
    "surrogate_r2": "lime_surrogate_r2",
    "positive_area_fraction": "lime_positive_area_fraction",
    "negative_area_fraction": "lime_negative_area_fraction",
    "runtime_seconds": "lime_runtime_seconds",
    "status": "shap_status"
})

path = OUT / "cross_method_comparison.csv"
comparison.to_csv(path, index=False)

print("Saved:", path)
print("Rows:", len(comparison))
print("Missing values by method:")
print(comparison[[
    "parasitized_cam_status", "uninfected_cam_status",
    "lime_surrogate_r2", "shap_sum", "shap_status"
]].isna().sum())
print("\nPreview:")
print(comparison[[
    "image_name", "class_name", "predicted_label",
    "lime_surrogate_r2", "shap_sum", "additivity_residual"
]].to_string(index=False))
