# Evidence-Gated Natural-Language Explainable AI for Malaria Detection

## Overview

This project explores Explainable AI (XAI) for malaria cell-image
classification using two deep learning models:

- Model 1: Xception
- Model 2: InceptionResNetV2

The XAI methods include:
- Grad-CAM
- LIME
- SHAP GradientExplainer

The Phase 2 extension focuses on Model 1 (Xception). It introduces
an evidence-gated natural-language reporting layer that communicates
measured explanation evidence and method-specific limitations.

The reporting layer uses deterministic rules and templates. It is
not an LLM and does not retrain or replace the classifier.

## Reference Paper

Parveen et al. (2025).
"Trustworthy deep learning for malaria diagnosis using explainable
artificial intelligence."

Scientific Reports.

https://doi.org/10.1038/s41598-025-28387-7

## Technologies

- Python 3.11
- TensorFlow
- NumPy
- Pandas
- OpenCV
- SHAP
- LIME
- scikit-image
- scikit-learn
- SciPy
- Matplotlib

## Environment Setup

Experiments were conducted using Python 3.11 in Ubuntu under WSL2.

Create and activate a virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

The dataset and trained model checkpoints are not included in the
Git repository. Place them in the expected local paths before running
the scripts. Some experiment scripts use script-specific paths that
may need to be configured.

## Important Scripts

| Script | Purpose |
|---|---|
| model1.py | Xception model implementation |
| model2.py | InceptionResNetV2 model implementation |
| run_xai_validated.py | Standalone validated XAI utility for Model 1 |
| run_xai_model2.py | XAI workflow for Model 2 |
| run_phase2_gradcam.py | Phase 2 Grad-CAM generation |
| run_phase2_lime_v2.py | Phase 2 LIME generation |
| run_gradient_shap_pilot.py | Gradient SHAP pilot and background generation |
| run_shap_batch_ns200.py | Gradient SHAP batch experiment |
| create_cross_method_comparison.py | Cross-method comparison |
| generate_baseline_explanations.py | Baseline explanation generation |
| generate_evidence_explanations_v3.py | Evidence-gated explanation generation |
| evaluate_explanations_v5.py | Automated explanation evaluation |
| run_faithfulness_deletion.py | Deletion-based faithfulness experiment |
| run_gradcam_stability.py | Grad-CAM stability experiment |

## Phase 2 Results

Curated and reproducible Phase 2 outputs are stored in:

    phase2_deliverables/results/

Important files include:

- cross_method_comparison.csv
- evidence_explanations.csv
- evidence_summary.txt
- explanations_report.md
- baseline_explanations.md
- automated_evaluation_summary_v5.csv
- automated_evaluation_details_v5.csv
- deletion_summary.csv
- stability_results.csv
- lime_summary.csv
- shap_batch_summary.csv

The experimental comparison report is available at:

    phase2_deliverables/Experimental_Results_Comparison.pdf

Raw experimental outputs are excluded from Git.

## Evaluation Scope and Limitations

- The pilot evaluation uses 20 images: 10 Parasitized and 10 Uninfected.
- The 20/20 correct pilot predictions must not be interpreted as full
  dataset or independent test-set accuracy.
- The evidence-gated extension was evaluated using Model 1 (Xception).
- The automated evaluation measures predefined evidence and limitation
  disclosure coverage. It is not an expert factuality or clinical-validity
  assessment.
- LIME R-squared measures local surrogate fit, not biological correctness.
- SHAP aggregate values do not establish individual pixel-level effects.
- The SHAP additivity residual threshold of 0.05 is a provisional
  project-specific reporting heuristic.
- The deletion experiment is exploratory and does not establish
  statistically significant faithfulness improvement over random ordering.
- Stability testing uses a limited five-image pilot and mild image changes.
- No independent pathologist-annotated segmentation masks were available.
- No clinician or user study was conducted.
- Heatmaps must not be interpreted as verified parasite localization.

### Dataset Source

The dataset used in this project is **Cell Images for Detecting Malaria**, publicly available on Kaggle.

- **Source:** [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Uploader:** iarunava
- **Total images:** 27,558
- **Classes:** Parasitized and Uninfected

The dataset is not included in this repository and must be downloaded separately.

## Reproducibility Notes

Use the same dataset, model checkpoints, preprocessing settings,
and dependency versions when reproducing experiments.

Inspect script-specific input and output paths before execution.
Some scripts create output directories with exclusive creation and
will not overwrite existing experiment outputs.
