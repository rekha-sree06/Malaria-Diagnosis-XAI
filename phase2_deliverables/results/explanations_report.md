# Evidence-Gated Explanations for Malaria Detection

These explanations describe model behaviour, not clinical diagnoses or verified parasite localization.

## Image 1: C68P29N_ThinF_IMG_20150819_133236_cell_144.png

The model predicted Parasitized with a probability of 99.9994% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.724. Positive-contribution superpixels cover 7.1% of the image, while negative-contribution superpixels cover 5.2%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.4859, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0110. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 2: C124P85ThinF_IMG_20151002_155026_cell_172.png

The model predicted Parasitized with a probability of 99.9549% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.743. Positive-contribution superpixels cover 9.5% of the image, while negative-contribution superpixels cover 4.4%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.5612, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0646. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 3: C103P64ThinF_IMG_20150918_164250_cell_173.png

The model predicted Parasitized with a probability of 99.6398% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.629. Positive-contribution superpixels cover 9.3% of the image, while negative-contribution superpixels cover 7.7%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.4210, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0719. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 4: C84P45ThinF_IMG_20150818_101226_cell_93.png

The model predicted Parasitized with a probability of 99.9999% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.829. Positive-contribution superpixels cover 9.6% of the image, while negative-contribution superpixels cover 5.9%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.6242, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.1274. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 5: C149P110ThinF_IMG_20151115_115154_cell_258.png

The model predicted Parasitized with a probability of 99.9984% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.557. Positive-contribution superpixels cover 6.2% of the image, while negative-contribution superpixels cover 16.0%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.4130, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0838. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 6: C137P98ThinF_IMG_20151005_163335_cell_3.png

The model predicted Parasitized with a probability of 90.8450% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.408. Positive-contribution superpixels cover 8.3% of the image, while negative-contribution superpixels cover 7.1%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.3765, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0007. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 7: C136P97ThinF_IMG_20151005_143329_cell_95.png

The model predicted Parasitized with a probability of 99.7700% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.721. Positive-contribution superpixels cover 8.8% of the image, while negative-contribution superpixels cover 3.8%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.5100, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0174. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 8: C129P90ThinF_IMG_20151004_134636_cell_20.png

The model predicted Parasitized with a probability of 99.9922% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.915. Positive-contribution superpixels cover 14.4% of the image, while negative-contribution superpixels cover 6.9%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.4697, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0270. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 9: C82P43ThinF_IMG_20150817_125114_cell_19.png

The model predicted Parasitized with a probability of 99.9959% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.537. Positive-contribution superpixels cover 11.0% of the image, while negative-contribution superpixels cover 8.7%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.3744, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.1224. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 10: C120P81ThinF_IMG_20151002_130715_cell_192.png

The model predicted Parasitized with a probability of 99.9732% for that class.

Grad-CAM: A Grad-CAM map from a previous run is available for the predicted class. It indicates image regions that influenced the model's prediction, but was not independently regenerated in the consolidated batch. It does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.557. Positive-contribution superpixels cover 12.5% of the image, while negative-contribution superpixels cover 6.1%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is -0.4734, indicating a net direction toward the Parasitized output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0228. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images. Grad-CAM provenance includes maps retained from a previous run.

---

## Image 11: C62P23N_ThinF_IMG_20150818_133307_cell_150.png

The model predicted Uninfected with a probability of 99.4486% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.334. Positive-contribution superpixels cover 8.1% of the image, while negative-contribution superpixels cover 3.9%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.4782, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0161. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 12: C76P37ThinF_IMG_20150815_173019_cell_141.png

The model predicted Uninfected with a probability of 99.9573% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.286. Positive-contribution superpixels cover 7.9% of the image, while negative-contribution superpixels cover 6.7%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.5383, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.0354. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 13: C238NThinF_IMG_20151207_114605_cell_18.png

The model predicted Uninfected with a probability of 91.5306% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.300. Positive-contribution superpixels cover 5.0% of the image, while negative-contribution superpixels cover 6.6%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.2616, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.1295. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 14: C120P81ThinF_IMG_20151002_131120_cell_130.png

The model predicted Uninfected with a probability of 99.6353% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.343. Positive-contribution superpixels cover 11.4% of the image, while negative-contribution superpixels cover 7.3%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.4227, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0752. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 15: C45P6ThinF_IMG_20151130_154725_cell_86.png

The model predicted Uninfected with a probability of 94.9593% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.287. Positive-contribution superpixels cover 6.9% of the image, while negative-contribution superpixels cover 7.6%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.3837, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0563. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 16: C208ThinF_IMG_20151029_160108_cell_187.png

The model predicted Uninfected with a probability of 99.8494% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.295. Positive-contribution superpixels cover 7.6% of the image, while negative-contribution superpixels cover 5.4%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.6341, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.1319. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 17: C107P68ThinF_IMG_20150924_222415_cell_23.png

The model predicted Uninfected with a probability of 99.8317% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.238. Positive-contribution superpixels cover 6.7% of the image, while negative-contribution superpixels cover 3.9%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.4994, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0028. The residual is within the provisional 0.05 reporting threshold. This does not establish exact additivity, biological correctness, or clinical validity. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 18: C107P68ThinF_IMG_20150924_221632_cell_45.png

The model predicted Uninfected with a probability of 98.5620% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.318. Positive-contribution superpixels cover 7.0% of the image, while negative-contribution superpixels cover 6.2%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.6186, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is -0.1283. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 19: C122P83ThinF_IMG_20151002_145014_cell_33.png

The model predicted Uninfected with a probability of 99.7610% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.213. Positive-contribution superpixels cover 5.2% of the image, while negative-contribution superpixels cover 4.2%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.3114, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.1902. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---

## Image 20: C150P111ThinF_IMG_20151115_115805_cell_56.png

The model predicted Uninfected with a probability of 99.9936% for that class.

Grad-CAM: A Grad-CAM map was generated successfully for the predicted class. It indicates image regions that influenced the model's prediction; it does not establish parasite location.

LIME: LIME produced a local surrogate R² of 0.359. Positive-contribution superpixels cover 6.7% of the image, while negative-contribution superpixels cover 12.1%. The R² describes local surrogate fit, not biological correctness.

SHAP: The sum of SHAP attributions across the image is +0.4391, indicating a net direction toward the Uninfected output for the model output P(Uninfected). This aggregate does not describe individual pixel contributions. The additivity residual is +0.0640. The absolute additivity residual exceeds the provisional 0.05 reporting threshold. Therefore, the aggregate SHAP direction is provisional and must not be treated as confirmed directional evidence for the model output. The 0.05 threshold is a project-specific reporting heuristic, not a universally validated cutoff.

Limitations: Visual attribution does not prove parasite presence or localization. No independent expert localization masks were available. LIME R² measures surrogate fit, not biological correctness. The pilot contains only 20 exploratory images.

---
