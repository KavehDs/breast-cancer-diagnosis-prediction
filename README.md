# Breast Cancer Diagnosis Prediction

Classify a breast fine-needle aspirate (FNA) as malignant or benign from
30 measurements of cell nuclei — and be honest about the three cases the
best model still misses.

> ⚠️ **This is an educational portfolio project on a small public
> research dataset, not a medical diagnosis tool. It must not be used
> for clinical decisions.** The model was trained on 569 cases from a
> single hospital in 1995 and has never been validated on new patients.

## Results (test set, n = 114; 42 malignant, 72 benign)

Stratified 80/20 train/test split (`random_state=42`). The positive
class is **malignant**: missing a malignant case (false negative) is
the costly error, so malignant recall is the headline metric alongside
ROC-AUC — not accuracy alone.

| Model | Accuracy | Precision (mal.) | Recall (mal.) | F1 (mal.) | ROC-AUC | Brier | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 | 0.0213 | 3 | 1 |
| XGBoost | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9934 | 0.0204 | 4 | 0 |
| **LightGBM** | 0.9737 | 1.0000 | 0.9286 | 0.9630 | **0.9967** | 0.0254 | 3 | 0 |
| SVM (RBF, calibrated) | 0.9737 | 0.9756 | **0.9524** | **0.9639** | 0.9947 | 0.0222 | 2 | 1 |

LightGBM is best by ROC-AUC; the calibrated SVM has the best malignant
recall (40 of 42 found) and the best F1. No model is best on every
metric. A majority-class baseline (always "benign") would score 63.2%
accuracy and find zero malignant cases.

## Threshold analysis (LightGBM)

Lowering the decision threshold does **not** help here — the three
missed malignant cases are predicted with probability below 0.1, so
they stay missed at every threshold from 0.5 down to 0.1:

| Threshold | Precision (mal.) | Recall (mal.) | F1 | FN | FP |
|---|---|---|---|---|---|
| 0.5 | 1.0000 | 0.9286 | 0.9630 | 3 | 0 |
| 0.4 | 1.0000 | 0.9286 | 0.9630 | 3 | 0 |
| 0.3 | 1.0000 | 0.9286 | 0.9630 | 3 | 0 |
| 0.2 | 1.0000 | 0.9286 | 0.9630 | 3 | 0 |
| 0.1 | 1.0000 | 0.9286 | 0.9630 | 3 | 0 |

The honest reading: threshold tuning cannot fix confidently-wrong
predictions. Fixing those three cases would need better features or
more data, not a different cut-off.

## Calibration

Probabilities are strongly polarised (most predictions are near 0 or
near 1), so the Brier scores (~0.02) look good while the reliability
plot shows the model is overconfident in the middle range. Treat the
probabilities as ranking scores, not as calibrated risks. Full
reliability bins are in `metrics.json`.

## Top drivers (SHAP, LightGBM, mean |SHAP| on the test set)

1. worst perimeter — 2.3183
2. worst concave points — 2.0332
3. mean concave points — 1.5924
4. worst concavity — 1.4798
5. worst area — 1.2785

The "worst" (largest) nucleus measurements dominate — size and shape
irregularity of the largest nuclei drive the malignant prediction,
which matches the clinical intuition behind the dataset.

## Method

- **Data:** Breast Cancer Wisconsin (Diagnostic), UCI #17
  (Wolberg, Street & Mangasarian, 1995). 569 cases, 30 numeric
  features: mean, standard error and "worst" of 10 nucleus
  measurements (radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal
  dimension). Malignant 212 / benign 357. No missing values.
  Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Features:** used as-is (no feature selection); Logistic Regression
  and SVM use a `StandardScaler`, tree models do not need one.
- **Evaluation:** single stratified 80/20 split, `random_state=42`,
  test n = 114. Metrics are reported on the test set only.
- **Reproduce:** `python src/train.py` — it prints these numbers and
  writes `metrics.json` plus the plots in `plots/`. The script was run
  twice and reproduces the metrics exactly.
- **Demo:** `streamlit run app.py` — enter the 10 mean measurements
  and see the predicted diagnosis and probability (LightGBM model).

## Limitations

- **Tiny and old:** 569 cases from one hospital, collected in 1995.
  No external validation, no information on patient age, ethnicity or
  imaging equipment — the model may not generalise to any other
  population or scanner.
- **High accuracy is partly the dataset:** the classes are well
  separated in this data (ROC-AUC ~0.99 for every model). A 97%
  accuracy here says more about the dataset than about the model.
- **Three malignant cases are missed** by the best model, and lowering
  the threshold does not catch them (see above).
- **Not a diagnosis:** this is a portfolio exercise. Real diagnosis
  needs a clinician, imaging, biopsy and follow-up — never a model
  score alone.

## Project structure

```
breast-cancer-diagnosis-prediction/
├── README.md
├── requirements.txt
├── metrics.json
├── app.py                  # Streamlit demo (10 mean features)
├── data/
│   └── breast_cancer.csv   # 569 rows, public UCI dataset
├── plots/
│   ├── confusion_matrix.svg
│   ├── roc_curves.svg
│   └── shap_top_drivers.svg
└── src/
    └── train.py            # trains, evaluates, writes metrics.json + plots
```
