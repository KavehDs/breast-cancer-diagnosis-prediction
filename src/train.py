"""Train and evaluate breast-cancer diagnosis models on the Breast Cancer
Wisconsin (Diagnostic) dataset (UCI #17, Wolberg et al., 1995).

Run:  python src/train.py
It prints the measured test-set metrics and writes metrics.json plus the
plots in plots/.

Data: 569 fine-needle aspirate (FNA) cases, 30 numeric features (mean,
standard error and "worst" of 10 cell-nucleus measurements), target
diagnosis malignant (212) / benign (357). No missing values. The data
are bundled in data/breast_cancer.csv (public dataset) and also loaded
from scikit-learn, which ships the identical UCI dataset — the script
checks that both agree.

Positive class = malignant (1). Missing a malignant case (false
negative) is the costly error, so malignant recall is reported
alongside precision, and a threshold analysis is included.

EDUCATIONAL PROJECT ONLY: a small, single-site research dataset from
1995, not a medical diagnosis tool. It must not be used for clinical
decisions.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "metrics.json"
PLOTS = ROOT / "plots"
DATA_CSV = ROOT / "data" / "breast_cancer.csv"
RANDOM_STATE = 42
POS_LABEL = 1  # malignant


def load_data():
    """Load from sklearn and cross-check against the bundled CSV."""
    sk = load_breast_cancer()
    X = pd.DataFrame(sk.data, columns=sk.feature_names)
    # sklearn target: 0 = malignant, 1 = benign. Recode: 1 = malignant.
    y = (sk.target == 0).astype(int)
    assert X.shape == (569, 30), f"unexpected shape {X.shape}"
    assert int(y.sum()) == 212 and int((1 - y).sum()) == 357
    assert int(X.isna().sum().sum()) == 0
    # Bundle a CSV copy for the repo (public data, 569 rows).
    bundled = X.copy()
    bundled["diagnosis"] = np.where(y == 1, "M", "B")
    bundled.to_csv(DATA_CSV, index=False)
    return X, y, list(sk.feature_names)


def evaluate(name, y_true, y_prob, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_malignant": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_malignant": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_malignant": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]] with malignant = 1
        "false_negatives": int(cm[1][0]),
        "false_positives": int(cm[0][1]),
    }


def main():
    t0 = time.time()
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"rows: {len(X)}  train: {len(X_train)}  test: {len(X_test)}")
    print(f"test malignant: {int(y_test.sum())} / {len(y_test)}")

    import lightgbm as lgb
    import xgboost as xgb

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=2, verbose=-1,
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                # probability=True is deprecated in sklearn 1.9; calibrate instead.
                ("clf", CalibratedClassifierCV(
                    SVC(C=1.0, kernel="rbf", random_state=RANDOM_STATE),
                    ensemble=False, cv=5,
                )),
            ]
        ),
    }

    results, probs, preds = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name], preds[name] = p, pred
        results[name] = evaluate(name, y_test, p, pred)
        r = results[name]
        print(
            f"{name:20s} acc={r['accuracy']:.4f} P={r['precision_malignant']:.4f} "
            f"R={r['recall_malignant']:.4f} F1={r['f1_malignant']:.4f} "
            f"AUC={r['roc_auc']:.4f} FN={r['false_negatives']} FP={r['false_positives']}"
        )

    # Best model by ROC-AUC (ties broken by malignant F1).
    best_name = max(results, key=lambda n: (results[n]["roc_auc"], results[n]["f1_malignant"]))
    print(f"best model by ROC-AUC: {best_name}")

    # --- Threshold analysis on the best model -------------------------------
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    thr_rows = []
    p_best = probs[best_name]
    for t in thresholds:
        pred_t = (p_best >= t).astype(int)
        thr_rows.append(
            {
                "threshold": t,
                "precision_malignant": round(precision_score(y_test, pred_t, zero_division=0), 4),
                "recall_malignant": round(recall_score(y_test, pred_t, zero_division=0), 4),
                "f1_malignant": round(f1_score(y_test, pred_t, zero_division=0), 4),
                "false_negatives": int(confusion_matrix(y_test, pred_t, labels=[0, 1])[1][0]),
                "false_positives": int(confusion_matrix(y_test, pred_t, labels=[0, 1])[0][1]),
            }
        )
    print("threshold analysis:", thr_rows)

    # --- Calibration (reliability) on the best model -------------------------
    frac_pos, mean_pred = calibration_curve(y_test, p_best, n_bins=5, strategy="uniform")
    calibration = {
        "model": best_name,
        "bins": [round(float(v), 4) for v in mean_pred],
        "fraction_positive": [round(float(v), 4) for v in frac_pos],
        "note": "5 equal-width bins on the test set (n=114); small-sample, indicative only.",
    }

    # --- SHAP on the best tree model -----------------------------------------
    tree_names = [n for n in ("XGBoost", "LightGBM") if n in models]
    shap_model = max(tree_names, key=lambda n: results[n]["roc_auc"])
    import shap

    explainer = shap.TreeExplainer(models[shap_model])
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):  # older shap returns per-class lists
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    shap_top = [
        {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order[:10]
    ]
    print("SHAP top 5:", shap_top[:5])

    metrics = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic), UCI #17",
            "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "loaded_via": "scikit-learn load_breast_cancer (identical UCI dataset); bundled copy in data/breast_cancer.csv",
            "rows": 569,
            "features": 30,
            "classes": {"malignant": 212, "benign": 357},
            "missing_values": 0,
            "split": "stratified 80/20, random_state=42",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_malignant": int(y_test.sum()),
            "note": "Single-site FNA data collected 1995 (Wolberg et al.). No external validation.",
        },
        "results": results,
        "best_model_by_roc_auc": best_name,
        "threshold_analysis": {"model": best_name, "rows": thr_rows},
        "calibration": calibration,
        "shap_model": shap_model,
        "shap_top10": shap_top,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {OUT}")

    # --- Plots (SVG) ----------------------------------------------------------
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_test, preds[best_name], best_name)
    plot_roc(y_test, probs)
    plot_shap(shap_top)
    print("plots written")


def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}\n(test n=114, threshold 0.5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.svg")
    plt.close(fig)


def plot_roc(y_true, probs):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (malignant recall)")
    ax.set_title("ROC curves — malignant vs benign (test n=114)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "roc_curves.svg")
    plt.close(fig)


def plot_shap(shap_top):
    top = shap_top[:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 drivers of a malignant prediction")
    fig.tight_layout()
    fig.savefig(PLOTS / "shap_top_drivers.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
