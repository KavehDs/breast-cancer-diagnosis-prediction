"""Streamlit demo: breast-cancer diagnosis prediction.

Trains a LightGBM model in memory on the bundled 569-row dataset and
lets the user enter the 10 mean nucleus measurements. Shows the
predicted diagnosis and the malignant probability.

EDUCATIONAL PROJECT ONLY: not a medical diagnosis tool.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Diagnosis Prediction", layout="centered")
st.title("Breast Cancer Diagnosis Prediction")
st.warning(
    "Educational portfolio project on a small public research dataset "
    "(569 cases, 1995). **Not a medical diagnosis tool** — never use it "
    "for clinical decisions."
)

MEAN_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
]


@st.cache_resource
def load_model():
    import lightgbm as lgb

    sk = load_breast_cancer()
    X = pd.DataFrame(sk.data, columns=sk.feature_names)
    y = (sk.target == 0).astype(int)  # 1 = malignant
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=2, verbose=-1,
    )
    model.fit(X, y)
    defaults = X[MEAN_FEATURES].median()
    return model, defaults, X


model, defaults, X = load_model()

st.subheader("Enter the 10 mean nucleus measurements")
st.caption("Defaults are the dataset medians. Other features use the median values.")
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(MEAN_FEATURES):
    with cols[i % 2]:
        inputs[feat] = st.number_input(
            feat, value=float(defaults[feat]), format="%.4f", key=feat
        )

if st.button("Predict"):
    row = X.median(numeric_only=True).to_dict()
    row.update(inputs)
    x = pd.DataFrame([row], columns=X.columns)
    prob = float(model.predict_proba(x)[0, 1])
    pred = "Malignant" if prob >= 0.5 else "Benign"
    st.subheader(f"Prediction: **{pred}**")
    st.metric("Probability of malignant", f"{prob:.1%}")
    st.progress(prob)
    if prob >= 0.5:
        st.error(
            "The model flags this case as malignant. In a real setting this "
            "would mean urgent clinical review — the model alone decides nothing."
        )
    else:
        st.success(
            "The model flags this case as benign. Note: on the held-out test "
            "set the best model still missed 3 of 42 malignant cases."
        )

st.divider()
st.caption(
    "Model: LightGBM — test ROC-AUC 0.9967, malignant recall 0.9286 "
    "(test n=114). See README.md for the full results and limitations."
)
