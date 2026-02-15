
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 — Classification Model Evaluation")

st.markdown(
    """
Upload your **test dataset** and select one of the trained models to evaluate metrics, confusion matrix, and classification report.
    """
)

@st.cache_resource
def load_artifacts():
    """
    Loads all trained model artifacts and metadata.

    """
    import os, json, joblib
    from pathlib import Path


    primary_dir = Path("models")

    def has_artifacts(d: Path) -> bool:
        must_have = [
            "logistic_regression.pkl",
            "decision_tree.pkl",
            "knn.pkl",
            "naive_bayes_gaussian.pkl",
            "random_forest.pkl",
            "xgboost.pkl",
            "label_encoder.pkl",
            "metadata.json",
            "metrics_summary.json",
        ]
        return all((d / f).exists() for f in must_have)

    if has_artifacts(primary_dir):
        model_dir = primary_dir

    # Map display names -> filenames
    mapping = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Gaussian Naive Bayes": "naive_bayes_gaussian.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
    }

    models = {}
    missing_models = []
    for label, fname in mapping.items():
        p = model_dir / fname          # <-- consistent dir
        if p.exists():
            models[label] = joblib.load(p)
        else:
            missing_models.append(str(p))

    # Metadata & encoder
    meta_path = model_dir / "metadata.json"
    le_path   = model_dir / "label_encoder.pkl"

    if not meta_path.exists() or not le_path.exists():
        # Provide a helpful error so the Streamlit UI can show the banner
        raise FileNotFoundError(
            f"Missing metadata or label encoder in {model_dir}. "
            f"Expected: {meta_path.name}, {le_path.name}"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)
    le = joblib.load(le_path)

    # If any models were missing, surface a gentle warning in the UI via st.warning
    if missing_models:
        st.warning(
            "Some model files are missing. Expected the following files under "
            f"`{model_dir}` but didn’t find:\n\n- " + "\n- ".join(missing_models)
        )

    return models, meta, le

try:
    models, meta, label_encoder = load_artifacts()
    target_col = meta["target_column"]
    class_names = meta["class_names"]
except Exception as e:
    st.error("Artifacts not found. Run training notebooks to create models/ and metadata.")
    st.stop()

uploaded = st.file_uploader("Upload test dataset (.xlsx or .csv)", type=["xlsx","csv"])

if uploaded:
    # Load Excel or CSV
    try:
        if uploaded.name.lower().endswith('.xlsx'):
            df = pd.read_excel(uploaded, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.write("### Preview", df.head())

    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` missing in uploaded data.")
        st.stop()

    X_test = df.drop(columns=[target_col])
    y_test_raw = df[target_col]

    try:
        y_test = label_encoder.transform(y_test_raw)
    except Exception:
        st.error("Label encoder mismatch. Ensure test uses same labels as training.")
        st.stop()

    model_choice = st.selectbox("Select Model", list(models.keys()))

    if st.button("Evaluate Model"):
        pipe = models[model_choice]
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef,  confusion_matrix, classification_report
        )
        from utils_metrics import safe_auc

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = safe_auc(y_test, y_proba) 

        st.subheader("✨ Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision (weighted)", f"{prec:.3f}")
        c3.metric("Recall (weighted)", f"{rec:.3f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("F1 (weighted)", f"{f1:.3f}")
        c5.metric("MCC", f"{mcc:.3f}")
        c6.metric("AUC", "N/A" if auc is None else f"{auc:.3f}")

        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().style.format('{:.3f}'))
else:
    st.info("Upload an Excel or CSV file to begin.")
