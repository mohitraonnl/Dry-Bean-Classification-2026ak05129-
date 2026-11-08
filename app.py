"""
BeanSort — Streamlit front-end for the Dry Bean classifiers.

Run from the project root:
    streamlit run app.py

Expected project layout (the app also searches a few common alternatives):
    app.py
    model/
        *.joblib or *.pkl       # five fitted scikit-learn pipelines
    outputs/
        test_data.csv
        training_summary.csv

The app never calls fit() and never retrains a model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR_CANDIDATES = [APP_DIR / "model", APP_DIR / "models"]
DATA_CANDIDATES = [
    APP_DIR / "outputs" / "test_data.csv",
    APP_DIR / "data" / "test_data.csv",
    APP_DIR / "test_data.csv",
    APP_DIR / "dataset" / "test_data.csv",
]
SUMMARY_CANDIDATES = [
    APP_DIR / "outputs" / "training_summary.csv",
    APP_DIR / "training_summary.csv",
    APP_DIR / "model" / "training_summary.csv",
]

st.set_page_config(
    page_title="BeanSort | Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

st.title("🫘 BeanSort — Dry Bean Variety Classifier")
st.caption(
    "Evaluate the saved classifiers on a CSV dataset. Models are loaded from disk; "
    "this application does not train or modify them."
)


def first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def load_model_file(path: Path) -> Any:
    """Load a previously fitted estimator. No fitting occurs here."""
    import joblib

    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as file:
            return pickle.load(file)


@st.cache_resource(show_spinner="Loading saved classifiers…")
def load_models() -> dict[str, Any]:
    model_dir = first_existing(MODEL_DIR_CANDIDATES)
    if model_dir is None:
        return {}

    extensions = {".joblib", ".pkl", ".pickle"}
    files = sorted(
        p for p in model_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    loaded: dict[str, Any] = {}
    for path in files:
        try:
            label_map = {
                "logistic_regression": "Logistic Regression",
                "decision_tree": "Decision Tree",
                "knn": "KNN",
                "naive_bayes": "Naive Bayes",
                "svm": "Support Vector Machine",
            }
            display_name = label_map.get(
                path.stem.lower(), path.stem.replace("_", " ").replace("-", " ").title()
            )
            loaded[display_name] = load_model_file(path)
        except Exception as exc:
            st.warning(f"Could not load `{path.name}`: {exc}")
    return loaded


@st.cache_data(show_spinner=False)
def load_training_summary() -> pd.DataFrame:
    """Load the CSV summary produced by train_models.py."""
    path = first_existing(SUMMARY_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read training summary: {exc}")
        return pd.DataFrame()


def get_feature_columns(model: Any, summary: Any) -> list[str] | None:
    """Prefer metadata, then use a fitted sklearn pipeline's feature names."""
    for key in ("feature_columns", "features", "feature_names", "X_columns"):
        value = summary.get(key)
        if isinstance(value, list) and value:
            return [str(x) for x in value]

    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    # Pipeline feature names are commonly exposed by the first/last fitted step.
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]
    return None


def get_target_column(df: pd.DataFrame, summary: Any) -> str | None:
    for key in ("target_column", "target", "label_column", "label"):
        value = summary.get(key)
        if isinstance(value, str) and value in df.columns:
            return value

    common_names = (
        "Class", "class", "target", "Target", "label", "Label",
        "variety", "Variety", "bean_type",
    )
    return next((name for name in common_names if name in df.columns), None)


def positive_auc(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Calculate multiclass one-vs-rest macro ROC-AUC, matching training."""
    if not hasattr(model, "predict_proba"):
        return float("nan")
    try:
        probabilities = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
            classes = getattr(final_estimator, "classes_", None)
        if classes is None:
            return float("nan")

        if len(classes) == 2:
            return float(roc_auc_score(y, probabilities[:, 1]))
        return float(
            roc_auc_score(
                y,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except (ValueError, IndexError):
        return float("nan")


def evaluate(model: Any, X: pd.DataFrame, y_true: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X)
    return {
        "predictions": np.asarray(y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": positive_auc(model, X, y_true),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def metric_label(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def confusion_chart(y_true: pd.Series, y_pred: np.ndarray, normalize: bool) -> alt.Chart:
    labels = sorted(pd.unique(pd.concat(
        [pd.Series(y_true).astype(str), pd.Series(y_pred).astype(str)],
        ignore_index=True,
    )).tolist())
    cm = confusion_matrix(
        y_true.astype(str), pd.Series(y_pred).astype(str), labels=labels,
        normalize="true" if normalize else None,
    )
    rows = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            value = float(cm[i, j])
            rows.append({
                "Actual": actual,
                "Predicted": predicted,
                "Value": value,
                "Display": f"{value:.1%}" if normalize else str(int(value)),
            })
    frame = pd.DataFrame(rows)
    color_title = "Row-normalized rate" if normalize else "Number of samples"
    base = alt.Chart(frame).encode(
        x=alt.X("Predicted:N", title="Predicted", sort=labels),
        y=alt.Y("Actual:N", title="Actual", sort=labels),
        tooltip=[
            alt.Tooltip("Actual:N"),
            alt.Tooltip("Predicted:N"),
            alt.Tooltip("Display:N", title=color_title),
        ],
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Value:Q", title=color_title, scale=alt.Scale(scheme="blues"))
    )
    text = base.mark_text(fontSize=11).encode(text="Display:N")
    return (heat + text).properties(height=max(300, 38 * len(labels)))


models = load_models()
summary = load_training_summary()

with st.sidebar:
    st.header("Data & model")
    data_source = st.radio(
        "Choose data source",
        ["Use bundled test_data.csv", "Upload my CSV"],
        index=0,
    )
    uploaded_file = None
    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader("Upload a test CSV", type=["csv"])
    else:
        bundled_path = first_existing(DATA_CANDIDATES)
        if bundled_path:
            st.success(f"Found `{bundled_path.relative_to(APP_DIR)}`")
        else:
            st.error("Bundled test_data.csv was not found.")

    if models:
        selected_model_name = st.selectbox("Select classifier", list(models.keys()))
    else:
        selected_model_name = None
        st.error("No saved models found in `model/` or `models/`.")

    normalize_cm = st.toggle("Normalize confusion matrix", value=False)

if data_source == "Upload my CSV":
    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    bundled_path = first_existing(DATA_CANDIDATES)
    if bundled_path is None:
        st.stop()
    try:
        data = pd.read_csv(bundled_path)
    except Exception as exc:
        st.error(f"Could not read bundled test CSV: {exc}")
        st.stop()

if data.empty:
    st.error("The selected CSV contains no rows.")
    st.stop()

if not models or selected_model_name is None:
    st.stop()

selected_model = models[selected_model_name]
feature_columns = get_feature_columns(selected_model, summary)
target_column = get_target_column(data, summary)

if feature_columns is None:
    # Fall back to numeric columns, excluding the likely target.
    feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in feature_columns:
        feature_columns.remove(target_column)

if not feature_columns:
    st.error(
        "Could not determine model input columns. save a model fitted with named feature columns."
    )
    st.stop()

missing_columns = [column for column in feature_columns if column not in data.columns]
if missing_columns:
    st.error("The CSV is missing required feature columns:")
    st.code(", ".join(missing_columns))
    st.stop()

st.subheader("Dataset preview")
st.write(f"**Rows:** {len(data):,}  ·  **Features required:** {len(feature_columns)}")
st.dataframe(data.head(10), use_container_width=True)

X = data.loc[:, feature_columns].copy()
# Numeric conversion catches accidental text values before model inference.
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")
if X.isna().any().any():
    bad_columns = X.columns[X.isna().any()].tolist()
    st.error(
        "Missing or non-numeric values were found in feature columns: "
        + ", ".join(bad_columns)
        + ". Clean the CSV before prediction."
    )
    st.stop()

tab_eval, tab_compare, tab_about = st.tabs(
    ["Selected model evaluation", "Compare all 5 models", "About the models"]
)

with tab_eval:
    st.subheader(f"Evaluation — {selected_model_name}")
    if target_column is None:
        st.warning(
            "No target/ground-truth column was detected. The app can still generate "
            "predictions, but evaluation metrics and the confusion matrix require a "
            "target column such as `Class`."
        )
        try:
            predictions = selected_model.predict(X)
            output = data.copy()
            output["Predicted"] = predictions
            st.dataframe(output, use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="bean_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    else:
        y_true = data[target_column]
        try:
            result = evaluate(selected_model, X, y_true)
        except Exception as exc:
            st.error(f"Model evaluation failed: {exc}")
            st.stop()

        metric_columns = st.columns(6)
        metric_values = [
            ("Accuracy", result["accuracy"]),
            ("AUC", result["auc"]),
            ("Precision", result["precision"]),
            ("Recall", result["recall"]),
            ("F1", result["f1"]),
            ("MCC", result["mcc"]),
        ]
        for column, (label, value) in zip(metric_columns, metric_values):
            column.metric(label, metric_label(value))

        left, right = st.columns([1.15, 0.85])
        with left:
            st.markdown("#### Confusion matrix")
            st.altair_chart(
                confusion_chart(y_true, result["predictions"], normalize_cm),
                use_container_width=True,
            )
        with right:
            st.markdown("#### Classification report")
            report = classification_report(
                y_true, result["predictions"], output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)

        output = data.copy()
        output["Predicted"] = result["predictions"]
        output["Correct"] = output[target_column].astype(str) == output["Predicted"].astype(str)
        misclassified = output.loc[~output["Correct"]].copy()

        st.markdown("#### Misclassified samples")
        st.caption(f"{len(misclassified):,} of {len(output):,} rows were misclassified.")
        if misclassified.empty:
            st.success("No misclassified rows in this dataset.")
        else:
            st.dataframe(misclassified, use_container_width=True, height=260)
            counts = (
                misclassified["Predicted"].astype(str)
                .value_counts().rename_axis("Predicted class")
                .reset_index(name="Misclassified count")
            )
            st.altair_chart(
                alt.Chart(counts).mark_bar().encode(
                    x=alt.X("Predicted class:N", sort="-y"),
                    y=alt.Y("Misclassified count:Q"),
                    tooltip=["Predicted class", "Misclassified count"],
                ).properties(height=250),
                use_container_width=True,
            )

        st.download_button(
            "Download predictions CSV",
            output.to_csv(index=False).encode("utf-8"),
            file_name="bean_predictions.csv",
            mime="text/csv",
        )

with tab_compare:
    st.subheader("Compare all saved classifiers on the same test data")
    if target_column is None:
        st.info("A target column (for example, `Class`) is required to compare metrics.")
    else:
        comparison_rows = []
        comparison_errors = []
        y_true = data[target_column]
        for name, model in models.items():
            try:
                result = evaluate(model, X, y_true)
                comparison_rows.append({
                    "Model": name,
                    "Accuracy": result["accuracy"],
                    "AUC": result["auc"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1": result["f1"],
                    "MCC": result["mcc"],
                })
            except Exception as exc:
                comparison_errors.append(f"{name}: {exc}")

        if comparison_rows:
            comparison = pd.DataFrame(comparison_rows).sort_values(
                "F1", ascending=False
            )
            st.dataframe(
                comparison.style.format({
                    metric: "{:.4f}" for metric in
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
                }),
                use_container_width=True,
                hide_index=True,
            )
            metric_to_plot = st.selectbox(
                "Metric for model comparison",
                ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                index=4,
            )
            st.altair_chart(
                alt.Chart(comparison).mark_bar().encode(
                    x=alt.X("Model:N", sort="-y"),
                    y=alt.Y(f"{metric_to_plot}:Q", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Model", metric_to_plot],
                ).properties(height=350),
                use_container_width=True,
            )
        for message in comparison_errors:
            st.warning(f"Could not evaluate {message}")

with tab_about:
    st.subheader("About the models")
    if not summary.empty:
        st.markdown("#### Hyperparameter tuning summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "The table is loaded from outputs/training_summary.csv and contains "
            "the best parameters, cross-validation macro-F1, and training time."
        )
    else:
        st.info(
            "No training_summary.csv was found. Expected location: "
            "`outputs/training_summary.csv`."
        )

    metrics_path = APP_DIR / "outputs" / "test_metrics.csv"
    if metrics_path.exists():
        st.markdown("#### Saved hold-out test results")
        try:
            saved_metrics = pd.read_csv(metrics_path)
            st.dataframe(
                saved_metrics.style.format({
                    metric: "{:.4f}" for metric in
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
                    if metric in saved_metrics.columns
                }),
                use_container_width=True,
                hide_index=True,
            )
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not read outputs/test_metrics.csv: {exc}")

    st.markdown("**Runtime behavior**")
    st.write(
        "- Loads the five fitted model files from the model folder.\\n"
        "- Uses the uploaded or bundled CSV only for prediction/evaluation.\\n"
        "- Does not call fit(), retrain, or overwrite any model files.\\n"
        "- Uses macro-averaged Precision, Recall, F1 and multiclass OVR AUC "
        "to match the training evaluation."
    )
