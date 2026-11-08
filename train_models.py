"""Train and evaluate five classifiers on the UCI Dry Bean dataset.

Creates a stratified 80/20 train/test split, tunes models using 5-fold
cross-validation on the training set only, evaluates on the held-out test set,
and saves the split CSVs, trained models, and evaluation summaries.
"""
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from bean_metrics import (
    BEAN_FEATURES, TARGET_COL, SEED, MODEL_FILES, score_classifier
)


def candidate_models():
    return {
        "Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, random_state=SEED))
            ]),
            {"clf__C": [0.1, 1.0, 10.0]}
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 5, 10]
            }
        ),
        "KNN": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier())
            ]),
            {
                "clf__n_neighbors": [5, 7, 9],
                "clf__weights": ["uniform", "distance"]
            }
        ),
        "Naive Bayes": (
            Pipeline([
                ("transform", PowerTransformer(method="yeo-johnson")),
                ("clf", GaussianNB())
            ]),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]}
        ),
        "Support Vector Machine": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    estimator=SVC(random_state=SEED),
                    method="sigmoid",
                    cv=3
                ))
            ]),
            {
                "clf__estimator__C": [1.0, 10.0],
                "clf__estimator__gamma": ["scale", "auto"]
            }
        )
    }


def main(data_path="data/dry_bean_full.csv"):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python download_data.py` first."
        )

    # Load data and normalize known column-name variants.
    beans = pd.read_csv(data_path)
    column_mapping = {}

    for col in beans.columns:
        clean = col.strip()
        if clean.lower() == "aspectratio":
            column_mapping[col] = "AspectRation"
        elif clean.lower() == "roundness":
            column_mapping[col] = "roundness"

    # Assignment instead of inplace=True avoids the pandas deprecation warning.
    beans = beans.rename(columns=column_mapping)
    beans = beans.drop_duplicates().reset_index(drop=True)

    required_columns = BEAN_FEATURES + [TARGET_COL]
    missing = [col for col in required_columns if col not in beans.columns]
    if missing:
        raise ValueError(
            f"Missing columns in dataset: {missing}\n"
            f"Available columns: {beans.columns.tolist()}"
        )

    X = beans[BEAN_FEATURES]
    y = beans[TARGET_COL]

    print("Full dataset shape:", beans.shape)
    print("Features:", X.shape)
    print("Classes:", y.nunique())

    # Create output directories.
    output_dir = Path("outputs")
    model_dir = Path("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Stratified 80/20 split. Test data is held out from all model tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    # Save train/test files with their target column included.
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    train_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training data saved: {train_path} ({len(train_df):,} rows)")
    print(f"Testing data saved:  {test_path} ({len(test_df):,} rows)")
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

    # Five-fold CV is performed only on the training set.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    tuning_rows = []
    metric_rows = []

    for name, (pipe, grid) in candidate_models().items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="f1_macro",
            cv=folds,
            n_jobs=-1,
            refit=True
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        # Persist fitted model.
        model_path = model_dir / MODEL_FILES[name]
        joblib.dump(best_model, model_path, compress=3)

        # Hold-out test metrics.
        scores = score_classifier(
            y_test, y_pred, y_proba, best_model.classes_
        )
        metric_rows.append({"Model": name, **scores})

        elapsed = round(time.time() - start_time, 1)
        tuning_rows.append({
            "Model": name,
            "Best params": str(search.best_params_),
            "CV macro-F1": round(search.best_score_, 4),
            "Time (s)": elapsed
        })

        print("Best parameters:", search.best_params_)
        print("CV macro-F1:", round(search.best_score_, 4))
        print("Test macro-F1:", round(scores["F1"], 4))
        print("Saved model:", model_path)

    # Save tuning and hold-out evaluation tables.
    tuning_df = pd.DataFrame(tuning_rows)
    metrics_df = pd.DataFrame(metric_rows)

    tuning_df.to_csv(output_dir / "training_summary.csv", index=False)
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    print("\nTraining and evaluation completed!")
    print("\nCross-validation summary:")
    print(tuning_df.to_string(index=False))
    print("\nHold-out test metrics:")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nAll output files are saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
