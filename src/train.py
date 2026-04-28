import argparse
import json
import logging
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = ["Age", "Gender", "family_history", "benefits", "care_options",
                "anonymity", "leave", "work_interfere"]
CATEGORICAL_COLS = ["Gender", "family_history", "benefits", "care_options",
                    "anonymity", "leave", "work_interfere"]
NUMERIC_COLS = ["Age"]


def _compute_baseline(df: pd.DataFrame) -> dict:
    baseline = {}
    for col in FEATURE_COLS:
        if col in NUMERIC_COLS:
            baseline[col] = {
                "type": "numeric",
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            baseline[col] = {
                "type": "categorical",
                "value_counts": df[col].value_counts(normalize=True).to_dict(),
            }
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/survey_cleaned.csv")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.30)
    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)

    df = df.dropna(subset=["treatment"])
    X = df[FEATURE_COLS]
    y = df["treatment"].apply(lambda x: 1 if x == "Yes" else 0)
    logger.info("Dataset: %d samples, %d features", len(X), len(FEATURE_COLS))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=0
    )

    # Save training-set baseline for drift detection
    baseline = _compute_baseline(X_train)
    baseline_path = "data/baseline_stats.json"
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Baseline statistics saved to %s", baseline_path)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), NUMERIC_COLS),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_COLS),
        ]
    )

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=args.max_depth)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(
            estimator=clf, n_estimators=args.n_estimators, random_state=42
        )),
    ])

    mlflow.set_experiment("mindcare-oracle-treatment")

    with mlflow.start_run():
        logger.info("Starting MLflow run")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("model_type", "AdaBoost")
        mlflow.log_param("base_estimator", f"DecisionTree(depth={args.max_depth},entropy)")
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        logger.info("Training model with n_estimators=%d", args.n_estimators)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("true_positives", int(tp))
        mlflow.log_metric("false_positives", int(fp))

        logger.info("Accuracy=%.4f  ROC-AUC=%.4f  F1=%.4f", acc, auc, f1)

        model_info = mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(baseline_path)
        logger.info("Model and baseline artifact logged to MLflow")

        # Register model version in MLflow Model Registry
        try:
            mlflow.register_model(model_info.model_uri, "mindcare-oracle-treatment")
            logger.info("Model registered in MLflow Model Registry as 'mindcare-oracle-treatment'")
        except Exception as exc:
            logger.warning("Model registry not available (OK in file-tracking mode): %s", exc)

        # Write metrics.json for DVC metrics tracking
        metrics_out = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
        metrics_path = "data/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info("Metrics written to %s: %s", metrics_path, metrics_out)


if __name__ == "__main__":
    main()
