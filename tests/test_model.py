"""Unit tests for the ML pipeline (train.py components)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


FEATURE_COLS = ["Age", "Gender", "family_history", "benefits",
                "care_options", "anonymity", "leave", "work_interfere"]


def _make_training_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Age": rng.integers(20, 60, n),
        "Gender": rng.choice(["male", "female"], n),
        "family_history": rng.choice(["Yes", "No"], n),
        "benefits": rng.choice(["Yes", "No", "Don't know"], n),
        "care_options": rng.choice(["Yes", "No", "Not sure"], n),
        "anonymity": rng.choice(["Yes", "No", "Don't know"], n),
        "leave": rng.choice(["Very easy", "Somewhat easy", "Don't know",
                              "Somewhat difficult", "Very difficult"], n),
        "work_interfere": rng.choice(["Often", "Rarely", "Never", "Sometimes"], n),
        "treatment": rng.choice(["Yes", "No"], n),
    })


def test_feature_columns_all_present():
    df = _make_training_df()
    for col in FEATURE_COLS:
        assert col in df.columns


def test_pipeline_trains_and_predicts():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

    df = _make_training_df(40)
    X = df[FEATURE_COLS]
    y = df["treatment"].apply(lambda v: 1 if v == "Yes" else 0)

    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), ["Age"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         [c for c in FEATURE_COLS if c != "Age"]),
    ])
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=1)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(estimator=clf, n_estimators=10, random_state=42)),
    ])
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_pipeline_predict_proba_shape():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

    df = _make_training_df(40)
    X = df[FEATURE_COLS]
    y = df["treatment"].apply(lambda v: 1 if v == "Yes" else 0)

    preprocessor = ColumnTransformer(transformers=[
        ("num", MinMaxScaler(), ["Age"]),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         [c for c in FEATURE_COLS if c != "Age"]),
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=10, random_state=42
        )),
    ])
    pipeline.fit(X, y)

    probas = pipeline.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_drift_detection_numeric():
    from drift_detection import check_drift

    baseline = {
        "Age": {
            "type": "numeric",
            "mean": 30.0,
            "std": 5.0,
            "min": 20.0,
            "max": 60.0,
        }
    }
    # Same distribution — should not drift
    records_ok = [{"Age": 30}, {"Age": 31}, {"Age": 29}]
    result_ok = check_drift(records_ok, baseline)
    assert not result_ok["is_drifted"]

    # Very different distribution — should drift (mean=60, z-score=6)
    records_bad = [{"Age": 60}, {"Age": 65}, {"Age": 55}]
    result_bad = check_drift(records_bad, baseline)
    assert result_bad["is_drifted"]


def test_drift_detection_categorical():
    from drift_detection import check_drift

    baseline = {
        "Gender": {
            "type": "categorical",
            "value_counts": {"male": 0.8, "female": 0.2},
        }
    }
    # Same distribution
    records = [{"Gender": "male"}] * 8 + [{"Gender": "female"}] * 2
    result = check_drift(records, baseline)
    assert not result["is_drifted"]
