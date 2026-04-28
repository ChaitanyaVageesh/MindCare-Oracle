"""Unit tests for src/data_preprocessing.py"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preprocessing import clean_data


def _make_survey_df(**overrides) -> pd.DataFrame:
    """Return a minimal valid survey row as a DataFrame."""
    row = {
        "Age": 28,
        "Gender": "Male",
        "self_employed": "No",
        "family_history": "No",
        "treatment": "Yes",
        "work_interfere": "Sometimes",
        "no_employees": "6-25",
        "remote_work": "No",
        "tech_company": "Yes",
        "anonymity": "Yes",
        "leave": "Somewhat easy",
        "mental_health_consequence": "No",
        "phys_health_consequence": "No",
        "coworkers": "Some of them",
        "supervisor": "Yes",
        "mental_health_interview": "No",
        "phys_health_interview": "Maybe",
        "mental_vs_physical": "Yes",
        "obs_consequence": "No",
        "benefits": "Yes",
        "care_options": "Not sure",
        "wellness_program": "No",
        "seek_help": "Yes",
        "Timestamp": "2024-01-01",
        "Country": "United States",
        "state": "CA",
        "comments": "",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_gender_male_variants(tmp_path):
    """Various Male strings must normalise to 'male'."""
    for variant in ["M", "male", "Male", "MALE", "Cis Male", "man"]:
        df = _make_survey_df(Gender=variant)
        in_csv = tmp_path / "in.csv"
        out_csv = tmp_path / "out.csv"
        df.to_csv(in_csv, index=False)
        result = clean_data(str(in_csv), str(out_csv))
        assert result["Gender"].iloc[0] == "male", f"Failed for '{variant}'"


def test_gender_female_variants(tmp_path):
    for variant in ["female", "F", "Woman", "femail", "cis female"]:
        df = _make_survey_df(Gender=variant)
        in_csv = tmp_path / "in.csv"
        out_csv = tmp_path / "out.csv"
        df.to_csv(in_csv, index=False)
        result = clean_data(str(in_csv), str(out_csv))
        assert result["Gender"].iloc[0] == "female", f"Failed for '{variant}'"


def test_age_outliers_replaced_with_median(tmp_path):
    rows = [_make_survey_df(Age=25), _make_survey_df(Age=5), _make_survey_df(Age=200)]
    df = pd.concat(rows, ignore_index=True)
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)
    result = clean_data(str(in_csv), str(out_csv))
    # All ages should be in [18, 120]
    assert all(18 <= a <= 120 for a in result["Age"]), result["Age"].tolist()


def test_nan_self_employed_filled(tmp_path):
    df = _make_survey_df(self_employed=float("nan"))
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)
    result = clean_data(str(in_csv), str(out_csv))
    assert result["self_employed"].iloc[0] == "No"


def test_output_file_created(tmp_path):
    df = _make_survey_df()
    in_csv = tmp_path / "survey.csv"
    out_csv = tmp_path / "processed" / "out.csv"
    df.to_csv(in_csv, index=False)
    clean_data(str(in_csv), str(out_csv))
    assert out_csv.exists()


def test_unnecessary_columns_dropped(tmp_path):
    df = _make_survey_df()
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)
    result = clean_data(str(in_csv), str(out_csv))
    for col in ["Timestamp", "Country", "state", "comments"]:
        assert col not in result.columns, f"Column '{col}' should have been dropped"


def test_age_range_column_created(tmp_path):
    df = _make_survey_df(Age=25)
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    df.to_csv(in_csv, index=False)
    result = clean_data(str(in_csv), str(out_csv))
    assert "age_range" in result.columns
    assert result["age_range"].iloc[0] == "21-30"
