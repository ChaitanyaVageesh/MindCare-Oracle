import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", input_path)
    train_df = pd.read_csv(input_path)
    logger.info("Loaded %d rows, %d columns", *train_df.shape)

    logger.info("Dropping unnecessary columns")
    cols_to_drop = ["comments", "state", "Timestamp", "Country"]
    for c in cols_to_drop:
        if c in train_df.columns:
            train_df.drop([c], axis=1, inplace=True)

    logger.info("Filling NaN values")
    defaultInt = 0
    defaultString = "NaN"

    intFeatures = ["Age"]
    stringFeatures = [
        "Gender", "self_employed", "family_history", "treatment",
        "work_interfere", "no_employees", "remote_work", "tech_company",
        "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor",
        "mental_health_interview", "phys_health_interview",
        "mental_vs_physical", "obs_consequence", "benefits",
        "care_options", "wellness_program", "seek_help",
    ]

    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)

    logger.info("Normalising Gender values")
    male_str = [
        "male", "m", "male-ish", "maile", "mal", "male (cis)", "make",
        "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male",
    ]
    trans_str = [
        "trans-female", "something kinda male?", "queer/she/they",
        "non-binary", "nah", "all", "enby", "fluid", "genderqueer",
        "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^",
        "trans woman", "neuter", "female (trans)", "queer",
        "ostensibly male, unsure what that really means",
    ]
    female_str = [
        "cis female", "f", "female", "woman", "femake", "female ",
        "cis-female/femme", "female (cis)", "femail",
    ]

    train_df["Gender"] = train_df["Gender"].str.lower()
    train_df["Gender"] = train_df["Gender"].apply(
        lambda x: "male" if x in male_str
        else ("female" if x in female_str else ("trans" if x in trans_str else x))
    )

    stk_list = ["a little about you", "p"]
    before = len(train_df)
    train_df = train_df[~train_df["Gender"].isin(stk_list)]
    logger.info("Removed %d junk gender rows", before - len(train_df))

    logger.info("Cleaning Age outliers")
    median_age = train_df["Age"].median()
    train_df["Age"].fillna(median_age, inplace=True)
    train_df.loc[train_df["Age"] < 18, "Age"] = median_age
    train_df.loc[train_df["Age"] > 120, "Age"] = median_age

    logger.info("Fixing categorical NaN strings")
    train_df["self_employed"] = train_df["self_employed"].replace([defaultString], "No")
    train_df["work_interfere"] = train_df["work_interfere"].replace([defaultString], "Don't know")

    train_df["age_range"] = pd.cut(
        train_df["Age"],
        [0, 20, 30, 65, 100],
        labels=["0-20", "21-30", "31-65", "66-100"],
        include_lowest=True,
    )

    logger.info("Saving processed data to %s", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    train_df.to_csv(output_path, index=False)
    logger.info("Preprocessing complete — %d rows written", len(train_df))
    return train_df


if __name__ == "__main__":
    clean_data("data/survey.csv", "data/processed/survey_cleaned.csv")
