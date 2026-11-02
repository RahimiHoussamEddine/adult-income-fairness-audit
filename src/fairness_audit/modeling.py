"""Group-blind model construction and threshold metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness_audit.constants import SAMPLING_COLUMN, SENSITIVE_COLUMNS, TARGET_COLUMN

NUMERIC_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
PROXY_COLUMNS = ["marital-status", "relationship"]


def feature_columns(
    columns: Sequence[str], *, remove_proxy_fields: bool = False
) -> list[str]:
    excluded = set(SENSITIVE_COLUMNS) | {SAMPLING_COLUMN, TARGET_COLUMN}
    if remove_proxy_fields:
        excluded.update(PROXY_COLUMNS)
    selected = [column for column in columns if column not in excluded]
    if set(selected) & (set(SENSITIVE_COLUMNS) | {SAMPLING_COLUMN, TARGET_COLUMN}):
        raise AssertionError(
            "Target, sensitive fields, and sampling weights stay excluded"
        )
    return selected


def prepare_features(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    features = frame.loc[:, list(columns)].copy()
    for column in NUMERIC_COLUMNS:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="raise")
    for column in features.columns:
        if column not in NUMERIC_COLUMNS:
            values = features[column].astype("object")
            features[column] = values.where(pd.notna(values), np.nan)
    return features


def build_model(columns: Sequence[str]) -> Pipeline:
    numeric = [column for column in columns if column in NUMERIC_COLUMNS]
    categorical = [column for column in columns if column not in NUMERIC_COLUMNS]
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers=transformers)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        solver="lbfgs",
        random_state=2026,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
