"""Preprocessing: one ColumnTransformer shared by training and inference."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SEED,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics, one-hot encode season, impute binary flag.

    Nothing is fitted here; fitting happens inside a Pipeline on training data only.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", binary, BINARY_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Fixed, stratified train/test split (random_state = seed) on the cleaned table."""
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=df[TARGET_CLASSIFICATION],
    )
    train, test = df.loc[train_idx], df.loc[test_idx]
    return {
        "train": train,
        "test": test,
        "X_train": train[FEATURES],
        "X_test": test[FEATURES],
        "y_reg_train": train[TARGET_REGRESSION],
        "y_reg_test": test[TARGET_REGRESSION],
        "y_clf_train": train[TARGET_CLASSIFICATION],
        "y_clf_test": test[TARGET_CLASSIFICATION],
    }
