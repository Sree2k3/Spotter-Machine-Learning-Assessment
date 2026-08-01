import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day",
    "day_of_week",
]

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "lane",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe without modifying the original data."""
    df = df.copy()

    for col in ["pickup", "delivery", "equipment"]:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = df[col].astype("string").fillna("Unknown").str.strip()

    df["lane"] = df["pickup"].astype(str) + "_to_" + df["delivery"].astype(str)

    if "date" not in df.columns:
        df["date"] = pd.NaT

    date_values = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = date_values.dt.month
    df["day"] = date_values.dt.day
    df["day_of_week"] = date_values.dt.dayofweek

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["weight"] = df["weight"].abs()

    return df


def split_features_target(df: pd.DataFrame):
    """Return model features X and target y if target exists."""
    df = clean_data(df)

    X = df[FEATURE_COLUMNS]

    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN]
    else:
        y = None

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
