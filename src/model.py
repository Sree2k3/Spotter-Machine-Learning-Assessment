from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from src.preprocess import (
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_data,
    split_features_target,
)


DEFAULT_TRAIN_PATH = Path("data/train_test.csv")
VALIDATION_START_DATE = "2025-09-01"
TREE_CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]


def build_ridge_model(alpha: float = 100.0) -> Pipeline:
    """Build a regularized linear model with one-hot encoded categories."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def build_tree_model() -> Pipeline:
    """Build a nonlinear model for city/equipment and numeric interactions."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, TREE_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    categorical_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(
        TREE_CATEGORICAL_FEATURES
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    categorical_features=categorical_mask,
                    random_state=42,
                ),
            ),
        ]
    )


def build_model(alpha: float = 100.0) -> VotingRegressor:
    """Build the blended final model used for validation and prediction."""
    return VotingRegressor(
        estimators=[
            ("ridge", build_ridge_model(alpha=alpha)),
            ("tree", build_tree_model()),
        ],
        weights=[0.5, 0.5],
    )


def time_based_split(
    df: pd.DataFrame,
    validation_start_date: str = VALIDATION_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train_test data into past training rows and future holdout rows."""
    df = clean_data(df)
    dates = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp(validation_start_date)

    train_df = df.loc[dates < cutoff].copy()
    valid_df = df.loc[dates >= cutoff].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Check validation_start_date.")

    if valid_df.empty:
        raise ValueError("Validation split is empty. Check validation_start_date.")

    return train_df, valid_df


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics for model validation."""
    mse = mean_squared_error(y_true, y_pred)
    nonzero_mask = y_true != 0

    if nonzero_mask.any():
        mape = (
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
        "mape": float(mape),
        "median_ae": median_absolute_error(y_true, y_pred),
    }


def predict_rates(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict freight rates and keep outputs physically valid."""
    X, _ = split_features_target(df)
    predictions = model.predict(X)

    return np.maximum(predictions, 0.0)


def train_and_validate(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
) -> tuple[Pipeline, dict[str, float]]:
    """Train on January-August and validate on September-October."""
    data = pd.read_csv(train_path)
    train_df, valid_df = time_based_split(data)

    X_train, y_train = split_features_target(train_df)
    X_valid, y_valid = split_features_target(valid_df)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_valid), 0.0)
    metrics = evaluate_predictions(y_valid, predictions)

    metrics["train_rows"] = float(len(train_df))
    metrics["validation_rows"] = float(len(valid_df))

    return model, metrics


def train_full_model(train_path: str | Path = DEFAULT_TRAIN_PATH) -> Pipeline:
    """Train final model on all labeled data before final predictions."""
    data = pd.read_csv(train_path)
    X, y = split_features_target(data)

    model = build_model()
    model.fit(X, y)

    return model


def main() -> None:
    _, metrics = train_and_validate()

    print("Validation results")
    print(f"Train rows: {int(metrics['train_rows'])}")
    print(f"Validation rows: {int(metrics['validation_rows'])}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Median AE: {metrics['median_ae']:.2f}")


if __name__ == "__main__":
    main()
