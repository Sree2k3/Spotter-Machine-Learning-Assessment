from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import predict_rates, train_full_model


TRAIN_PATH = Path("data/train_test.csv")
VALIDATION_PATH = Path("data/validation.csv")
VALIDATION_TEMPLATE_PATH = Path("data/validation_predictions_template.csv")
DECEMBER_INPUT_PATH = Path("data/december_chart_inputs.csv")
VALIDATION_OUTPUT_PATH = Path("validation_predictions.csv")
PREDICTION_COLUMN = "predicted_rate"


def _positive_rates(predictions: np.ndarray) -> np.ndarray:
    """Keep generated rates positive and rounded to cents."""
    return np.maximum(predictions, 0.01).round(2)


def create_validation_predictions(
    validation: pd.DataFrame,
    template: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build the final validation_predictions.csv content."""
    required_columns = ["load_id", PREDICTION_COLUMN]
    if list(template.columns) != required_columns:
        raise ValueError(
            "Validation template must contain exactly load_id,predicted_rate."
        )

    if len(validation) != len(template):
        raise ValueError("Validation data and prediction template row counts differ.")

    if not validation["load_id"].astype(str).equals(template["load_id"].astype(str)):
        raise ValueError("Validation load_id values do not match the template order.")

    output = template.copy()
    output[PREDICTION_COLUMN] = _positive_rates(predictions)

    return output[required_columns]


def fill_december_predictions(
    december_inputs: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Fill predicted_rate for the fixed December chart inputs."""
    output = december_inputs.copy()
    output[PREDICTION_COLUMN] = _positive_rates(predictions)

    return output[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            PREDICTION_COLUMN,
        ]
    ]


def run_predictions(
    train_path: str | Path = TRAIN_PATH,
    validation_path: str | Path = VALIDATION_PATH,
    validation_template_path: str | Path = VALIDATION_TEMPLATE_PATH,
    december_input_path: str | Path = DECEMBER_INPUT_PATH,
    validation_output_path: str | Path = VALIDATION_OUTPUT_PATH,
) -> None:
    """Train on all labeled data and write the required prediction files."""
    model = train_full_model(train_path)

    validation = pd.read_csv(validation_path)
    template = pd.read_csv(validation_template_path)
    validation_predictions = predict_rates(model, validation)
    validation_output = create_validation_predictions(
        validation=validation,
        template=template,
        predictions=validation_predictions,
    )
    validation_output.to_csv(validation_output_path, index=False)

    december_inputs = pd.read_csv(december_input_path)
    december_predictions = predict_rates(model, december_inputs)
    december_output = fill_december_predictions(
        december_inputs=december_inputs,
        predictions=december_predictions,
    )
    december_output.to_csv(december_input_path, index=False)

    print(f"Wrote {validation_output_path}")
    print(f"Filled {december_input_path}")


def main() -> None:
    run_predictions()


if __name__ == "__main__":
    main()
