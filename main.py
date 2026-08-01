from pathlib import Path

from score import (
    EXPECTED_ROWS,
    read_csv,
    save_december_chart,
    validate_december,
    validate_predictions,
)
from src.model import train_and_validate
from src.predict import DECEMBER_INPUT_PATH, VALIDATION_OUTPUT_PATH, run_predictions


SCORER_OUTPUT_DIR = Path("scorer_results")
DECEMBER_CHART_PATH = SCORER_OUTPUT_DIR / "candidate_december.png"


def print_stage(step: int, total: int, title: str) -> None:
    print(f"\n[{step}/{total}] {title}")


def print_metrics(metrics: dict[str, float]) -> None:
    print(f"Train rows: {int(metrics['train_rows'])}")
    print(f"Validation rows: {int(metrics['validation_rows'])}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Median AE: {metrics['median_ae']:.2f}")


def validate_outputs() -> None:
    predictions = read_csv(VALIDATION_OUTPUT_PATH, "predictions")
    validate_predictions(predictions)

    december = validate_december(read_csv(DECEMBER_INPUT_PATH, "December predictions"))
    SCORER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_december_chart(december, DECEMBER_CHART_PATH)

    print(f"Validated {EXPECTED_ROWS:,} final predictions.")
    print("Validated 31 fixed December predictions.")
    print(f"Created chart: {DECEMBER_CHART_PATH}")


def main() -> None:
    print("Freight Rate ML Pipeline")

    print_stage(1, 3, "Train and validate model")
    _, metrics = train_and_validate()
    print_metrics(metrics)

    print_stage(2, 3, "Train final model and write predictions")
    run_predictions()

    print_stage(3, 3, "Validate submission files and create chart")
    validate_outputs()

    print("\nPipeline complete.")
    print(f"Ready file: {VALIDATION_OUTPUT_PATH}")
    print(f"Ready file: {DECEMBER_INPUT_PATH}")
    print(f"Ready chart: {DECEMBER_CHART_PATH}")


if __name__ == "__main__":
    main()
