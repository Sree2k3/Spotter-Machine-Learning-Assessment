# Freight Rate Prediction Assessment

This project trains a machine learning model to predict freight `posted_rate` values.
It uses the labeled development data in `data/train_test.csv`, generates predictions
for `data/validation.csv`, fills the fixed December chart input file, and validates
the required submission files with the provided `score.py`.

## Project Structure

```text
.
|-- data/
|   |-- train_test.csv
|   |-- validation.csv
|   |-- validation_predictions_template.csv
|   `-- december_chart_inputs.csv
|-- src/
|   |-- freight_rate_analysis.ipynb
|   |-- preprocess.py
|   |-- model.py
|   `-- predict.py
|-- main.py
|-- score.py
|-- requirements.txt
`-- validation_predictions.csv
```

## Model Summary

The final model is a blended regression ensemble:

- Ridge Regression with one-hot encoded location, equipment, and lane features
- HistGradientBoostingRegressor for nonlinear numeric and categorical patterns

The labeled data is split by time for validation:

- Training: January 2025 through August 2025
- Validation: September 2025 through October 2025

Current validation results:

```text
MAE: 124.58
RMSE: 635.73
R2: 0.8265
MAPE: 6.38%
Median AE: 45.27
```

## Setup

Run these commands from the project root.

### Git Bash

```bash
uv venv --python 3.12
source .venv/Scripts/activate
uv pip install -r requirements.txt
```

### PowerShell

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

If `python` points to Anaconda or another environment, use the explicit project
Python command shown below.

## Run The Full Pipeline

If the virtual environment is activated:

```bash
python main.py
```

If activation is not working or Python points to the wrong environment:

```bash
./.venv/Scripts/python.exe main.py
```

`main.py` runs the project in three stages:

```text
[1/3] Train and validate model
[2/3] Train final model and write predictions
[3/3] Validate submission files and create chart
```

It writes:

```text
validation_predictions.csv
data/december_chart_inputs.csv
scorer_results/candidate_december.png
```

## Run Only Model Validation

```bash
python -m src.model
```

or:

```bash
./.venv/Scripts/python.exe -m src.model
```

## Generate Predictions Only

```bash
python -m src.predict
```

or:

```bash
./.venv/Scripts/python.exe -m src.predict
```

## Validate Submission Files

After running the pipeline, validate the output files with:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

or:

```bash
./.venv/Scripts/python.exe score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Expected scorer output:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

## Submission Checklist

Submit:

- An accessible GitHub repository with code, dependencies, and run instructions
- `validation_predictions.csv` with exactly `load_id,predicted_rate`
- A PDF or DOCX report containing the validation approach and the December chart
- A 2-3 minute Loom video covering EDA findings, data-quality handling, model choice, validation split, and code walkthrough

The generated December chart for the report is:

```text
scorer_results/candidate_december.png
```
