FlowML — Minimal MLflow example project (pandas, numpy, scikit-learn)

This project demonstrates how to set up MLflow and run a small training job on a public dataset using pandas, numpy, and scikit-learn. It logs parameters, metrics, and a trained model into the local `mlruns/` directory.

What it does
- Loads the Wine dataset from scikit-learn
- Uses pandas/numpy for data handling
- Trains a simple pipeline: `StandardScaler` + `LogisticRegression`
- Logs parameters, metrics (accuracy, F1-macro), and the model to MLflow

Prerequisites
- Python 3.10+ recommended (3.11 tested)
- Windows PowerShell

Option A: Quickstart with pip (recommended)
1) Create and activate a virtual environment
   PowerShell:
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1

2) Install dependencies
   - pip install -r requirements.txt

3) Run training
   - python src/train.py --C 1.0 --max_iter 200 --test_size 0.2 --random_state 42

4) View MLflow UI
   In a separate terminal with the same venv activated:
   - python -m mlflow ui --port 5000 --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root ".\\mlruns"
   Open: http://127.0.0.1:5000

Option B: Run as an MLflow Project (with Conda)
If you have Conda installed, MLflow can build and run in an isolated environment automatically.
   - mlflow run . -P C=1.0 -P max_iter=200 -P test_size=0.2 -P random_state=42

Key Files
- src/train.py: Training script that logs to MLflow
- requirements.txt: Minimal dependencies
- conda.yaml: Environment spec for `mlflow run`
- MLproject: MLflow Project spec enabling parameterized runs
- .gitignore: Typical Python ignores and `mlruns/`

Outputs
- Tracking/Model Registry backend: local SQLite DB at `mlflow.db` (configured in code)
- Artifacts (models, dicts, etc.) stored under: `mlruns/`
- Each run contains params, metrics, and the serialized scikit-learn model

Troubleshooting
 - Filesystem model registry backend is deprecated by MLflow; this project now uses a local SQLite database (`sqlite:///mlflow.db`) to avoid the warning.
 - If `mlflow` command is not recognized on Windows, use `python -m mlflow ...` (works even if the script isn’t on PATH). Example: `python -m mlflow ui --port 5000 --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root ".\\mlruns"`
 - If port 5000 is in use, choose a different port with `--port 5001` (and open that URL)
