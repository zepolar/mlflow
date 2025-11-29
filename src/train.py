import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple Wine classifier and log with MLflow")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse of regularization strength for LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations for LogisticRegression")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size fraction for the split")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    wine = datasets.load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")
    class_map = {int(i): str(name) for i, name in enumerate(wine.target_names)}
    return X, y, class_map


def main():
    args = parse_args()

    # Create local artifacts folder (optional; MLflow will manage artifacts, but we may save extras)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y, class_map = load_data()

    # Simple numpy usage example: class counts
    class_counts = np.bincount(y.astype(int))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=args.random_state, multi_class="auto"),
            ),
        ]
    )

    # Use a SQLite backend for tracking/registry to avoid deprecated filesystem registry warnings
    # This creates a local file 'mlflow.db' in the project root.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")

    # Define name of experiment
    mlflow.set_experiment("wine-logic-regression")

    with mlflow.start_run():
        # Log parameters and some data stats
        mlflow.log_params({
            "C": args.C,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "random_state": args.random_state,
        })

        mlflow.log_dict(class_map, "class_map.json")
        mlflow.log_dict({"class_counts": class_counts.tolist()}, "class_counts.json")

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro"))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1m)

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(json.dumps({"accuracy": acc, "f1_macro": f1m}, indent=2))


if __name__ == "__main__":
    main()
