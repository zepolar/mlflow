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


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "wine-lr"
ARTIFACTS_DIR = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple Wine classifier and log with MLflow"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength for LogisticRegression",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Max iterations for LogisticRegression",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test size fraction for the split",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    wine = datasets.load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    targets = pd.Series(wine.target, name="target")
    class_map = {int(i): str(name) for i, name in enumerate(wine.target_names)}
    return features, targets, class_map


def compute_class_counts(targets: pd.Series) -> np.ndarray:
    """Compute per-class sample counts."""
    return np.bincount(targets.astype(int))


def split_data(
    features: pd.DataFrame,
    targets: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and test subsets."""
    return train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )


def build_pipeline(C: float, max_iter: int, random_state: int) -> Pipeline:
    """Create the preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    multi_class="auto",
                ),
            ),
        ]
    )


def configure_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> None:
    """
    Configure MLflow tracking and experiment.
    Uses a local SQLite backend to avoid deprecated filesystem registry warnings.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Fit the model and compute evaluation metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    return {"accuracy": accuracy, "f1_macro": f1_macro}


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    # Create local artifacts folder (optional; MLflow will manage artifacts, but we may save extras)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    features, targets, class_map = load_data()
    class_counts = compute_class_counts(targets)

    X_train, X_test, y_train, y_test = split_data(
        features,
        targets,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    pipeline = build_pipeline(
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    configure_mlflow()

    with mlflow.start_run():
        # Log parameters and some data stats
        mlflow.log_params(
            {
                "C": args.C,
                "max_iter": args.max_iter,
                "test_size": args.test_size,
                "random_state": args.random_state,
            }
        )
        mlflow.log_dict(class_map, "class_map.json")
        mlflow.log_dict({"class_counts": class_counts.tolist()}, "class_counts.json")

        metrics = train_and_evaluate(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_macro", metrics["f1_macro"])

        # Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()