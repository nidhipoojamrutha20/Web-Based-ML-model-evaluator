# Machine Learning Model Training Module
#
# US-17: Implement ML algorithm selection (Assigned to: @gaganraghavan) -- DONE
# US-18: Implement model training (Assigned to: @aryannang123)

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class ModelTrainer:
    """Trains classification and regression models using scikit-learn."""

    SUPPORTED_ALGORITHMS = [
        "logistic_regression",
        "linear_regression",
        "decision_tree",
        "random_forest",
        "knn",
    ]

    ALGORITHM_CATALOG = {
        "classification": [
            {"id": "logistic_regression", "label": "Logistic Regression", "description": "Linear classifier for categorical targets."},
            {"id": "decision_tree", "label": "Decision Tree", "description": "Interpretable tree-based classifier."},
            {"id": "random_forest", "label": "Random Forest", "description": "Ensemble of randomized decision trees."},
            {"id": "knn", "label": "K-Nearest Neighbors", "description": "Classifies using nearby training samples."},
        ],
        "regression": [
            {"id": "linear_regression", "label": "Linear Regression", "description": "Linear model for numeric targets."},
            {"id": "decision_tree", "label": "Decision Tree Regressor", "description": "Tree-based regressor for numeric targets."},
            {"id": "random_forest", "label": "Random Forest Regressor", "description": "Ensemble regressor using randomized trees."},
            {"id": "knn", "label": "K-Nearest Neighbors Regressor", "description": "Predicts from nearby training samples."},
        ],
    }

    _ESTIMATOR_MAP = {
        ("logistic_regression", "classification"): LogisticRegression,
        ("linear_regression",   "regression"):     LinearRegression,
        ("decision_tree",       "classification"): DecisionTreeClassifier,
        ("decision_tree",       "regression"):     DecisionTreeRegressor,
        ("random_forest",       "classification"): RandomForestClassifier,
        ("random_forest",       "regression"):     RandomForestRegressor,
        ("knn",                 "classification"): KNeighborsClassifier,
        ("knn",                 "regression"):     KNeighborsRegressor,
    }

    @classmethod
    def algorithms_for(cls, task_type=None):
        """Return a copy of the supported algorithm catalog."""
        if task_type is None:
            return {task: [dict(item) for item in entries] for task, entries in cls.ALGORITHM_CATALOG.items()}
        if task_type not in cls.ALGORITHM_CATALOG:
            raise ValueError("task_type must be classification or regression.")
        return [dict(item) for item in cls.ALGORITHM_CATALOG[task_type]]

    @classmethod
    def create_estimator(cls, algorithm_name: str, task_type: str):
        algorithm_key = str(algorithm_name).lower().strip()
        task_key = str(task_type).lower().strip()

        if algorithm_key not in cls.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm_name}'. "
                f"Supported algorithms: {', '.join(cls.SUPPORTED_ALGORITHMS)}"
            )

        if task_key not in ("classification", "regression"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Accepted values are 'classification' or 'regression'."
            )

        if task_key == "classification":
            estimators = {
                "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
                "decision_tree": DecisionTreeClassifier(random_state=42),
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "knn": KNeighborsClassifier(n_neighbors=5),
            }
        else:
            estimators = {
                "linear_regression": LinearRegression(),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "knn": KNeighborsRegressor(n_neighbors=5),
            }

        estimator = estimators.get(algorithm_key)
        if estimator is None:
            raise ValueError(
                f"Algorithm '{algorithm_name}' is not compatible with task type '{task_type}'. "
                "For example, 'logistic_regression' supports classification only and "
                "'linear_regression' supports regression only."
            )
        return estimator

    @classmethod
    def train(cls, algorithm_name: str, task_type: str, X_train, y_train, **kwargs):
        """Fit a supported estimator with preprocessing learned from training data (US-18)."""
        algorithm_key = str(algorithm_name).lower().strip()
        task_key = str(task_type).lower().strip()

        if algorithm_key not in cls.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm_name}'. "
                f"Supported algorithms: {', '.join(cls.SUPPORTED_ALGORITHMS)}"
            )

        if task_key not in ("classification", "regression"):
            raise ValueError(
                f"Unsupported task_type '{task_type}'. "
                "Accepted values are 'classification' or 'regression'."
            )

        estimator = cls.create_estimator(algorithm_key, task_key)

        from ml.preprocessor import DataPreprocessor

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(
                X_train, columns=[f"col_{i}" for i in range(X_train.shape[1])]
            )

        scale_numeric = algorithm_key in {
            "logistic_regression", "linear_regression", "knn"
        }
        pipeline = DataPreprocessor.build_pipeline(
            X_train, estimator, scale_numeric=scale_numeric
        )
        pipeline.fit(X_train, y_train)
        return pipeline

    @staticmethod
    def save(model, filepath: str):
        """Serialize model weights to *filepath* using joblib.

        Parameters
        ----------
        model : fitted sklearn estimator
        filepath : str
            Absolute or relative path where the model file will be written.
            Parent directories are created automatically.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(model, filepath)
