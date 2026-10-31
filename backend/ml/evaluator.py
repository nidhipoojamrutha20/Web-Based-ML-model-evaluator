"""Persisted evaluation metrics for classification and regression models."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ModelEvaluator:
    """Compute comparable metrics and classification details."""

    @staticmethod
    def evaluate(model, X_test, y_test, task_type: str):
        predictions = model.predict(X_test)
        if task_type == "classification":
            precision = float(precision_score(y_test, predictions, average="weighted", zero_division=0))
            recall = float(recall_score(y_test, predictions, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                # US-20 chart names and US-19's explicit weighted names share the
                # same persisted weighted-average values.
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
            }
            return metrics, confusion_matrix(y_test, predictions).tolist(), classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )

        if task_type == "regression":
            mse = float(mean_squared_error(y_test, predictions))
            score = float(r2_score(y_test, predictions))
            return {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": score if math.isfinite(score) else None,
            }, None, None
        raise ValueError("Task type must be classification or regression.")

    @staticmethod
    def generate_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred).tolist()
