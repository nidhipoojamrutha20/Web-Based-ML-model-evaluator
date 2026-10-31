import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, classification_report)

class ModelEvaluator:
    """Computes evaluation metrics and generates comparison data."""

    @staticmethod
    def evaluate(model, X_test, y_test, task_type: str):
        """Calculate Accuracy, Precision, Recall, F1 or MSE, RMSE, R2 (US-19)."""
        predictions = model.predict(X_test)
        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            }
            return metrics, confusion_matrix(y_test, predictions).tolist(), classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
        if task_type == "regression":
            mse = float(mean_squared_error(y_test, predictions))
            return {
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2_score(y_test, predictions)),
            }, None, None
        raise ValueError("Task type must be classification or regression.")

    @staticmethod
    def generate_confusion_matrix(y_true, y_pred):
        """Generate confusion matrix array for classification tasks (US-19)."""
        return confusion_matrix(y_true, y_pred).tolist()
