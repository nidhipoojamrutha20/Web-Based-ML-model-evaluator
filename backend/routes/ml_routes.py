import io
import hashlib
import math
import os
import textwrap
import uuid

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split

from database.db import db
from ml.evaluator import ModelEvaluator
from ml.trainer import ModelTrainer
from models.dataset import Dataset
from models.evaluation import EvaluationResult
from models.model import MLModel
from routes.auth_utils import (
    get_owned_dataset, get_owned_evaluation, get_owned_model,
    login_required, owned_evaluation_query, owned_model_query,
)

ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")
CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1")
REGRESSION_METRICS = ("mae", "mse", "rmse", "r2")


def _model_response(model):
    return {
        "id": model.id, "dataset_id": model.dataset_id,
        "dataset_name": model.dataset.filename if model.dataset else None,
        "model_name": model.model_name, "algorithm": model.algorithm,
        "run_label": f"{model.model_name} · Run #{model.id}",
        "status": model.status, "configuration": model.config_json,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "evaluation_id": model.evaluations[-1].id if model.evaluations else None,
        "metrics": model.evaluations[-1].metrics_json if model.evaluations else None,
    }


def _evaluation_response(result):
    return {
        "id": result.id, "model_id": result.model_id,
        "algorithm": result.model.algorithm, "metrics": result.metrics_json,
        "confusion_matrix": result.confusion_matrix_json,
        "report": result.report_json,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


def _safe_model_path(model):
    root = os.path.realpath(os.path.join(current_app.config["SAVED_MODELS_FOLDER"], str(model.user_id)))
    path = os.path.realpath(model.model_path or "")
    try:
        safe = os.path.commonpath([root, path]) == root and path != root
    except ValueError:
        safe = False
    if not safe:
        return None
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@ml_bp.route("/algorithms", methods=["GET"])
@login_required
def get_algorithms():
    return jsonify({
        "classification": ["logistic_regression", "decision_tree", "random_forest", "knn"],
        "regression": ["linear_regression", "decision_tree", "random_forest", "knn"],
    }), 200


@ml_bp.route("/models", methods=["GET"])
@login_required
def list_models():
    models = owned_model_query(g.current_user.id).order_by(
        MLModel.created_at.desc(), MLModel.id.desc()
    ).all()
    return jsonify({"models": [_model_response(item) for item in models]}), 200


@ml_bp.route("/models/<int:model_id>", methods=["GET"])
@login_required
def get_model(model_id):
    model = get_owned_model(model_id, g.current_user.id)
    if model is None:
        return jsonify({"error": "Model not found"}), 404
    return jsonify({"model": _model_response(model)}), 200


@ml_bp.route("/evaluations", methods=["GET"])
@login_required
def list_evaluations():
    results = owned_evaluation_query(g.current_user.id).order_by(
        EvaluationResult.created_at.desc(), EvaluationResult.id.desc()
    ).all()
    return jsonify({"evaluations": [_evaluation_response(item) for item in results]}), 200


@ml_bp.route("/evaluations/<int:evaluation_id>", methods=["GET"])
@login_required
def get_evaluation(evaluation_id):
    result = get_owned_evaluation(evaluation_id, g.current_user.id)
    if result is None:
        return jsonify({"error": "Evaluation not found"}), 404
    return jsonify({"evaluation": _evaluation_response(result)}), 200


@ml_bp.route("/train", methods=["POST"])
@login_required
def train_model():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required."}), 400
    dataset_id = payload.get("dataset_id")
    target = payload.get("target_column")
    algorithm = payload.get("algorithm")
    task = payload.get("task_type")
    if not isinstance(dataset_id, int) or isinstance(dataset_id, bool) or not isinstance(target, str) or not target.strip() or not isinstance(algorithm, str) or not algorithm:
        return jsonify({"error": "Dataset, target column, and algorithm are required."}), 400
    target = target.strip()
    if task not in ("classification", "regression"):
        return jsonify({"error": "Task type must be classification or regression."}), 400
    dataset = get_owned_dataset(dataset_id, g.current_user.id)
    if dataset is None:
        return jsonify({"error": "Dataset not found."}), 404
    source = os.path.realpath(dataset.file_path)
    upload_root = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    try:
        safe = os.path.commonpath([upload_root, source]) == upload_root
    except ValueError:
        safe = False
    if not safe or not os.path.isfile(source):
        return jsonify({"error": "The dataset file is missing or unavailable."}), 404
    try:
        frame = pd.read_csv(source)
        with open(source, "rb") as dataset_file:
            dataset_fingerprint = hashlib.sha256(dataset_file.read()).hexdigest()
        if target not in frame.columns:
            return jsonify({"error": "The selected target column is unavailable."}), 400
        frame = frame.dropna(subset=[target])
        if len(frame) < 4:
            return jsonify({"error": "At least four rows with a target value are required for training."}), 400
        X = frame.drop(columns=[target])
        y = frame[target]
        try:
            test_size = float(payload.get("test_size", 0.2))
            random_state = int(payload.get("random_state", 42))
        except (TypeError, ValueError):
            return jsonify({"error": "Split and random seed values must be valid numbers."}), 400
        if not 0.1 <= test_size <= 0.5:
            return jsonify({"error": "Test split must be between 0.1 and 0.5."}), 400
        test_rows = math.ceil(len(frame) * test_size)
        train_rows = len(frame) - test_rows
        if test_rows < 2 or train_rows < 2:
            return jsonify({"error": "The selected split must leave at least two training and two test rows."}), 400
        class_count = y.nunique() if task == "classification" else 0
        use_stratification = (
            task == "classification" and class_count > 1
            and y.value_counts().min() >= 2
            and test_rows >= class_count and train_rows >= class_count
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if use_stratification else None,
        )
        pipeline = ModelTrainer.train(algorithm, task, X_train, y_train)
        metrics, matrix, report = ModelEvaluator.evaluate(pipeline, X_test, y_test, task)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return jsonify({"error": "The dataset could not be read for training."}), 400
    except Exception:
        current_app.logger.exception("Model training failed")
        return jsonify({"error": "Training failed. Check the data and selected options."}), 500

    model_id = None
    artifact_path = None
    try:
        model_dir = os.path.join(current_app.config["SAVED_MODELS_FOLDER"], str(g.current_user.id))
        os.makedirs(model_dir, mode=0o700, exist_ok=True)
        os.chmod(model_dir, 0o700)
        artifact_path = os.path.abspath(os.path.join(model_dir, f"{uuid.uuid4().hex}.joblib"))
        ModelTrainer.save(pipeline, artifact_path)
        os.chmod(artifact_path, 0o600)
        artifact_sha256 = _sha256_file(artifact_path)
        config = {
            "task_type": task, "target_column": target,
            "test_size": test_size, "train_size": 1 - test_size,
            "random_state": random_state, "preprocessing": {
                "dataset_preprocessing": (dataset.metadata_json or {}).get("last_preprocessing"),
                "numeric": "median imputation",
                "categorical": "most frequent imputation and one-hot encoding",
            }, "feature_columns": [str(column) for column in X.columns],
            "dataset_filename": dataset.filename, "row_count": int(len(frame)),
            "dataset_fingerprint": dataset_fingerprint,
            "artifact_sha256": artifact_sha256,
        }
        model = MLModel(
            user_id=g.current_user.id, dataset_id=dataset.id,
            model_name=f"{algorithm.replace('_', ' ').title()} - {task.title()}",
            algorithm=algorithm, model_path=artifact_path, status="trained", config_json=config,
        )
        db.session.add(model)
        db.session.flush()
        result = EvaluationResult(
            user_id=g.current_user.id, model_id=model.id,
            metrics_json=metrics, confusion_matrix_json=matrix, report_json=report,
        )
        db.session.add(result)
        db.session.commit()
        model_id = model.id
    except Exception:
        db.session.rollback()
        if artifact_path and os.path.isfile(artifact_path):
            try:
                os.remove(artifact_path)
            except OSError:
                current_app.logger.warning("Could not remove incomplete model artifact")
        current_app.logger.exception("Could not persist trained model")
        return jsonify({"error": "The trained model could not be saved."}), 500
    return jsonify({
        "message": "Model trained and evaluated successfully.",
        "model": _model_response(db.session.get(MLModel, model_id)),
        "evaluation": _evaluation_response(db.session.get(EvaluationResult, result.id)),
    }), 201


@ml_bp.route("/evaluate", methods=["POST"])
@login_required
def evaluate_model():
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id") if isinstance(payload, dict) else None
    model = get_owned_model(model_id, g.current_user.id) if isinstance(model_id, int) else None
    if model is None:
        return jsonify({"error": "Model not found."}), 404
    # Evaluation records are created atomically during training using its exact held-out split.
    result = model.evaluations[-1] if model.evaluations else None
    if result is None:
        return jsonify({"error": "No persisted evaluation is available for this model."}), 404
    return jsonify({"evaluation": _evaluation_response(result)}), 200


@ml_bp.route("/compare", methods=["POST"])
@login_required
def compare_models():
    payload = request.get_json(silent=True) or {}
    model_ids = payload.get("model_ids") if isinstance(payload, dict) else None
    if not isinstance(model_ids, list) or len(model_ids) < 2 or any(not isinstance(value, int) or isinstance(value, bool) for value in model_ids) or len(set(model_ids)) != len(model_ids):
        return jsonify({"error": "Select at least two distinct model runs to compare."}), 400
    models = owned_model_query(g.current_user.id).filter(MLModel.id.in_(model_ids)).all()
    if len(models) != len(model_ids):
        return jsonify({"error": "One or more model runs are unavailable."}), 404
    models.sort(key=lambda model: model_ids.index(model.id))
    configs = [model.config_json or {} for model in models]
    results = [next(iter(sorted(model.evaluations, key=lambda item: item.id, reverse=True)), None) for model in models]
    if any(result is None for result in results):
        return jsonify({"error": "An evaluation result is unavailable for one or more model runs."}), 409
    task = configs[0].get("task_type")
    if task not in ("classification", "regression") or any(config.get("task_type") != task for config in configs):
        return jsonify({"error": "Classification and regression runs cannot be compared together."}), 400
    compatibility = ("dataset_id", "target_column", "test_size", "random_state", "preprocessing", "dataset_fingerprint")
    if any(model.dataset_id != models[0].dataset_id for model in models) or any(
        any(config.get(key) != configs[0].get(key) for key in compatibility[1:]) for config in configs[1:]
    ):
        return jsonify({"error": "Runs must use the same dataset, target, split, seed, and preprocessing settings."}), 400
    metric_names = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    if any(
        not isinstance(result.metrics_json.get(name), (int, float))
        or not math.isfinite(result.metrics_json.get(name))
        for result in results for name in metric_names
    ):
        return jsonify({"error": "A compatible persisted metric is unavailable for one or more runs."}), 409
    metrics = {name: [{"model_id": model.id, "model_name": model.model_name, "run_label": f"{model.model_name} · Run #{model.id}", "value": result.metrics_json.get(name)}
                      for model, result in zip(models, results)] for name in metric_names}
    return jsonify({
        "task_type": task, "dataset_id": models[0].dataset_id,
        "dataset_name": models[0].dataset.filename,
        "models": [{**_model_response(model), "evaluation_id": result.id, "metrics": result.metrics_json}
                   for model, result in zip(models, results)],
        "metric_names": list(metric_names), "chart": {"labels": [f"{model.model_name} · Run #{model.id}" for model in models], "metrics": metrics},
        "limitations": "Metrics are computed on the same fixed random holdout split; results may vary for other splits.",
    }), 200


def _download_file(model):
    path = _safe_model_path(model)
    if path is None:
        return jsonify({"error": "The saved model artifact is unavailable."}), 404
    if not os.path.isfile(path):
        return jsonify({"error": "The saved model artifact is missing."}), 404
    try:
        if os.path.getsize(path) == 0 or _sha256_file(path) != (model.config_json or {}).get("artifact_sha256"):
            return jsonify({"error": "The saved model artifact is corrupted."}), 500
    except OSError:
        return jsonify({"error": "The saved model artifact could not be read."}), 500
    filename = f"model-{model.id}.joblib"
    return send_file(path, mimetype="application/octet-stream", as_attachment=True, download_name=filename)


@ml_bp.route("/models/<int:model_id>/download", methods=["GET"])
@login_required
def download_model(model_id):
    model = get_owned_model(model_id, g.current_user.id)
    if model is None:
        return jsonify({"error": "Model not found."}), 404
    return _download_file(model)


@ml_bp.route("/evaluations/<int:evaluation_id>/report", methods=["GET"])
@login_required
def download_report(evaluation_id):
    result = get_owned_evaluation(evaluation_id, g.current_user.id)
    if result is None:
        return jsonify({"error": "Evaluation report not found."}), 404
    model = result.model
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter, pageCompression=0)
        _width, height = letter
        y = height - 54
        lines = [
            "Model Evaluation Report", f"Model: {model.model_name}",
            f"Algorithm: {model.algorithm}", f"Dataset: {model.dataset.filename}",
            f"Dataset rows: {(model.config_json or {}).get('row_count', 'Unavailable')}",
            f"Task type: {(model.config_json or {}).get('task_type', 'Unavailable')}",
            f"Target: {(model.config_json or {}).get('target_column', 'Unavailable')}",
            f"Train/test split: {(model.config_json or {}).get('train_size', 'Unavailable')} / {(model.config_json or {}).get('test_size', 'Unavailable')}",
            f"Evaluation timestamp: {result.created_at.isoformat() if result.created_at else 'Unavailable'}",
            "Evaluation metrics:",
        ]
        lines.extend(f"  {name}: {value}" for name, value in (result.metrics_json or {}).items())
        lines.extend(["", "Limitations: Results reflect this run's fixed held-out split. Preprocessing was fitted to training features and applied to test features."])
        for line in lines:
            for wrapped in textwrap.wrap(line, width=90, break_long_words=True) or [""]:
                if y < 54:
                    pdf.showPage()
                    y = height - 54
                pdf.drawString(48, y, wrapped.encode("latin-1", "replace").decode("latin-1"))
                y -= 17
        pdf.save()
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                         download_name=f"evaluation-{result.id}.pdf")
    except Exception:
        current_app.logger.exception("PDF report generation failed for evaluation %s", evaluation_id)
        return jsonify({"error": "The evaluation report could not be generated."}), 500
