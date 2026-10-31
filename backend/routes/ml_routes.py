import hashlib
import math
import os
from io import BytesIO
from numbers import Real
from uuid import uuid4

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.exc import SQLAlchemyError

from database.db import db
from ml.evaluator import ModelEvaluator
from ml.preprocessor import DataPreprocessor
from ml.trainer import ModelTrainer
from models.dataset import Dataset
from models.evaluation import EvaluationResult
from models.model import MLModel
from routes.auth_utils import (
    get_owned_dataset,
    get_owned_evaluation,
    get_owned_model,
    login_required,
    non_instructor_required,
    role_required,
    log_activity,
    owned_evaluation_query,
    owned_model_query,
)


ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")


def _model_response(model):
    return {
        "id": model.id,
        "dataset_id": model.dataset_id,
        "dataset_name": model.dataset.filename if model.dataset else None,
        "model_name": model.model_name,
        "algorithm": model.algorithm,
        "run_label": f"{model.model_name} · Run #{model.id}",
        "status": model.status,
        "configuration": model.config_json,
        "created_at": model.created_at.isoformat() if model.created_at else None,
        "evaluation_id": model.evaluations[-1].id if model.evaluations else None,
        "metrics": model.evaluations[-1].metrics_json if model.evaluations else None,
    }


def _evaluation_response(result):
    return {
        "id": result.id,
        "model_id": result.model_id,
        "algorithm": result.model.algorithm,
        "metrics": result.metrics_json,
        "confusion_matrix": result.confusion_matrix_json,
        "report": result.report_json,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "shared_with_instructors": bool(result.shared_with_instructors),
        "model_name": result.model.model_name,
        "configuration": result.model.config_json,
    }


def _instructor_evaluation_response(result):
    """Return only the fields required for read-only instructor review."""
    report = result.report_json or {}
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
    return {
        "id": result.id,
        "model_id": result.model_id,
        "model_name": result.model.model_name,
        "algorithm": result.model.algorithm,
        "metrics": result.metrics_json,
        "confusion_matrix": result.confusion_matrix_json,
        "confusion_matrix_labels": [key for key in report if key not in aggregate_keys],
        "report": report or None,
        "created_at": result.created_at.isoformat() if result.created_at else None,
        "owner": result.user.username,
    }


@ml_bp.route("/algorithms", methods=["GET"])
@login_required
def get_algorithms():
    return jsonify({
        **ModelTrainer.algorithms_for()
    }), 200


@ml_bp.route("/models", methods=["GET"])
@non_instructor_required
def list_models():
    models = owned_model_query(g.current_user.id).order_by(
        MLModel.created_at.desc(), MLModel.id.desc()
    ).all()
    return jsonify({"models": [_model_response(item) for item in models]}), 200


@ml_bp.route("/models/<int:model_id>", methods=["GET"])
@non_instructor_required
def get_model(model_id):
    model = get_owned_model(model_id, g.current_user.id)
    if model is None:
        return jsonify({"error": "Model not found"}), 404
    return jsonify({"model": _model_response(model)}), 200


@ml_bp.route("/evaluations", methods=["GET"])
@non_instructor_required
def list_evaluations():
    results = owned_evaluation_query(g.current_user.id).order_by(
        EvaluationResult.created_at.desc(), EvaluationResult.id.desc()
    ).all()
    return jsonify({"evaluations": [_evaluation_response(item) for item in results]}), 200


@ml_bp.route("/evaluations/<int:evaluation_id>", methods=["GET"])
@non_instructor_required
def get_evaluation(evaluation_id):
    result = get_owned_evaluation(evaluation_id, g.current_user.id)
    if result is None:
        return jsonify({"error": "Evaluation not found"}), 404
    return jsonify({"evaluation": _evaluation_response(result)}), 200



@ml_bp.route("/compare", methods=["POST"])
@non_instructor_required
def compare_evaluations():
    payload = request.get_json(silent=True) if request.is_json else None
    ids = payload.get("evaluation_ids") if isinstance(payload, dict) else None
    id_kind = "evaluation_ids"
    if ids is None and isinstance(payload, dict):
        ids = payload.get("model_ids")
        id_kind = "model_ids"
    valid_ids = isinstance(ids, list) and 2 <= len(ids) <= 10
    if not valid_ids or any(type(item) is not int for item in ids) or len(set(ids)) != len(ids):
        return jsonify({"error": "Select between 2 and 10 distinct model or evaluation IDs."}), 400

    if id_kind == "model_ids":
        found = owned_model_query(g.current_user.id).filter(MLModel.id.in_(ids)).all()
        if len(found) != len(ids):
            return jsonify({"error": "One or more model runs were not found."}), 404
        models_by_id = {item.id: item for item in found}
        models = [models_by_id[item] for item in ids]
        evaluations = [
            max(model.evaluations, key=lambda item: item.id) if model.evaluations else None
            for model in models
        ]
    else:
        found = owned_evaluation_query(g.current_user.id).filter(EvaluationResult.id.in_(ids)).all()
        if len(found) != len(ids):
            return jsonify({"error": "One or more evaluations were not found."}), 404
        evaluations_by_id = {item.id: item for item in found}
        evaluations = [evaluations_by_id[item] for item in ids]
        models = [item.model for item in evaluations]

    if any(item is None for item in evaluations):
        return jsonify({"error": "An evaluation result is unavailable for one or more runs."}), 409

    configs = [model.config_json or {} for model in models]
    tasks = [config.get("task_type") for config in configs]
    if tasks[0] not in ("classification", "regression") or any(task != tasks[0] for task in tasks):
        return jsonify({"error": "Classification and regression runs cannot be compared together."}), 400
    compatibility_keys = ("target_column", "test_size", "random_state", "preprocessing", "dataset_fingerprint")
    if any(model.dataset_id != models[0].dataset_id for model in models) or any(
        any(config.get(key) != configs[0].get(key) for key in compatibility_keys)
        for config in configs[1:]
    ):
        return jsonify({"error": "Runs must use the same dataset, target, split, seed, and preprocessing settings."}), 400

    metric_names = (
        ("accuracy", "precision", "recall", "f1")
        if tasks[0] == "classification" else ("mae", "mse", "rmse", "r2")
    )
    metric_points = {}
    for name in metric_names:
        points = []
        for model, evaluation in zip(models, evaluations):
            value = (evaluation.metrics_json or {}).get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return jsonify({"error": "A comparable persisted metric is unavailable for one or more runs."}), 409
            points.append({
                "model_id": model.id,
                "evaluation_id": evaluation.id,
                "model_name": model.model_name,
                "run_label": f"{model.model_name} · Run #{model.id}",
                "value": value,
            })
        metric_points[name] = points

    model_rows = []
    for model, evaluation in zip(models, evaluations):
        row = _model_response(model)
        row.update({"evaluation_id": evaluation.id, "metrics": evaluation.metrics_json})
        model_rows.append(row)
    evaluation_rows = [_evaluation_response(item) for item in evaluations]
    return jsonify({
        "task_type": tasks[0],
        "dataset_id": models[0].dataset_id,
        "dataset_name": models[0].dataset.filename if models[0].dataset else None,
        "models": model_rows,
        "evaluations": evaluation_rows,
        "metric_names": list(metric_names),
        "chart": {"labels": [item["run_label"] for item in model_rows], "metrics": metric_points},
        "limitations": "Metrics use each run's persisted held-out evaluation split.",
    }), 200


def _owned_artifact(model_id):
    model = get_owned_model(model_id, g.current_user.id)
    if model is None:
        return None, None
    folder = os.path.realpath(current_app.config["SAVED_MODELS_FOLDER"])
    path = os.path.realpath(model.model_path or "")
    try:
        safe = os.path.commonpath([folder, path]) == folder
    except ValueError:
        safe = False
    if not safe or not os.path.isfile(path):
        return model, None
    return model, path


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@ml_bp.route("/models/<int:model_id>/download", methods=["GET"])
@non_instructor_required
def download_model(model_id):
    model, path = _owned_artifact(model_id)
    if model is None or path is None:
        return jsonify({"error": "Model artifact not found."}), 404
    if os.path.getsize(path) == 0:
        return jsonify({"error": "The saved model artifact is corrupted."}), 500
    expected_hash = (model.config_json or {}).get("artifact_sha256")
    if expected_hash:
        try:
            if _sha256_file(path) != expected_hash:
                return jsonify({"error": "The saved model artifact is corrupted."}), 500
        except OSError:
            return jsonify({"error": "The saved model artifact could not be read."}), 500
    return send_file(path, as_attachment=True, download_name=f"model-{model.id}-{model.algorithm}.joblib")


@ml_bp.route("/evaluations/<int:evaluation_id>/report", methods=["GET"])
@non_instructor_required
def download_report(evaluation_id):
    result = get_owned_evaluation(evaluation_id, g.current_user.id)
    if result is None:
        return jsonify({"error": "Evaluation not found."}), 404
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter, pageCompression=0)
    width, height = letter
    y = height - 54
    pdf.setTitle(f"Evaluation report {result.id}")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(48, y, "ML Model Evaluation Report")
    y -= 30
    pdf.setFont("Helvetica", 10)
    config = result.model.config_json or {}
    lines = [
        f"Run ID: {result.id}", f"Model: {result.model.model_name}",
        f"Algorithm: {result.model.algorithm}",
        f"Dataset: {result.model.dataset.filename if result.model.dataset else 'Unavailable'}",
        f"Task type: {config.get('task_type', 'Unavailable')}",
        f"Target: {config.get('target_column', 'Unavailable')}",
        f"Train/test split: {config.get('train_size', 'Unavailable')} / {config.get('test_size', 'Unavailable')}",
        f"Created: {result.created_at}", "Evaluation metrics:",
    ]
    lines.extend(f"{key}: {value}" for key, value in (result.metrics_json or {}).items())
    lines.append("Limitations: Metrics reflect this run's persisted held-out evaluation split.")
    for line in lines:
        if y < 48:
            pdf.showPage()
            y = height - 48
            pdf.setFont("Helvetica", 10)
        pdf.drawString(48, y, line[:110])
        y -= 17
    pdf.save()
    output.seek(0)
    return send_file(output, mimetype="application/pdf", as_attachment=True,
                     download_name=f"evaluation-{result.id}.pdf")


@ml_bp.route("/evaluations/<int:evaluation_id>/sharing", methods=["PATCH"])
@non_instructor_required
def update_evaluation_sharing(evaluation_id):
    result = get_owned_evaluation(evaluation_id, g.current_user.id)
    if result is None:
        return jsonify({"error": "Evaluation not found."}), 404
    payload = request.get_json(silent=True) if request.is_json else None
    if not isinstance(payload, dict) or type(payload.get("shared_with_instructors")) is not bool:
        return jsonify({"error": "shared_with_instructors must be a boolean."}), 400
    result.shared_with_instructors = payload["shared_with_instructors"]
    db.session.commit()
    return jsonify({"evaluation_id": result.id,
                    "shared_with_instructors": result.shared_with_instructors}), 200


@ml_bp.route("/instructor/evaluations", methods=["GET"])
@role_required("instructor")
def instructor_evaluations():
    results = (EvaluationResult.query.join(MLModel).join(Dataset)
               .filter(EvaluationResult.shared_with_instructors.is_(True))
               .order_by(EvaluationResult.created_at.desc(), EvaluationResult.id.desc()).all())
    return jsonify({"evaluations": [
        _instructor_evaluation_response(item) for item in results
    ]}), 200


@ml_bp.route("/instructor/evaluations/<int:evaluation_id>", methods=["GET"])
@role_required("instructor")
def instructor_evaluation(evaluation_id):
    result = (EvaluationResult.query.join(MLModel).join(Dataset)
              .filter(EvaluationResult.id == evaluation_id,
                      EvaluationResult.shared_with_instructors.is_(True)).first())
    if result is None:
        return jsonify({"error": "Evaluation not found."}), 404
    return jsonify({"evaluation": _instructor_evaluation_response(result)}), 200


@ml_bp.route("/instructor/compare", methods=["POST"])
@role_required("instructor")
def compare_shared_evaluations():
    payload = request.get_json(silent=True) if request.is_json else None
    ids = payload.get("evaluation_ids") if isinstance(payload, dict) else None
    valid_ids = isinstance(ids, list) and 2 <= len(ids) <= 10
    if not valid_ids or any(type(item) is not int for item in ids) or len(set(ids)) != len(ids):
        return jsonify({"error": "Select between 2 and 10 distinct evaluation IDs."}), 400
    results = (EvaluationResult.query.join(MLModel).join(Dataset)
               .filter(EvaluationResult.id.in_(ids),
                       EvaluationResult.shared_with_instructors.is_(True)).all())
    if len(results) != len(ids):
        return jsonify({"error": "One or more shared evaluations were not found."}), 404
    by_id = {item.id: item for item in results}
    results = [by_id[item] for item in ids]
    tasks = {(item.model.config_json or {}).get("task_type") for item in results}
    if len(tasks) != 1:
        return jsonify({"error": "Only shared evaluations for the same task type can be compared."}), 400
    return jsonify({"evaluations": [
        _instructor_evaluation_response(item) for item in results
    ]}), 200


def _train_and_persist(body):
    """Shared training/evaluation path for both public training endpoints."""
    if not isinstance(body, dict):
        return jsonify({"error": "A valid JSON object is required."}), 400

    dataset_id = body.get("dataset_id")
    target = body.get("target_column")
    algorithm = body.get("algorithm")
    task = body.get("task_type")
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, int):
        return jsonify({"error": "dataset_id must be an integer."}), 400
    if not isinstance(target, str) or not target.strip():
        return jsonify({"error": "Choose a target column."}), 400
    if not isinstance(algorithm, str) or not algorithm.strip():
        return jsonify({"error": "Choose a supported algorithm."}), 400
    target, algorithm = target.strip(), algorithm.strip()
    if task not in ("classification", "regression"):
        return jsonify({"error": "Choose a classification or regression task."}), 400
    try:
        ModelTrainer.create_estimator(algorithm, task)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    dataset = get_owned_dataset(dataset_id, g.current_user.id)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404
    upload_root = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    source_path = os.path.realpath(dataset.file_path)
    try:
        safe_source = os.path.commonpath([upload_root, source_path]) == upload_root
    except ValueError:
        safe_source = False
    if not safe_source or not os.path.isfile(source_path):
        return jsonify({"error": "The dataset file is missing or unavailable."}), 404

    try:
        with open(source_path, "rb") as source_file:
            dataset_fingerprint = hashlib.sha256(source_file.read()).hexdigest()
        frame = pd.read_csv(source_path)
    except (OSError, PermissionError):
        return jsonify({"error": "The dataset file could not be read."}), 500
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return jsonify({"error": "The dataset CSV is malformed or unreadable."}), 400
    if target not in frame.columns:
        return jsonify({"error": f"Target column '{target}' was not found in the dataset."}), 400

    preprocessing = body.get("preprocessing") or {}
    if not isinstance(preprocessing, dict):
        return jsonify({"error": "preprocessing must be an object."}), 400
    preprocessing_config = {
        "missing_values": preprocessing.get("missing_values", "none"),
        "encoding": preprocessing.get("encoding", "none"),
        "scaling": preprocessing.get("scaling", "none"),
        "fill_value": preprocessing.get("fill_value"),
    }
    try:
        frame, preprocessing_summary = DataPreprocessor.preprocess(
            frame, config=preprocessing_config, target_column=target
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if target not in frame.columns:
        return jsonify({"error": "The target column was removed during preprocessing."}), 400
    frame = frame.dropna(subset=[target]).dropna(axis=1, how="all")
    X = frame.drop(columns=[target])
    y = frame[target]
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        usable = y.notna()
        X, y = X.loc[usable], y.loc[usable]
    else:
        y = y.astype(str)
    if X.shape[1] == 0:
        return jsonify({"error": "At least one feature column is required."}), 400
    if len(y) < 2:
        return jsonify({"error": "At least 2 usable samples are required for training and testing."}), 400

    test_size = body.get("test_size", 0.2)
    if isinstance(test_size, bool) or not isinstance(test_size, Real):
        return jsonify({"error": "test_size must be a finite number greater than 0 and less than 1."}), 400
    test_size = float(test_size)
    if not math.isfinite(test_size) or not 0 < test_size < 1:
        return jsonify({"error": "test_size must be a finite number greater than 0 and less than 1."}), 400
    random_state = body.get("random_state", 42)
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        return jsonify({"error": "random_state must be an integer."}), 400

    stratify = None
    if task == "classification":
        counts = y.value_counts()
        if counts.min() < 2:
            return jsonify({"error": "Each target class needs at least 2 samples for a stratified split."}), 400
        stratify = y
    try:
        X_train, X_test, y_train, y_test = DataPreprocessor.split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    minimum_train_count = 5 if algorithm == "knn" else 2
    if len(X_train) < minimum_train_count:
        return jsonify({
            "error": (
                f"The selected split leaves {len(X_train)} training samples, but "
                f"{algorithm} requires at least {minimum_train_count}. Choose a different ratio "
                "or provide more data."
            )
        }), 400
    if algorithm == "logistic_regression" and y_train.value_counts().min() < 2:
        return jsonify({"error": "Logistic regression needs at least 2 training samples from every target class."}), 400
    try:
        fitted_model = ModelTrainer.train(algorithm, task, X_train, y_train)
        metrics, confusion, report = ModelEvaluator.evaluate(fitted_model, X_test, y_test, task)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"The model could not be trained with this data: {exc}"}), 400
    except Exception:
        current_app.logger.exception("Model training failed")
        return jsonify({"error": "Training failed. Check the data and selected options."}), 500

    model_path = None
    try:
        model_root = os.path.realpath(current_app.config["SAVED_MODELS_FOLDER"])
        user_folder = os.path.join(model_root, str(g.current_user.id))
        os.makedirs(user_folder, mode=0o700, exist_ok=True)
        os.chmod(user_folder, 0o700)
        model_path = os.path.abspath(os.path.join(user_folder, f"{uuid4().hex}.joblib"))
        if os.path.commonpath([model_root, model_path]) != model_root:
            return jsonify({"error": "Unable to safely save the trained model."}), 500
        ModelTrainer.save(fitted_model, model_path)
        os.chmod(model_path, 0o600)
        artifact_sha256 = _sha256_file(model_path)
        split_summary = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_fraction": len(X_train) / len(X),
            "test_fraction": len(X_test) / len(X),
            "requested_test_size": test_size,
        }
        config = {
            "algorithm": algorithm,
            "task_type": task,
            "target_column": target,
            "test_size": test_size,
            "train_size": len(X_train) / len(X),
            "random_state": random_state,
            "preprocessing": {
                **preprocessing_config,
                "summary": preprocessing_summary,
                "dataset_preprocessing": (dataset.metadata_json or {}).get("last_preprocessing"),
                "numeric": "median imputation",
                "categorical": "most frequent imputation and one-hot encoding",
            },
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X.shape[1]),
            "features": [str(column) for column in X.columns],
            "feature_columns": [str(column) for column in X.columns],
            "dataset_filename": dataset.filename,
            "row_count": int(len(frame)),
            "dataset_fingerprint": dataset_fingerprint,
            "artifact_sha256": artifact_sha256,
            **split_summary,
        }
        model_name = str(body.get("model_name") or f"{algorithm.replace('_', ' ').title()} - {task.title()}").strip()
        if not model_name:
            model_name = f"{algorithm.replace('_', ' ').title()} - {task.title()}"
        model = MLModel(
            user_id=g.current_user.id,
            dataset_id=dataset.id,
            model_name=model_name,
            algorithm=algorithm,
            model_path=model_path,
            status="completed",
            config_json=config,
        )
        db.session.add(model)
        db.session.flush()
        evaluation = EvaluationResult(
            user_id=g.current_user.id,
            model_id=model.id,
            metrics_json=metrics,
            confusion_matrix_json=confusion,
            report_json=report,
        )
        db.session.add(evaluation)
        db.session.commit()
        log_activity(
            action="model_trained",
            user_id=g.current_user.id,
            details={
                "model_id": model.id,
                "algorithm": algorithm,
                "task_type": task,
                "dataset_id": dataset.id,
            },
        )
    except (OSError, SQLAlchemyError):
        db.session.rollback()
        if model_path:
            try:
                os.remove(model_path)
            except OSError:
                pass
        current_app.logger.exception("Could not persist trained model and evaluation")
        return jsonify({"error": "The trained model and evaluation could not be saved."}), 500

    model_response = _model_response(model)
    evaluation_response = _evaluation_response(evaluation)
    # The US-20 training response keeps its compact chart metric names; the
    # persisted evaluation retains the explicit weighted aliases used by US-19.
    evaluation_response["metrics"] = {
        key: metrics[key] for key in ("accuracy", "precision", "recall", "f1")
        if key in metrics
    } if task == "classification" else metrics
    return jsonify({
        "message": "Model trained and evaluated successfully.",
        "model": model_response,
        "evaluation": evaluation_response,
        "model_id": model.id,
        "evaluation_id": evaluation.id,
        "algorithm": algorithm,
        "task_type": task,
        "metrics": metrics,
        "confusion_matrix": confusion,
        "report": report,
        "split": split_summary,
    }), 201


@ml_bp.route("/train", methods=["POST"])
@non_instructor_required
def train_model():
    """Train, persist, and evaluate a model run (US-18 and US-20)."""
    payload = request.get_json(silent=True) if request.is_json else None
    if payload is None:
        return jsonify({"error": "A JSON request body is required."}), 400
    return _train_and_persist(payload)


@ml_bp.route("/evaluate", methods=["POST"])
@non_instructor_required
def evaluate_model():
    """Train and evaluate on a dataset split, persisting both records."""
    payload = request.get_json(silent=True) if request.is_json else None
    if payload is None:
        return jsonify({"error": "A JSON request body is required."}), 400
    return _train_and_persist(payload)
