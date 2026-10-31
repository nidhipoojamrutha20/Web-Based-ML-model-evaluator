
import csv
import io
import math
import os
from uuid import uuid4

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request

from database.db import db
from ml.preprocessor import DataPreprocessor
from models.dataset import Dataset
from routes.auth_utils import (
    non_instructor_required,
    get_owned_dataset,
    login_required,
    owned_dataset_query,
    log_activity,
)


dataset_bp = Blueprint("dataset", __name__, url_prefix="/api/dataset")


def _parse_csv(contents, min_data_rows=2, min_data_columns=2):
    """Validate a UTF-8 CSV and return its DataFrame and original headers."""
    if b"\x00" in contents:
        raise ValueError(
            "The CSV contains binary NUL characters and cannot be read."
        )

    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The file must be a readable UTF-8 CSV."
        ) from exc

    try:
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next((row for row in rows if row), None)

        if header is None:
            raise ValueError(
                "The CSV file is empty or has no header row."
            )

        normalized_headers = [name.strip() for name in header]

        if any(not name for name in normalized_headers):
            raise ValueError(
                "Every CSV column must have a non-empty header."
            )

        if len(set(normalized_headers)) != len(normalized_headers):
            raise ValueError(
                "CSV column headers must be unique."
            )

        if len(header) < min_data_columns:
            raise ValueError(
                f"The CSV must contain at least {min_data_columns} columns "
                "(one feature and one target)."
            )

        for row in rows:
            if not row:
                continue

            if len(row) != len(header):
                raise ValueError(
                    "Each CSV row must contain the same number of "
                    "fields as the header."
                )

            if all(not value.strip() for value in row):
                raise ValueError(
                    "The CSV contains an empty data row."
                )

    except csv.Error as exc:
        raise ValueError(
            "The CSV file is malformed and could not be parsed."
        ) from exc

    try:
        frame = pd.read_csv(
            io.BytesIO(contents),
            encoding="utf-8-sig",
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "The CSV file is malformed and could not be parsed."
        ) from exc

    frame = frame.dropna(how="all")

    if frame.shape[1] < min_data_columns:
        raise ValueError(
            f"The CSV must contain at least {min_data_columns} columns "
            "(one feature and one target)."
        )

    if len(frame.index) < min_data_rows:
        raise ValueError(
            f"The CSV must contain at least {min_data_rows} "
            "non-empty data rows."
        )

    usable_columns = frame.notna().any(axis=0).sum()

    if usable_columns < min_data_columns:
        raise ValueError(
            f"The CSV must contain at least {min_data_columns} "
            "columns with data."
        )

    return frame, header


def _persist_dataset(dataset):
    """Persist a newly uploaded dataset."""
    db.session.add(dataset)
    db.session.commit()


def _dataset_response(dataset):
    """Return dataset metadata without exposing its server file path."""
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "file_size": dataset.file_size,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "metadata": dataset.metadata_json,
        "created_at": (
            dataset.created_at.isoformat()
            if dataset.created_at
            else None
        ),
    }


def _store_upload(contents, upload_folder):
    """Save uploaded bytes to a unique path inside the upload directory."""
    upload_folder = os.path.abspath(upload_folder)
    os.makedirs(upload_folder, exist_ok=True)

    stored_path = os.path.abspath(
        os.path.join(upload_folder, f"{uuid4().hex}.csv")
    )

    if os.path.commonpath([upload_folder, stored_path]) != upload_folder:
        raise ValueError("Unable to safely store this upload.")

    with open(stored_path, "xb") as destination:
        destination.write(contents)

    return stored_path


def _remove_file(path):
    """Best-effort cleanup for a file created by an upload request."""
    if not path:
        return

    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        current_app.logger.exception(
            "Failed to remove dataset file: %s", path
        )


def _read_and_validate_upload():
    """Read and validate a CSV from the current multipart request."""
    if "file" not in request.files:
        return None, None, None, (
            jsonify({"error": "Choose a CSV file to upload."}),
            400,
        )

    uploaded_file = request.files["file"]
    filename = uploaded_file.filename or ""

    if not filename:
        return None, None, None, (
            jsonify({"error": "Choose a CSV file to upload."}),
            400,
        )

    if not filename.lower().endswith(".csv"):
        return None, None, None, (
            jsonify({
                "error": "Unsupported file type. Please choose a .csv file."
            }),
            400,
        )

    if len(filename) > 255:
        return None, None, None, (
            jsonify({
                "error": "The filename must be 255 characters or fewer."
            }),
            400,
        )

    contents = uploaded_file.read()

    if not contents:
        return None, None, None, (
            jsonify({"error": "The selected CSV file is empty."}),
            400,
        )

    try:
        frame, columns = _parse_csv(
            contents,
            min_data_rows=current_app.config.get("MIN_DATA_ROWS", 2),
            min_data_columns=current_app.config.get("MIN_DATA_COLUMNS", 2),
        )
    except ValueError as exc:
        return None, None, None, (
            jsonify({"error": str(exc)}),
            400,
        )

    return (filename, contents, (frame, columns)), None, None, None


@dataset_bp.route("", methods=["GET"])
@non_instructor_required
def list_datasets():
    """List datasets belonging to the logged-in user."""
    datasets = (
        owned_dataset_query(g.current_user.id)
        .order_by(Dataset.created_at.asc(), Dataset.id.asc())
        .all()
    )

    return jsonify({
        "datasets": [_dataset_response(item) for item in datasets]
    }), 200


@dataset_bp.route("/<int:dataset_id>", methods=["GET"])
@non_instructor_required
def get_dataset(dataset_id):
    """Retrieve an owned dataset's metadata."""
    dataset = get_owned_dataset(dataset_id, g.current_user.id)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    return jsonify({"dataset": _dataset_response(dataset)}), 200


@dataset_bp.route("/upload", methods=["POST"])
@non_instructor_required
def upload_dataset():
    """Upload a new dataset without replacing existing dataset records."""
    result, _, _, error_response = _read_and_validate_upload()

    if error_response is not None:
        return error_response

    filename, contents, parsed = result
    frame, columns = parsed
    stored_path = None

    try:
        stored_path = _store_upload(
            contents,
            current_app.config["UPLOAD_FOLDER"],
        )

        dataset = Dataset(
            user_id=g.current_user.id,
            filename=filename,
            file_path=stored_path,
            file_size=len(contents),
            row_count=len(frame.index),
            column_count=len(columns),
            metadata_json={"columns": columns},
        )

        _persist_dataset(dataset)
        log_activity(
            "DATASET_UPLOAD",
            user_id=g.current_user.id,
            details=f"Uploaded dataset {filename} ({len(frame.index)} rows, {len(columns)} cols)",
            ip_address=request.remote_addr
        )

    except Exception:
        db.session.rollback()
        _remove_file(stored_path)

        current_app.logger.exception(
            "Failed to save uploaded dataset"
        )

        return jsonify({
            "error": "The dataset could not be saved. Please try again."
        }), 500

    return jsonify({
        "message": "Dataset uploaded successfully.",
        "dataset_id": dataset.id,
        "filename": filename,
        "row_count": len(frame.index),
        "column_count": len(columns),
        "columns": columns,
        "file_size": len(contents),
    }), 201


@dataset_bp.route("/<int:dataset_id>/replace", methods=["POST"])
@non_instructor_required
def replace_dataset(dataset_id):
    """Replace an owned dataset while preserving its database ID."""
    dataset = get_owned_dataset(dataset_id, g.current_user.id)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    result, _, _, error_response = _read_and_validate_upload()

    if error_response is not None:
        return error_response

    filename, contents, parsed = result
    frame, columns = parsed
    old_path = dataset.file_path
    stored_path = None

    try:
        stored_path = _store_upload(
            contents,
            current_app.config["UPLOAD_FOLDER"],
        )

        dataset.filename = filename
        dataset.file_path = stored_path
        dataset.file_size = len(contents)
        dataset.row_count = len(frame.index)
        dataset.column_count = len(columns)
        dataset.metadata_json = {"columns": columns}

        db.session.commit()
        log_activity(
            "DATASET_REPLACE",
            user_id=g.current_user.id,
            details=f"Replaced dataset #{dataset.id} with {filename} ({len(frame.index)} rows, {len(columns)} cols)",
            ip_address=request.remote_addr
        )

    except Exception:
        db.session.rollback()
        _remove_file(stored_path)

        current_app.logger.exception(
            "Failed to replace dataset %s", dataset_id
        )

        return jsonify({
            "error": (
                "The dataset could not be replaced. "
                "Please try again."
            )
        }), 500

    # Delete the old file only after the database commit succeeds.
    if (
        old_path
        and os.path.abspath(old_path) != os.path.abspath(stored_path)
    ):
        _remove_file(old_path)

    return jsonify({
        "message": "Dataset replaced successfully.",
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "columns": columns,
        "file_size": dataset.file_size,
    }), 200


@dataset_bp.route("/<int:dataset_id>/preview", methods=["GET"])
@non_instructor_required
def preview_dataset(dataset_id):
    dataset = get_owned_dataset(dataset_id, g.current_user.id)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    # Uploads are stored beneath UPLOAD_FOLDER. Validate the persisted path as
    # well, so a damaged database row cannot make this endpoint read an
    # arbitrary server file.
    upload_folder = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    stored_path = os.path.realpath(dataset.file_path)
    try:
        inside_upload_folder = os.path.commonpath([upload_folder, stored_path]) == upload_folder
    except ValueError:
        inside_upload_folder = False
    if not inside_upload_folder or not os.path.isfile(stored_path):
        return jsonify({"error": "The dataset file is missing or unavailable."}), 404

    try:
        # Chunked parsing bounds memory use while still counting every row.
        # Dtypes are combined across chunks so later mixed values are noticed.
        sample_rows = []
        column_names = None
        inferred_dtypes = {}
        row_count = 0
        for chunk in pd.read_csv(stored_path, chunksize=10000):
            if column_names is None:
                column_names = [str(name) for name in chunk.columns]
                inferred_dtypes = {str(name): str(chunk[name].dtype) for name in chunk.columns}
            else:
                for name in chunk.columns:
                    key = str(name)
                    previous = inferred_dtypes[key]
                    current = str(chunk[name].dtype)
                    if previous == current:
                        continue
                    if previous.startswith(("int", "uint")) and current.startswith(("int", "uint")):
                        continue
                    if previous.startswith(("int", "uint", "float")) and current.startswith(("int", "uint", "float")):
                        inferred_dtypes[key] = "float64"
                    else:
                        inferred_dtypes[key] = "object"
            row_count += len(chunk.index)
            if len(sample_rows) < 10:
                sample_rows.extend(chunk.head(10 - len(sample_rows)).to_dict(orient="records"))
    except (OSError, PermissionError):
        return jsonify({"error": "The dataset file could not be read."}), 500
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        current_app.logger.info("Could not parse dataset preview for dataset %s: %s", dataset_id, exc)
        return jsonify({"error": "The dataset CSV is malformed or unreadable."}), 400

    def json_value(value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    rows = [
        {str(name): json_value(value) for name, value in row.items()}
        for row in sample_rows
    ]
    return jsonify({
        "dataset_id": dataset.id,
        "columns": column_names or [],
        "data_types": {name: inferred_dtypes[name] for name in (column_names or [])},
        "rows": rows,
        "row_count": row_count,
        "column_count": len(column_names or []),
        "sample_size": len(rows),
    }), 200


@dataset_bp.route("/<int:dataset_id>/statistics", methods=["GET"])
@non_instructor_required
def dataset_statistics(dataset_id):
    """Return descriptive statistics for an owned dataset."""
    dataset = get_owned_dataset(dataset_id, g.current_user.id)

    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    # Validate the stored path before reading the dataset file.
    upload_folder = os.path.realpath(
        current_app.config["UPLOAD_FOLDER"]
    )
    stored_path = os.path.realpath(dataset.file_path)

    try:
        inside_upload_folder = (
            os.path.commonpath([upload_folder, stored_path])
            == upload_folder
        )
    except ValueError:
        inside_upload_folder = False

    if not inside_upload_folder or not os.path.isfile(stored_path):
        return jsonify({
            "error": "The dataset file is missing or unavailable."
        }), 404

    try:
        frame = pd.read_csv(stored_path)
    except (OSError, PermissionError):
        current_app.logger.exception(
            "Unable to read dataset %s for statistics", dataset_id
        )
        return jsonify({
            "error": "The dataset file could not be read."
        }), 500
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ):
        current_app.logger.exception(
            "Unable to parse dataset %s for statistics", dataset_id
        )
        return jsonify({
            "error": "The dataset CSV is malformed or unreadable."
        }), 400

    def json_number(value):
        """Convert numeric results to JSON-safe values."""
        if pd.isna(value):
            return None

        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    columns = []

    for column_name in frame.columns:
        series = frame[column_name]
        missing_count = int(series.isna().sum())

        if pd.api.types.is_numeric_dtype(series):
            column_type = "numeric"
            values = series.dropna()

            statistics = {
                "mean": json_number(values.mean())
                if not values.empty else None,
                "median": json_number(values.median())
                if not values.empty else None,
                "min": json_number(values.min())
                if not values.empty else None,
                "max": json_number(values.max())
                if not values.empty else None,
            }
        else:
            column_type = "categorical"
            statistics = None

        columns.append({
            "name": str(column_name),
            "type": column_type,
            "missing_values": missing_count,
            "statistics": statistics,
        })

    return jsonify({
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": int(len(frame.index)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
    }), 200


@dataset_bp.route("/<int:dataset_id>/preprocess", methods=["POST"])
@non_instructor_required
def preprocess_dataset(dataset_id):
    """Apply data preprocessing operations to an owned dataset (US-14 / FR-3.1)."""
    dataset = get_owned_dataset(dataset_id, g.current_user.id)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404

    upload_folder = os.path.realpath(current_app.config["UPLOAD_FOLDER"])
    stored_path = os.path.realpath(dataset.file_path)

    try:
        inside_upload_folder = (
            os.path.commonpath([upload_folder, stored_path]) == upload_folder
        )
    except ValueError:
        inside_upload_folder = False

    if not inside_upload_folder or not os.path.isfile(stored_path):
        return jsonify({"error": "The dataset file is missing or unavailable."}), 404

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON request body."}), 400

    missing_strategy = str(payload.get("missing_values", "none")).lower()
    encoding_strategy = str(payload.get("encoding", "none")).lower()
    scaling_strategy = str(payload.get("scaling", "none")).lower()
    target_column = payload.get("target_column")
    if target_column:
        target_column = str(target_column).strip()
    else:
        target_column = None
    fill_value = payload.get("fill_value")
    save_changes = payload.get("save", True)

    # Validate option identifiers (SEC-INP-1)
    if missing_strategy not in DataPreprocessor.SUPPORTED_MISSING_STRATEGIES:
        return jsonify({
            "error": (
                f"Invalid missing value option '{missing_strategy}'. "
                f"Supported: {', '.join(DataPreprocessor.SUPPORTED_MISSING_STRATEGIES)}"
            )
        }), 400

    if encoding_strategy not in DataPreprocessor.SUPPORTED_ENCODING_STRATEGIES:
        return jsonify({
            "error": (
                f"Invalid encoding option '{encoding_strategy}'. "
                f"Supported: {', '.join(DataPreprocessor.SUPPORTED_ENCODING_STRATEGIES)}"
            )
        }), 400

    if scaling_strategy not in DataPreprocessor.SUPPORTED_SCALING_STRATEGIES:
        return jsonify({
            "error": (
                f"Invalid scaling option '{scaling_strategy}'. "
                f"Supported: {', '.join(DataPreprocessor.SUPPORTED_SCALING_STRATEGIES)}"
            )
        }), 400

    try:
        frame = pd.read_csv(stored_path)
    except Exception as exc:
        current_app.logger.exception(
            "Failed reading dataset %s for preprocessing: %s", dataset_id, exc
        )
        return jsonify({"error": "The dataset file could not be read."}), 500

    if target_column and target_column not in frame.columns:
        return jsonify({
            "error": f"Target column '{target_column}' does not exist in dataset."
        }), 400

    try:
        config = {
            "missing_values": missing_strategy,
            "encoding": encoding_strategy,
            "scaling": scaling_strategy,
            "fill_value": fill_value,
        }
        processed_df, summary = DataPreprocessor.preprocess(
            frame, config, target_column=target_column
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        current_app.logger.exception(
            "Unexpected error preprocessing dataset %s: %s", dataset_id, exc
        )
        return jsonify({
            "error": "Preprocessing failed due to an unexpected error."
        }), 500

    if save_changes:
        temp_path = f"{stored_path}.tmp"
        try:
            processed_df.to_csv(temp_path, index=False, encoding="utf-8")
            os.replace(temp_path, stored_path)

            existing_meta = dataset.metadata_json or {}
            existing_meta["columns"] = [str(c) for c in processed_df.columns]
            existing_meta["preprocessed"] = True
            existing_meta["last_preprocessing"] = summary

            dataset.row_count = int(len(processed_df.index))
            dataset.column_count = int(len(processed_df.columns))
            dataset.file_size = int(os.path.getsize(stored_path))
            dataset.metadata_json = existing_meta

            db.session.commit()

            log_activity(
                "DATASET_PREPROCESS",
                user_id=g.current_user.id,
                details=(
                    f"Preprocessed dataset #{dataset.id} ({dataset.filename}): "
                    f"{len(processed_df.index)} rows, {len(processed_df.columns)} cols "
                    f"(missing={missing_strategy}, encoding={encoding_strategy}, scaling={scaling_strategy})"
                ),
                ip_address=request.remote_addr,
            )
        except Exception as exc:
            db.session.rollback()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            current_app.logger.exception(
                "Failed saving preprocessed dataset %s: %s", dataset_id, exc
            )
            return jsonify({
                "error": "Failed to save preprocessed dataset."
            }), 500

    def json_value(val):
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float) and not math.isfinite(val):
            return None
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)

    sample_rows = [
        {str(col): json_value(val) for col, val in row.items()}
        for row in processed_df.head(10).to_dict(orient="records")
    ]

    return jsonify({
        "message": "Dataset preprocessed successfully.",
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": int(len(processed_df.index)),
        "column_count": int(len(processed_df.columns)),
        "columns": [str(c) for c in processed_df.columns],
        "preview": sample_rows,
        "preprocessing_summary": summary,
    }), 200
