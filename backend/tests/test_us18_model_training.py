"""
US-18 Model Training Test Suite
================================
Traceability: FR-4.1 and FR-4.2 (Corrected_Test_Plan.md)

  FR-4.1-UT-1   Display supported algorithms and accept a compatible selection
  FR-4.1-IT-2   Prevent incompatible algorithm/task configuration
  FR-4.2-IT-1   Complete training for a valid configuration
  FR-4.2-IT-2   Handle training failure without corrupting prior runs

Test Plan Section: 3.4 Model Configuration / 3.4 Model Training
"""

import io
import os
import tempfile
import unittest

import pandas as pd
from werkzeug.security import generate_password_hash

from app import create_app
from database.db import db
from ml.trainer import ModelTrainer
from models.dataset import Dataset
from models.model import MLModel
from models.user import User


# ---------------------------------------------------------------------------
# Shared CSV fixtures
# ---------------------------------------------------------------------------

# Small deterministic classification dataset (no missing values)
_CSV_CLASSIFICATION = (
    "age,salary,label\n"
    "25,50000,0\n"
    "30,60000,1\n"
    "35,55000,0\n"
    "28,48000,1\n"
    "40,70000,0\n"
    "22,42000,1\n"
    "33,65000,0\n"
    "45,80000,1\n"
)

# Small deterministic regression dataset
_CSV_REGRESSION = (
    "x1,x2,price\n"
    "1.0,2.0,10.5\n"
    "2.0,3.0,15.0\n"
    "3.0,4.0,20.5\n"
    "4.0,5.0,25.0\n"
    "5.0,6.0,30.5\n"
    "6.0,7.0,35.0\n"
    "7.0,8.0,40.5\n"
    "8.0,9.0,45.0\n"
)


def _read_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


# ===========================================================================
# Unit tests: ModelTrainer class (no HTTP layer)
# ===========================================================================


class TestModelTrainerAlgorithmSupport(unittest.TestCase):
    """FR-4.1-UT-1: ModelTrainer advertises and accepts supported algorithms."""

    def test_supported_algorithms_list_is_not_empty(self):
        """SUPPORTED_ALGORITHMS must not be empty."""
        self.assertGreater(len(ModelTrainer.SUPPORTED_ALGORITHMS), 0)

    def test_supported_algorithms_contains_required_entries(self):
        """Required algorithms from the SRS must appear in the list."""
        required = {
            "logistic_regression",
            "linear_regression",
            "decision_tree",
            "random_forest",
            "knn",
        }
        self.assertTrue(
            required.issubset(set(ModelTrainer.SUPPORTED_ALGORITHMS)),
            f"Missing algorithms: {required - set(ModelTrainer.SUPPORTED_ALGORITHMS)}",
        )

    def test_train_logistic_regression_classification(self):
        """FR-4.1-UT-1: logistic_regression is compatible with classification."""
        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("logistic_regression", "classification", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_linear_regression_regression(self):
        """FR-4.1-UT-1: linear_regression is compatible with regression."""
        df = _read_csv(_CSV_REGRESSION)
        X = df[["x1", "x2"]].values
        y = df["price"].values
        model = ModelTrainer.train("linear_regression", "regression", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_decision_tree_classification(self):
        """FR-4.1-UT-1: decision_tree is compatible with classification."""
        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("decision_tree", "classification", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_decision_tree_regression(self):
        """FR-4.1-UT-1: decision_tree is compatible with regression."""
        df = _read_csv(_CSV_REGRESSION)
        X = df[["x1", "x2"]].values
        y = df["price"].values
        model = ModelTrainer.train("decision_tree", "regression", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_random_forest_classification(self):
        """FR-4.1-UT-1: random_forest is compatible with classification."""
        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("random_forest", "classification", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_random_forest_regression(self):
        """FR-4.1-UT-1: random_forest is compatible with regression."""
        df = _read_csv(_CSV_REGRESSION)
        X = df[["x1", "x2"]].values
        y = df["price"].values
        model = ModelTrainer.train("random_forest", "regression", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_knn_classification(self):
        """FR-4.1-UT-1: knn is compatible with classification."""
        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("knn", "classification", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_train_knn_regression(self):
        """FR-4.1-UT-1: knn is compatible with regression."""
        df = _read_csv(_CSV_REGRESSION)
        X = df[["x1", "x2"]].values
        y = df["price"].values
        model = ModelTrainer.train("knn", "regression", X, y)
        self.assertTrue(hasattr(model, "predict"))

    def test_trained_model_can_predict(self):
        """FR-4.1-UT-1: Trained model must return predictions for unseen samples."""
        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("random_forest", "classification", X, y)
        predictions = model.predict([[32, 58000]])
        self.assertEqual(len(predictions), 1)


class TestModelTrainerIncompatibleConfigs(unittest.TestCase):
    """FR-4.1-IT-2: Incompatible algorithm/task combinations raise ValueError."""

    def _X_y_clf(self):
        df = _read_csv(_CSV_CLASSIFICATION)
        return df[["age", "salary"]].values, df["label"].values

    def test_logistic_regression_with_regression_task_raises(self):
        """FR-4.1-IT-2: logistic_regression is incompatible with regression task."""
        X, y = self._X_y_clf()
        with self.assertRaises(ValueError) as ctx:
            ModelTrainer.train("logistic_regression", "regression", X, y)
        self.assertIn("logistic_regression", str(ctx.exception))

    def test_linear_regression_with_classification_task_raises(self):
        """FR-4.1-IT-2: linear_regression is incompatible with classification task."""
        X, y = self._X_y_clf()
        with self.assertRaises(ValueError) as ctx:
            ModelTrainer.train("linear_regression", "classification", X, y)
        self.assertIn("linear_regression", str(ctx.exception))

    def test_unknown_algorithm_raises(self):
        """FR-4.1-IT-2: An unrecognised algorithm name raises ValueError."""
        X, y = self._X_y_clf()
        with self.assertRaises(ValueError) as ctx:
            ModelTrainer.train("svm", "classification", X, y)
        self.assertIn("svm", str(ctx.exception))

    def test_unknown_task_type_raises(self):
        """FR-4.1-IT-2: An unrecognised task_type raises ValueError."""
        X, y = self._X_y_clf()
        with self.assertRaises(ValueError) as ctx:
            ModelTrainer.train("random_forest", "clustering", X, y)
        self.assertIn("clustering", str(ctx.exception))

    def test_error_message_mentions_supported_algorithms(self):
        """FR-4.1-IT-2: ValueError for unknown algorithm mentions supported options."""
        X, y = self._X_y_clf()
        with self.assertRaises(ValueError) as ctx:
            ModelTrainer.train("neural_network", "classification", X, y)
        msg = str(ctx.exception)
        self.assertIn("Supported", msg)


class TestModelTrainerSave(unittest.TestCase):
    """ModelTrainer.save correctly serialises the estimator to disk."""

    def test_save_writes_file(self):
        """save() must create a readable file at the given path."""
        import joblib

        df = _read_csv(_CSV_CLASSIFICATION)
        X = df[["age", "salary"]].values
        y = df["label"].values
        model = ModelTrainer.train("decision_tree", "classification", X, y)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "subdir", "model.joblib")
            ModelTrainer.save(model, path)
            self.assertTrue(os.path.isfile(path))
            loaded = joblib.load(path)
            preds = loaded.predict([[30, 55000]])
            self.assertEqual(len(preds), 1)


# ===========================================================================
# Integration tests: POST /api/ml/train endpoint
# ===========================================================================


class TestTrainEndpoint(unittest.TestCase):
    """Integration tests for POST /api/ml/train (US-18)."""

    # -------------------------------------------------------------------
    # Test fixtures
    # -------------------------------------------------------------------

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        test_config = {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": (
                f"sqlite:///{os.path.join(self.temp_dir, 'test_us18.db')}"
            ),
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
            "SECRET_KEY": "test-secret-key-us18",
            "UPLOAD_FOLDER": os.path.join(self.temp_dir, "uploads"),
            "SAVED_MODELS_FOLDER": os.path.join(self.temp_dir, "models"),
            "WTF_CSRF_ENABLED": False,
            "MIN_DATA_ROWS": 2,
            "MIN_DATA_COLUMNS": 2,
        }
        os.makedirs(test_config["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(test_config["SAVED_MODELS_FOLDER"], exist_ok=True)

        self.app = create_app(type("TestConfig", (), test_config))
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

            # User A (the primary test user)
            self.user_a = User(
                username="user_a",
                email="usera@example.com",
                password_hash=generate_password_hash("PassA123"),
                role="student",
                is_active=True,
            )
            # User B (for isolation tests)
            self.user_b = User(
                username="user_b",
                email="userb@example.com",
                password_hash=generate_password_hash("PassB123"),
                role="student",
                is_active=True,
            )
            db.session.add_all([self.user_a, self.user_b])
            db.session.commit()

            # Write classification CSV for user_a
            clf_path = os.path.join(test_config["UPLOAD_FOLDER"], "clf.csv")
            with open(clf_path, "w") as fh:
                fh.write(_CSV_CLASSIFICATION)

            # Write regression CSV for user_a
            reg_path = os.path.join(test_config["UPLOAD_FOLDER"], "reg.csv")
            with open(reg_path, "w") as fh:
                fh.write(_CSV_REGRESSION)

            self.clf_dataset = Dataset(
                user_id=self.user_a.id,
                filename="clf.csv",
                file_path=clf_path,
                file_size=len(_CSV_CLASSIFICATION),
                row_count=8,
                column_count=3,
            )
            self.reg_dataset = Dataset(
                user_id=self.user_a.id,
                filename="reg.csv",
                file_path=reg_path,
                file_size=len(_CSV_REGRESSION),
                row_count=8,
                column_count=3,
            )
            db.session.add_all([self.clf_dataset, self.reg_dataset])
            db.session.commit()

            self.clf_dataset_id = self.clf_dataset.id
            self.reg_dataset_id = self.reg_dataset.id

    def _login(self, username, password):
        return self.client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )

    # -------------------------------------------------------------------
    # FR-4.1-UT-1: GET /api/ml/algorithms — supported list
    # -------------------------------------------------------------------

    def test_get_algorithms_returns_200_for_authenticated_user(self):
        """FR-4.1-UT-1: Authenticated user receives algorithm list with HTTP 200."""
        self._login("user_a", "PassA123")
        resp = self.client.get("/api/ml/algorithms")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("classification", data)
        self.assertIn("regression", data)

    def test_get_algorithms_returns_401_for_unauthenticated(self):
        """FR-4.1-UT-1: Unauthenticated request to /algorithms returns 401."""
        resp = self.client.get("/api/ml/algorithms")
        self.assertEqual(resp.status_code, 401)

    def test_algorithms_list_contains_required_algorithms(self):
        """FR-4.1-UT-1: Response must list all SRS-required algorithm names."""
        self._login("user_a", "PassA123")
        resp = self.client.get("/api/ml/algorithms")
        data = resp.get_json()
        clf_items = data.get("classification", [])
        clf_set = {item["id"] if isinstance(item, dict) else item for item in clf_items}
        reg_items = data.get("regression", [])
        reg_set = {item["id"] if isinstance(item, dict) else item for item in reg_items}
        self.assertIn("logistic_regression", clf_set)
        self.assertIn("linear_regression", reg_set)
        self.assertIn("decision_tree", clf_set)
        self.assertIn("random_forest", clf_set)
        self.assertIn("knn", clf_set)

    # -------------------------------------------------------------------
    # FR-4.2-IT-1: Complete training for a valid configuration
    # -------------------------------------------------------------------

    def test_train_classification_random_forest_returns_201(self):
        """FR-4.2-IT-1: Valid classification training request returns HTTP 201."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertIn("model", data)
        self.assertEqual(data["model"]["status"], "completed")
        self.assertEqual(data["model"]["algorithm"], "random_forest")

    def test_train_regression_linear_regression_returns_201(self):
        """FR-4.2-IT-1: Valid regression training request returns HTTP 201."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.reg_dataset_id,
                "algorithm": "linear_regression",
                "task_type": "regression",
                "target_column": "price",
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data["model"]["status"], "completed")

    def test_train_stores_model_in_database(self):
        """FR-4.2-IT-1: Completed training must persist a model record in the DB."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "decision_tree",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 201)
        model_id = resp.get_json()["model"]["id"]

        with self.app.app_context():
            record = db.session.get(MLModel, model_id)
            self.assertIsNotNone(record)
            self.assertEqual(record.status, "completed")
            self.assertIsNotNone(record.model_path)

    def test_train_response_contains_configuration_snapshot(self):
        """FR-4.2-IT-1: Response model must include the training configuration."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "knn",
                "task_type": "classification",
                "target_column": "label",
                "test_size": 0.25,
            },
        )
        self.assertEqual(resp.status_code, 201)
        config = resp.get_json()["model"]["configuration"]
        self.assertEqual(config["algorithm"], "knn")
        self.assertEqual(config["task_type"], "classification")
        self.assertEqual(config["target_column"], "label")
        self.assertAlmostEqual(config["test_size"], 0.25, places=5)

    def test_train_with_preprocessing_succeeds(self):
        """FR-4.2-IT-1: Training with preprocessing options completes successfully."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "logistic_regression",
                "task_type": "classification",
                "target_column": "label",
                "preprocessing": {
                    "missing_values": "mean",
                    "encoding": "none",
                    "scaling": "standard",
                },
            },
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertEqual(data["model"]["status"], "completed")

    def test_train_model_file_is_created_on_disk(self):
        """FR-4.2-IT-1: joblib model file must exist at the reported path after training."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 201)
        model_id = resp.get_json()["model"]["id"]

        with self.app.app_context():
            record = db.session.get(MLModel, model_id)
            self.assertTrue(os.path.isfile(record.model_path))

    def test_get_model_after_training_returns_completed_record(self):
        """FR-6.1-IT-1 / FR-4.2-IT-1: GET /api/ml/models/<id> returns the trained model."""
        self._login("user_a", "PassA123")
        train_resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "decision_tree",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(train_resp.status_code, 201)
        model_id = train_resp.get_json()["model"]["id"]

        get_resp = self.client.get(f"/api/ml/models/{model_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.get_json()["model"]["status"], "completed")

    def test_list_models_includes_trained_model(self):
        """FR-6.1-IT-1: GET /api/ml/models lists the user's trained model."""
        self._login("user_a", "PassA123")
        self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "knn",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        resp = self.client.get("/api/ml/models")
        self.assertEqual(resp.status_code, 200)
        models = resp.get_json()["models"]
        self.assertGreater(len(models), 0)

    # -------------------------------------------------------------------
    # FR-4.1-IT-2: Reject incompatible algorithm/task configurations
    # -------------------------------------------------------------------

    def test_logistic_regression_with_regression_task_returns_400(self):
        """FR-4.1-IT-2: logistic_regression + regression task rejected with 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "logistic_regression",
                "task_type": "regression",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_linear_regression_with_classification_task_returns_400(self):
        """FR-4.1-IT-2: linear_regression + classification task rejected with 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "linear_regression",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_unsupported_algorithm_returns_400(self):
        """FR-4.1-IT-2: An unsupported algorithm name is rejected with HTTP 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "svm",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 400)
        error_msg = resp.get_json().get("error", "")
        self.assertIn("svm", error_msg)

    def test_invalid_task_type_returns_400(self):
        """FR-4.1-IT-2: An invalid task_type is rejected with HTTP 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "clustering",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 400)

    def test_missing_required_fields_returns_400(self):
        """FR-4.1-IT-2: Omitting required fields returns HTTP 400."""
        self._login("user_a", "PassA123")
        # Missing algorithm and task_type
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_nonexistent_target_column_returns_400(self):
        """FR-4.1-IT-2: A target column that doesn't exist in the CSV returns 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "nonexistent_col",
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("nonexistent_col", resp.get_json().get("error", ""))

    def test_invalid_test_size_returns_400(self):
        """FR-3.3-UT-2 / FR-4.1-IT-2: Invalid test_size value returns 400."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
                "test_size": 1.5,  # invalid
            },
        )
        self.assertEqual(resp.status_code, 400)

    # -------------------------------------------------------------------
    # FR-4.2-IT-2: Failure does not corrupt prior completed runs
    # -------------------------------------------------------------------

    def test_prior_completed_run_intact_after_failed_training(self):
        """FR-4.2-IT-2: A failed training attempt does not corrupt a prior completed run."""
        self._login("user_a", "PassA123")

        # First, complete a valid training run
        good_resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "decision_tree",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(good_resp.status_code, 201)
        prior_model_id = good_resp.get_json()["model"]["id"]

        # Then deliberately trigger a failed training (incompatible algorithm+task)
        bad_resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "logistic_regression",
                "task_type": "regression",  # incompatible
                "target_column": "label",
            },
        )
        self.assertEqual(bad_resp.status_code, 400)

        # The prior completed run must still be accessible and intact
        get_resp = self.client.get(f"/api/ml/models/{prior_model_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.get_json()["model"]["status"], "completed")

    def test_models_list_still_shows_completed_run_after_failed_training(self):
        """FR-4.2-IT-2: GET /api/ml/models still includes prior runs after a failure."""
        self._login("user_a", "PassA123")

        self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "knn",
                "task_type": "classification",
                "target_column": "label",
            },
        )

        # Trigger an invalid training attempt
        self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "svm",  # unsupported, rejected before DB write
                "task_type": "classification",
                "target_column": "label",
            },
        )

        resp = self.client.get("/api/ml/models")
        models = resp.get_json()["models"]
        completed = [m for m in models if m["status"] == "completed"]
        self.assertGreater(len(completed), 0)

    # -------------------------------------------------------------------
    # Security: ownership and authentication
    # -------------------------------------------------------------------

    def test_train_unauthenticated_returns_401(self):
        """SEC-AZ: Unauthenticated training request returns HTTP 401."""
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 401)

    def test_train_on_another_users_dataset_returns_404(self):
        """SEC-AZ: User B cannot train on User A's dataset (returns 404)."""
        self._login("user_b", "PassB123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,  # owned by user_a
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 404)

    def test_train_on_nonexistent_dataset_returns_404(self):
        """SEC-AZ: Training on a non-existent dataset_id returns 404."""
        self._login("user_a", "PassA123")
        resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": 999999,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(resp.status_code, 404)

    def test_model_not_visible_to_other_user(self):
        """SEC-AZ: User B cannot retrieve User A's trained model record."""
        self._login("user_a", "PassA123")
        train_resp = self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "decision_tree",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        self.assertEqual(train_resp.status_code, 201)
        model_id = train_resp.get_json()["model"]["id"]

        # Log out User A, log in User B
        self.client.post("/api/auth/logout")
        self._login("user_b", "PassB123")
        get_resp = self.client.get(f"/api/ml/models/{model_id}")
        self.assertEqual(get_resp.status_code, 404)

    def test_models_list_isolated_per_user(self):
        """SEC-AZ: User B's /models list does not include User A's models."""
        self._login("user_a", "PassA123")
        self.client.post(
            "/api/ml/train",
            json={
                "dataset_id": self.clf_dataset_id,
                "algorithm": "random_forest",
                "task_type": "classification",
                "target_column": "label",
            },
        )
        # Switch to user_b
        self.client.post("/api/auth/logout")
        self._login("user_b", "PassB123")
        resp = self.client.get("/api/ml/models")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["models"], [])


if __name__ == "__main__":
    unittest.main()
