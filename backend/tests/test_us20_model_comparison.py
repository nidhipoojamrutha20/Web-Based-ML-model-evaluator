import io
import os
import tempfile
import unittest
from uuid import uuid4

from app import create_app
from database.db import db
from models.dataset import Dataset
from models.evaluation import EvaluationResult
from models.model import MLModel
from models.user import User


class TestUS20ModelComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="ml-evaluator-us20-")

        class TestConfig:
            TESTING = True
            SECRET_KEY = "us20-test-key"
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            UPLOAD_FOLDER = os.path.join(cls.temp_dir.name, "uploads")
            SAVED_MODELS_FOLDER = os.path.join(cls.temp_dir.name, "models")

        cls.app = create_app(TestConfig)
        cls.context = cls.app.app_context()
        cls.context.push()

    @classmethod
    def tearDownClass(cls):
        db.session.remove()
        db.drop_all()
        cls.context.pop()
        cls.temp_dir.cleanup()

    def setUp(self):
        self.client = self.app.test_client()
        self.username = f"us20_{uuid4().hex[:12]}"
        self.password = "StrongPass123!"
        response = self.client.post("/api/auth/register", json={
            "username": self.username,
            "email": f"{self.username}@example.com",
            "password": self.password,
        })
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.post("/api/auth/login", json={
            "username": self.username, "password": self.password,
        }).status_code, 200)
        self.upload_classification()

    def tearDown(self):
        db.session.rollback()
        EvaluationResult.query.delete()
        MLModel.query.delete()
        Dataset.query.delete()
        User.query.delete()
        db.session.commit()
        for folder in (self.app.config["UPLOAD_FOLDER"], self.app.config["SAVED_MODELS_FOLDER"]):
            if os.path.isdir(folder):
                for root, _dirs, files in os.walk(folder, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))

    def upload_classification(self):
        rows = [f"{i},{(i * 7) % 13},{'yes' if i % 2 else 'no'}" for i in range(40)]
        csv_data = "x1,x2,label\n" + "\n".join(rows) + "\n"
        response = self.client.post("/api/dataset/upload", data={
            "file": (io.BytesIO(csv_data.encode()), "classification.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201, response.json)
        self.dataset_id = response.json["dataset_id"]

    def train(self, algorithm="decision_tree", task="classification", dataset_id=None, target="label"):
        return self.client.post("/api/ml/train", json={
            "dataset_id": dataset_id or self.dataset_id,
            "target_column": target, "algorithm": algorithm,
            "task_type": task, "test_size": 0.25, "random_state": 42,
        })

    def test_train_persists_and_lists_real_evaluation_runs(self):
        response = self.train()
        self.assertEqual(response.status_code, 201, response.json)
        model = MLModel.query.get(response.json["model"]["id"])
        self.assertTrue(os.path.isfile(model.model_path))
        # US-18's canonical persisted status is "completed"; US-20 uses the
        # same status for its persisted comparison runs.
        self.assertEqual(model.status, "completed")
        self.assertEqual(model.config_json["test_size"], 0.25)
        self.assertEqual(set(response.json["evaluation"]["metrics"]), {"accuracy", "precision", "recall", "f1"})
        listed = self.client.get("/api/ml/models")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json["models"][0]["id"], model.id)

    def test_comparison_returns_actual_metrics_chart_and_rejects_incompatible_task(self):
        one = self.train("decision_tree").json
        two = self.train("random_forest").json
        response = self.client.post("/api/ml/compare", json={"model_ids": [one["model"]["id"], two["model"]["id"]]})
        self.assertEqual(response.status_code, 200, response.json)
        data = response.json
        self.assertEqual(data["task_type"], "classification")
        # Chart labels include run IDs so repeated algorithms remain distinguishable.
        # Model rows, chart labels, and metric points all retain the requested ID order.
        self.assertEqual(
            [model["id"] for model in data["models"]],
            [one["model"]["id"], two["model"]["id"]],
        )
        self.assertEqual(
            data["chart"]["labels"],
            [model["run_label"] for model in data["models"]],
        )
        self.assertEqual(
            [point["model_id"] for point in data["chart"]["metrics"]["accuracy"]],
            [one["model"]["id"], two["model"]["id"]],
        )
        self.assertEqual(data["chart"]["metrics"]["accuracy"][0]["value"], one["evaluation"]["metrics"]["accuracy"])
        self.assertEqual(data["chart"]["metrics"]["f1"][1]["value"], two["evaluation"]["metrics"]["f1"])
        self.assertEqual(self.client.post("/api/ml/compare", json={"model_ids": [one["model"]["id"]]}).status_code, 400)

    def test_model_and_pdf_downloads_are_real_and_have_safe_headers(self):
        trained = self.train().json
        model_id = trained["model"]["id"]
        evaluation_id = trained["evaluation"]["id"]
        model = self.client.get(f"/api/ml/models/{model_id}/download")
        self.assertEqual(model.status_code, 200)
        self.assertTrue(model.data.startswith(b"\x80"))
        self.assertIn("attachment", model.headers["Content-Disposition"])
        self.assertEqual(model.mimetype, "application/octet-stream")
        report = self.client.get(f"/api/ml/evaluations/{evaluation_id}/report")
        self.assertEqual(report.status_code, 200)
        self.assertTrue(report.data.startswith(b"%PDF-"))
        self.assertIn(b"Decision Tree - Classification", report.data)
        self.assertIn(b"classification.csv", report.data)
        self.assertIn(b"Task type: classification", report.data)
        self.assertIn(b"Train/test split:", report.data)
        self.assertIn(b"Evaluation metrics", report.data)
        self.assertIn(b"Limitations:", report.data)
        self.assertEqual(report.mimetype, "application/pdf")

    def test_authentication_and_cross_user_ownership(self):
        trained = self.train().json
        model_id = trained["model"]["id"]
        evaluation_id = trained["evaluation"]["id"]
        anonymous = self.app.test_client()
        self.assertEqual(anonymous.get(f"/api/ml/models/{model_id}/download").status_code, 401)
        other = self.app.test_client()
        other_name = f"other_{uuid4().hex[:12]}"
        other.post("/api/auth/register", json={"username": other_name, "email": f"{other_name}@example.com", "password": self.password})
        other.post("/api/auth/login", json={"username": other_name, "password": self.password})
        self.assertEqual(other.get(f"/api/ml/models/{model_id}/download").status_code, 404)
        self.assertEqual(other.get(f"/api/ml/evaluations/{evaluation_id}/report").status_code, 404)
        self.assertEqual(other.post("/api/ml/compare", json={"model_ids": [model_id, model_id + 1]}).status_code, 404)

    def test_missing_and_corrupted_artifacts_return_clear_errors(self):
        trained = self.train().json
        model = MLModel.query.get(trained["model"]["id"])
        os.remove(model.model_path)
        missing = self.client.get(f"/api/ml/models/{model.id}/download")
        self.assertEqual(missing.status_code, 404)
        with open(model.model_path, "wb") as artifact:
            artifact.write(b"not the saved model")
        corrupted = self.client.get(f"/api/ml/models/{model.id}/download")
        self.assertEqual(corrupted.status_code, 500)
        self.assertIn("corrupted", corrupted.json["error"])
        nonexistent = self.client.get("/api/ml/evaluations/999999/report")
        self.assertEqual(nonexistent.status_code, 404)

    def test_task_type_compatibility_and_frontend_handlers_are_wired(self):
        first = self.train().json
        rows = [f"{i},{(i * 7) % 13},{i * 1.5 + 2}" for i in range(40)]
        response = self.client.post("/api/dataset/upload", data={
            "file": (io.BytesIO(("x1,x2,target\n" + "\n".join(rows) + "\n").encode()), "regression.csv")
        }, content_type="multipart/form-data")
        regression = self.train("linear_regression", "regression", response.json["dataset_id"], "target")
        self.assertEqual(regression.status_code, 201, regression.json)
        regression_two = self.train("decision_tree", "regression", response.json["dataset_id"], "target")
        self.assertEqual(regression_two.status_code, 201, regression_two.json)
        regression_comparison = self.client.post("/api/ml/compare", json={
            "model_ids": [regression.json["model"]["id"], regression_two.json["model"]["id"]]
        })
        self.assertEqual(regression_comparison.status_code, 200, regression_comparison.json)
        self.assertEqual(regression_comparison.json["metric_names"], ["mae", "mse", "rmse", "r2"])
        rejected = self.client.post("/api/ml/compare", json={"model_ids": [first["model"]["id"], regression.json["model"]["id"]]})
        self.assertEqual(rejected.status_code, 400)
        script = open(os.path.join(os.path.dirname(__file__), "../../frontend/js/app.js"), encoding="utf-8").read()
        self.assertIn('apiRequest("/ml/compare"', script)
        self.assertIn("downloadFile(`/ml/models/", script)
        self.assertIn("downloadFile(`/ml/evaluations/", script)
        self.assertIn("textContent = `${model.run_label", script)


if __name__ == "__main__":
    unittest.main()
