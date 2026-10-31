import io
import os
import shutil
import tempfile
import unittest
from uuid import uuid4

from app import create_app
from database.db import db
from ml.preprocessor import DataPreprocessor
from models.dataset import Dataset
from models.evaluation import EvaluationResult
from models.model import MLModel
from models.user import User


class TestTrainTestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="ml-evaluator-split-tests-")

        class TestConfig:
            TESTING = True
            SECRET_KEY = "split-test-key"
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            UPLOAD_FOLDER = os.path.join(cls.temp_dir.name, "uploads")
            SAVED_MODELS_FOLDER = os.path.join(cls.temp_dir.name, "models")
            MIN_DATA_ROWS = 2
            MIN_DATA_COLUMNS = 2

        cls.app = create_app(TestConfig)
        cls.context = cls.app.app_context()
        cls.context.push()
        db.create_all()

    @classmethod
    def tearDownClass(cls):
        db.session.remove()
        db.drop_all()
        cls.context.pop()
        cls.temp_dir.cleanup()

    def setUp(self):
        db.session.remove()
        self.client = self.app.test_client()
        suffix = uuid4().hex
        self.username = f"split_{suffix[:12]}"
        self.client.post("/api/auth/register", json={
            "username": self.username,
            "email": f"{suffix}@example.com",
            "password": "StrongPass123!",
        })
        login = self.client.post("/api/auth/login", json={
            "username": self.username,
            "password": "StrongPass123!",
        })
        self.assertEqual(login.status_code, 200)
        self.user = User.query.filter_by(username=self.username).one()

    def tearDown(self):
        db.session.rollback()
        EvaluationResult.query.delete()
        MLModel.query.delete()
        Dataset.query.delete()
        User.query.delete()
        db.session.commit()
        for folder in (self.app.config["UPLOAD_FOLDER"], self.app.config["SAVED_MODELS_FOLDER"]):
            if os.path.isdir(folder):
                shutil.rmtree(folder)

    @staticmethod
    def csv_rows(count):
        rows = ["feature,target"]
        rows.extend(f"{index},{'yes' if index % 2 else 'no'}" for index in range(count))
        return ("\n".join(rows) + "\n").encode("utf-8")

    def upload(self, count=20):
        response = self.client.post(
            "/api/dataset/upload",
            data={"file": (io.BytesIO(self.csv_rows(count)), "data.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.json["dataset_id"]

    def evaluate(self, dataset_id, endpoint="evaluate", **overrides):
        payload = {
            "dataset_id": dataset_id,
            "target_column": "target",
            "algorithm": "decision_tree",
            "task_type": "classification",
        }
        payload.update(overrides)
        return self.client.post(f"/api/ml/{endpoint}", json=payload)

    def test_split_helper_defaults_to_80_20_and_is_deterministic(self):
        X = list(range(10))
        y = list(range(10))
        first = DataPreprocessor.split(X, y)
        second = DataPreprocessor.split(X, y)
        self.assertEqual((len(first[0]), len(first[1])), (8, 2))
        self.assertEqual(first, second)

    def test_split_helper_uses_custom_70_30_ratio(self):
        train_x, test_x, train_y, test_y = DataPreprocessor.split(
            list(range(20)), list(range(20)), test_size=0.3
        )
        self.assertEqual((len(train_x), len(test_x)), (14, 6))
        self.assertEqual((len(train_y), len(test_y)), (14, 6))

    def test_split_helper_rejects_invalid_ratio_types_and_boundaries(self):
        for value in ("0.2", True, 0, 1, -0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "test_size"):
                    DataPreprocessor.split([1, 2], [1, 2], test_size=value)

    def test_endpoint_defaults_to_real_80_20_partitions(self):
        dataset_id = self.upload()

        response = self.evaluate(dataset_id)

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        self.assertEqual(response.json["split"]["requested_test_size"], 0.2)
        self.assertEqual(response.json["split"]["train_rows"], 16)
        self.assertEqual(response.json["split"]["test_rows"], 4)
        model = db.session.get(MLModel, response.json["model_id"])
        self.assertEqual(model.config_json["test_size"], 0.2)
        self.assertIsNotNone(db.session.get(EvaluationResult, response.json["evaluation_id"]))

    def test_endpoint_uses_custom_70_30_partition_sizes(self):
        dataset_id = self.upload()

        response = self.evaluate(dataset_id, test_size=0.3)

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        self.assertEqual(response.json["split"]["requested_test_size"], 0.3)
        self.assertEqual(response.json["split"]["train_rows"], 14)
        self.assertEqual(response.json["split"]["test_rows"], 6)

    def test_train_endpoint_uses_the_same_selected_split(self):
        dataset_id = self.upload()

        response = self.evaluate(dataset_id, endpoint="train", test_size=0.3)

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        self.assertEqual(response.json["split"]["train_rows"], 14)
        self.assertEqual(response.json["split"]["test_rows"], 6)

    def test_endpoint_rejects_invalid_ratio_values(self):
        dataset_id = self.upload()
        for value in ("30%", True, 0, 1, -0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(value=value):
                response = self.evaluate(dataset_id, test_size=value)
                self.assertEqual(response.status_code, 400)
                self.assertIn("test_size", response.json["error"])

    def test_endpoint_rejects_splits_without_both_partitions(self):
        dataset_id = self.upload(count=5)

        response = self.evaluate(dataset_id, test_size=0.99)

        self.assertEqual(response.status_code, 400)
        self.assertIn("training or testing", response.json["error"].lower())

    def test_endpoint_enforces_algorithm_sample_minimum(self):
        dataset_id = self.upload(count=7)

        response = self.evaluate(dataset_id, algorithm="knn", test_size=0.3)

        self.assertEqual(response.status_code, 400)
        self.assertIn("requires at least 5", response.json["error"])

    def test_evaluate_endpoint_requires_authentication(self):
        unauthenticated = self.app.test_client()
        response = unauthenticated.post("/api/ml/evaluate", json={"dataset_id": 1})
        self.assertEqual(response.status_code, 401)

    def test_evaluate_endpoint_denies_another_users_dataset(self):
        dataset_id = self.upload()
        other_client = self.app.test_client()
        token = uuid4().hex
        other_client.post("/api/auth/register", json={
            "username": f"other_{token[:12]}",
            "email": f"{token}@example.com",
            "password": "StrongPass123!",
        })
        other_client.post("/api/auth/login", json={
            "username": f"other_{token[:12]}",
            "password": "StrongPass123!",
        })

        response = other_client.post("/api/ml/evaluate", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "algorithm": "decision_tree",
            "task_type": "classification",
            "test_size": 0.3,
        })

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["error"], "Dataset not found")


    def test_us17_catalog_and_every_algorithm_task_pair_are_supported(self):
        from ml.trainer import ModelTrainer
        catalog = self.client.get("/api/ml/algorithms")
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual({key: {item["id"] for item in value} for key, value in catalog.json.items()}, {
            "classification": {"logistic_regression", "decision_tree", "random_forest", "knn"},
            "regression": {"linear_regression", "decision_tree", "random_forest", "knn"},
        })
        for task, items in catalog.json.items():
            for item in items:
                with self.subTest(task=task, algorithm=item["id"]):
                    self.assertIsNotNone(ModelTrainer.create_estimator(item["id"], task))
        bad = self.evaluate(self.upload(), algorithm="linear_regression", task_type="classification")
        self.assertEqual(bad.status_code, 400)

    def test_us20_compare_and_download_artifacts_for_owner(self):
        dataset_id = self.upload()
        first, second = self.evaluate(dataset_id), self.evaluate(dataset_id, algorithm="random_forest")
        self.assertEqual(first.status_code, 201, first.json)
        self.assertEqual(second.status_code, 201, second.json)
        compared = self.client.post("/api/ml/compare", json={"evaluation_ids": [first.json["evaluation_id"], second.json["evaluation_id"]]})
        self.assertEqual(compared.status_code, 200, compared.json)
        self.assertEqual(len(compared.json["evaluations"]), 2)
        report = self.client.get(f"/api/ml/evaluations/{first.json['evaluation_id']}/report")
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.mimetype, "application/pdf")
        model = self.client.get(f"/api/ml/models/{first.json['model_id']}/download")
        self.assertEqual(model.status_code, 200)
        self.assertTrue(model.data)

    def test_us19a_instructor_sees_only_shared_runs_and_cannot_write(self):
        dataset_id = self.upload()
        private_run = self.evaluate(dataset_id)
        self.assertEqual(private_run.status_code, 201, private_run.json)

        peer_client = self.app.test_client()
        peer_suffix = uuid4().hex
        self.assertEqual(peer_client.post("/api/auth/register", json={
            "username": f"peer_{peer_suffix[:12]}",
            "email": f"{peer_suffix}@example.com",
            "password": "StrongPass123!",
        }).status_code, 201)
        self.assertEqual(peer_client.post("/api/auth/login", json={
            "username": f"peer_{peer_suffix[:12]}",
            "password": "StrongPass123!",
        }).status_code, 200)
        peer_upload = peer_client.post(
            "/api/dataset/upload",
            data={"file": (io.BytesIO(self.csv_rows(20)), "peer.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(peer_upload.status_code, 201, peer_upload.json)
        peer_dataset_id = peer_upload.json["dataset_id"]
        run = peer_client.post("/api/ml/evaluate", json={
            "dataset_id": peer_dataset_id,
            "target_column": "target",
            "algorithm": "decision_tree",
            "task_type": "classification",
        })
        second_shared_run = peer_client.post("/api/ml/evaluate", json={
            "dataset_id": peer_dataset_id,
            "target_column": "target",
            "algorithm": "random_forest",
            "task_type": "classification",
        })
        self.assertEqual(run.status_code, 201, run.json)
        self.assertEqual(second_shared_run.status_code, 201, second_shared_run.json)
        evaluation_id = run.json["evaluation_id"]
        self.assertEqual(peer_client.patch(f"/api/ml/evaluations/{evaluation_id}/sharing", json={"shared_with_instructors": True}).status_code, 200)
        second_shared_id = second_shared_run.json["evaluation_id"]
        self.assertEqual(peer_client.patch(f"/api/ml/evaluations/{second_shared_id}/sharing", json={"shared_with_instructors": True}).status_code, 200)
        self.user.role = "instructor"
        db.session.commit()
        review = self.client.get("/api/ml/instructor/evaluations")
        self.assertEqual(review.status_code, 200, review.json)
        self.assertEqual({item["id"] for item in review.json["evaluations"]}, {evaluation_id, second_shared_id})
        detail = self.client.get(f"/api/ml/instructor/evaluations/{evaluation_id}")
        self.assertEqual(detail.status_code, 200, detail.json)
        self.assertIn("metrics", detail.json["evaluation"])
        self.assertIn("confusion_matrix", detail.json["evaluation"])
        self.assertNotIn("configuration", detail.json["evaluation"])
        self.assertEqual(self.client.get(f"/api/ml/instructor/evaluations/{private_run.json['evaluation_id']}").status_code, 404)
        missing_detail = self.client.get("/api/ml/instructor/evaluations/999999")
        self.assertEqual(missing_detail.status_code, 404)
        self.assertEqual(missing_detail.json, {"error": "Evaluation not found."})
        compared = self.client.post("/api/ml/instructor/compare", json={"evaluation_ids": [evaluation_id, second_shared_id]})
        self.assertEqual(compared.status_code, 200, compared.json)
        self.assertEqual(len(compared.json["evaluations"]), 2)
        private_comparison = self.client.post("/api/ml/instructor/compare", json={"evaluation_ids": [evaluation_id, private_run.json["evaluation_id"]]})
        self.assertEqual(private_comparison.status_code, 404)
        self.assertEqual(self.client.post("/api/ml/compare", json={"evaluation_ids": [evaluation_id, second_shared_id]}).status_code, 403)
        self.assertEqual(self.client.get("/api/dataset").status_code, 403)
        self.assertEqual(self.client.get(f"/api/dataset/{dataset_id}").status_code, 403)
        self.assertEqual(self.client.get(f"/api/dataset/{dataset_id}/preview").status_code, 403)
        self.assertEqual(self.client.get(f"/api/dataset/{dataset_id}/statistics").status_code, 403)
        self.assertEqual(self.client.get("/api/ml/models").status_code, 403)
        self.assertEqual(self.client.get("/api/ml/evaluations").status_code, 403)
        self.assertEqual(self.client.get(f"/api/ml/evaluations/{evaluation_id}").status_code, 403)
        self.assertEqual(self.client.get(f"/api/ml/models/{private_run.json['model_id']}/download").status_code, 403)
        self.assertEqual(self.client.get(f"/api/ml/evaluations/{private_run.json['evaluation_id']}/report").status_code, 403)
        self.assertEqual(self.client.get(f"/api/ml/evaluations/{evaluation_id}/report").status_code, 403)
        self.assertEqual(self.client.post("/api/ml/evaluate", json={"dataset_id": dataset_id}).status_code, 403)
        self.assertEqual(self.client.post("/api/ml/train", json={"dataset_id": dataset_id}).status_code, 403)
        upload = self.client.post("/api/dataset/upload", data={"file": (io.BytesIO(self.csv_rows(20)), "other.csv")}, content_type="multipart/form-data")
        self.assertEqual(upload.status_code, 403)
        replace = self.client.post(f"/api/dataset/{dataset_id}/replace", data={"file": (io.BytesIO(self.csv_rows(20)), "replacement.csv")}, content_type="multipart/form-data")
        self.assertEqual(replace.status_code, 403)
        self.assertEqual(self.client.post(f"/api/dataset/{dataset_id}/preprocess", json={}).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/admin/datasets/{dataset_id}").status_code, 403)
        self.assertEqual(self.client.patch(f"/api/ml/evaluations/{evaluation_id}/sharing", json={"shared_with_instructors": False}).status_code, 403)

    def test_us17_all_supported_estimators_fit_for_their_task_type(self):
        import pandas as pd
        from ml.trainer import ModelTrainer
        frame = pd.DataFrame({"numeric": list(range(40)), "category": ["a", "b"] * 20})
        targets = {"classification": pd.Series(["yes", "no"] * 20), "regression": pd.Series([float(i * 2 + 1) for i in range(40)])}
        for task, items in ModelTrainer.algorithms_for().items():
            for item in items:
                with self.subTest(task=task, algorithm=item["id"]):
                    fitted = ModelTrainer.train(item["id"], task, frame.iloc[:32], targets[task].iloc[:32])
                    self.assertEqual(len(fitted.predict(frame.iloc[32:])), 8)

    def test_us20_comparison_rejects_invalid_and_missing_run_requests(self):
        first = self.evaluate(self.upload())
        self.assertEqual(first.status_code, 201)
        self.assertEqual(self.client.post("/api/ml/compare", json={"evaluation_ids": [first.json["evaluation_id"]]}).status_code, 400)
        self.assertEqual(self.client.post("/api/ml/compare", json={"evaluation_ids": [first.json["evaluation_id"], 999999]}).status_code, 404)

    def test_us15_target_column_is_validated_and_bad_requests_do_not_save_runs(self):
        dataset_id = self.upload()
        before = EvaluationResult.query.count()
        missing = self.evaluate(dataset_id, target_column="not_a_column")
        self.assertEqual(missing.status_code, 400)
        self.assertIn("target column", missing.json["error"].lower())
        self.assertEqual(EvaluationResult.query.count(), before)

    def test_us18_failed_training_does_not_remove_prior_successful_run(self):
        dataset_id = self.upload()
        first = self.evaluate(dataset_id)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(self.evaluate(dataset_id, algorithm="unsupported").status_code, 400)
        history = self.client.get("/api/ml/evaluations")
        self.assertEqual(history.status_code, 200)
        self.assertEqual([item["id"] for item in history.json["evaluations"]], [first.json["evaluation_id"]])

    def test_us20_comparison_rejects_mixed_classification_and_regression(self):
        classification = self.evaluate(self.upload())
        regression_csv = "feature,target\n" + "\n".join(f"{i},{i * 2 + 1}" for i in range(20)) + "\n"
        uploaded = self.client.post("/api/dataset/upload", data={"file": (io.BytesIO(regression_csv.encode()), "regression.csv")}, content_type="multipart/form-data")
        self.assertEqual(uploaded.status_code, 201)
        regression = self.evaluate(uploaded.json["dataset_id"], algorithm="linear_regression", task_type="regression")
        self.assertEqual(classification.status_code, 201)
        self.assertEqual(regression.status_code, 201, regression.json)
        response = self.client.post("/api/ml/compare", json={"evaluation_ids": [classification.json["evaluation_id"], regression.json["evaluation_id"]]})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
