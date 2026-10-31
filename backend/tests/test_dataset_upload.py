
import io
import os
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from app import create_app
from database.db import db
from models.dataset import Dataset
from models.user import User
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import generate_password_hash


class TestDatasetUpload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(
            prefix="ml-evaluator-upload-tests-"
        )

        class TestConfig:
            TESTING = True
            SECRET_KEY = "test-only"
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            UPLOAD_FOLDER = os.path.join(cls.temp_dir.name, "uploads")
            SAVED_MODELS_FOLDER = os.path.join(cls.temp_dir.name, "models")
            MAX_CONTENT_LENGTH = 16 * 1024 * 1024
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
        self.user = User(
            username=f"u_{uuid4().hex[:12]}",
            email=f"{uuid4().hex}@example.com",
            password_hash=generate_password_hash("test-password-123"),
        )
        db.session.add(self.user)
        db.session.commit()

    def tearDown(self):
        db.session.query(Dataset).delete()
        db.session.query(User).delete()
        db.session.commit()

        for name in os.listdir(self.app.config["UPLOAD_FOLDER"]):
            os.remove(os.path.join(self.app.config["UPLOAD_FOLDER"], name))

    def authenticate(self, user_id=None):
        if user_id is None:
            response = self.client.post(
                "/api/auth/login",
                json={
                    "username": self.user.username,
                    "password": "test-password-123",
                },
            )
            self.assertEqual(response.status_code, 200)
            return

        with self.client.session_transaction() as test_session:
            test_session["user_id"] = user_id
            test_session["session_id"] = "unknown-session-id"

    def upload(self, contents, filename="data.csv", extra_fields=None):
        data = {"file": (io.BytesIO(contents), filename)}

        if extra_fields:
            data.update(extra_fields)

        return self.client.post(
            "/api/dataset/upload",
            data=data,
            content_type="multipart/form-data",
        )

    def test_valid_upload_persists_metadata_and_uses_unique_server_path(self):
        self.authenticate()
        contents = b"name,score\nAda,98\nLin,95\nGrace,97\n"

        response = self.upload(contents, "../../students.csv")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["filename"], "../../students.csv")
        self.assertEqual(response.json["row_count"], 3)
        self.assertEqual(response.json["column_count"], 2)
        self.assertEqual(response.json["columns"], ["name", "score"])

        dataset = db.session.get(Dataset, response.json["dataset_id"])
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.user_id, self.user.id)
        self.assertEqual(dataset.filename, "../../students.csv")
        self.assertEqual(dataset.file_size, len(contents))
        self.assertEqual(dataset.row_count, 3)
        self.assertEqual(dataset.column_count, 2)
        self.assertEqual(dataset.metadata_json, {"columns": ["name", "score"]})
        self.assertEqual(
            os.path.dirname(dataset.file_path),
            os.path.abspath(self.app.config["UPLOAD_FOLDER"]),
        )
        self.assertTrue(os.path.isfile(dataset.file_path))

        with open(dataset.file_path, "rb") as saved:
            self.assertEqual(saved.read(), contents)

        second = self.upload(contents, "students.csv")
        self.assertEqual(second.status_code, 201)
        self.assertNotEqual(
            dataset.file_path,
            db.session.get(Dataset, second.json["dataset_id"]).file_path,
        )

    def test_csv_meeting_the_two_row_two_column_minimum_is_accepted(self):
        self.authenticate()

        response = self.upload(b"feature,target\n1,yes\n2,no\n")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["row_count"], 2)
        self.assertEqual(response.json["column_count"], 2)

    def test_upload_requires_a_verified_session_user(self):
        response = self.upload(b"a\n1\n")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(Dataset.query.count(), 0)

    def test_request_user_id_cannot_override_authenticated_owner(self):
        self.authenticate()
        other_user = User(
            username=f"other_{uuid4().hex[:12]}",
            email=f"{uuid4().hex}@example.com",
            password_hash=generate_password_hash("other-password-123"),
        )
        db.session.add(other_user)
        db.session.commit()

        response = self.upload(
            b"feature,target\n1,yes\n2,no\n",
            extra_fields={"user_id": str(other_user.id)},
        )

        self.assertEqual(response.status_code, 201)
        dataset = db.session.get(Dataset, response.json["dataset_id"])
        self.assertEqual(dataset.user_id, self.user.id)

    def test_unknown_session_user_is_rejected(self):
        self.authenticate(user_id=self.user.id + 1000)

        response = self.upload(b"a\n1\n")

        self.assertEqual(response.status_code, 401)

    def test_missing_file_is_rejected(self):
        self.authenticate()
        response = self.client.post("/api/dataset/upload", data={})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Choose a CSV", response.json["error"])

    def test_non_csv_extension_is_rejected(self):
        self.authenticate()
        response = self.upload(b"a\n1\n", "data.txt")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json["error"])

    def test_empty_file_is_rejected(self):
        self.authenticate()
        response = self.upload(b"")

        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json["error"].lower())

    def test_invalid_csvs_are_rejected_with_clear_errors(self):
        self.authenticate()
        invalid_files = [
            (b"\n\n", "no header"),
            (b"name,\nAda,1\n", "non-empty header"),
            (b"name,name\nAda,Lovelace\n", "unique"),
            (b"name,score\nAda\n", "same number of fields"),
            (b'name,score\n"Ada,98\n', "malformed"),
            (b"name,score\n", "at least 2 non-empty data rows"),
            (b"name,score\nAda,1\n", "at least 2 non-empty data rows"),
            (b"name,score\n,\n,\n", "empty data row"),
            (b"name,score\nNA,NA\nNA,NA\n", "at least 2 non-empty data rows"),
            (b"feature,target\n1,\n2,\n", "columns with data"),
            (b"name\nAda\nLin\n", "at least 2 columns"),
            (b"name,score\nAda,1\x00\nLin,2\n", "NUL"),
            (b"name\n\xff\n", "UTF-8"),
        ]

        for contents, expected_message in invalid_files:
            with self.subTest(contents=contents):
                response = self.upload(contents)
                self.assertEqual(response.status_code, 400)
                self.assertIn(
                    expected_message.lower(),
                    response.json["error"].lower(),
                )

        self.assertEqual(Dataset.query.count(), 0)
        self.assertEqual(os.listdir(self.app.config["UPLOAD_FOLDER"]), [])

    def test_over_limit_upload_returns_json_413(self):
        self.authenticate()
        original_limit = self.app.config["MAX_CONTENT_LENGTH"]
        self.app.config["MAX_CONTENT_LENGTH"] = 128

        try:
            response = self.upload(b"a\n" + b"x" * 256 + b"\n")
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = original_limit

        self.assertEqual(response.status_code, 413)
        self.assertIn(
            "128 bytes request-size limit",
            response.json["error"].lower(),
        )
        self.assertEqual(Dataset.query.count(), 0)
        self.assertEqual(os.listdir(self.app.config["UPLOAD_FOLDER"]), [])

    def test_database_failure_rolls_back_and_removes_uploaded_file(self):
        self.authenticate()

        with patch(
            "routes.dataset_routes._persist_dataset",
            side_effect=SQLAlchemyError("test failure"),
        ):
            response = self.upload(b"feature,target\n1,yes\n2,no\n")

        self.assertEqual(response.status_code, 500)
        self.assertIn("could not be saved", response.json["error"].lower())
        self.assertEqual(Dataset.query.count(), 0)
        self.assertEqual(os.listdir(self.app.config["UPLOAD_FOLDER"]), [])

    def test_authenticated_preview_returns_headers_values_types_and_full_dimensions(self):
        self.authenticate()
        contents = "name,score,ratio\n" + "".join(
            f"student-{index},{index},{index + 0.5}\n"
            for index in range(13)
        )

        uploaded = self.upload(contents.encode("utf-8"))
        self.assertEqual(uploaded.status_code, 201)

        response = self.client.get(
            f"/api/dataset/{uploaded.json['dataset_id']}/preview"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["columns"], ["name", "score", "ratio"])
        self.assertEqual(
            response.json["rows"][0],
            {"name": "student-0", "score": 0, "ratio": 0.5},
        )
        self.assertIn(response.json["data_types"]["name"], {"object", "str"})
        self.assertEqual(response.json["data_types"]["score"], "int64")
        self.assertEqual(response.json["data_types"]["ratio"], "float64")
        self.assertEqual(response.json["row_count"], 13)
        self.assertEqual(response.json["column_count"], 3)
        self.assertEqual(response.json["sample_size"], 10)
        self.assertEqual(len(response.json["rows"]), 10)
        self.assertNotIn("file_path", response.json)

    def test_preview_serializes_missing_cells_as_null(self):
        self.authenticate()
        uploaded = self.upload(b"feature,target\n1,\n,yes\n3,no\n")
        self.assertEqual(uploaded.status_code, 201)

        response = self.client.get(
            f"/api/dataset/{uploaded.json['dataset_id']}/preview"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json["rows"][0],
            {"feature": 1, "target": None},
        )
        self.assertEqual(
            response.json["rows"][1],
            {"feature": None, "target": "yes"},
        )

    # US-13: Dataset statistics endpoint tests.
    def test_statistics_returns_numeric_summary_and_missing_counts(self):
        self.authenticate()
        contents = (
            b"hours,score,category\n"
            b"2,60,A\n"
            b"4,,B\n"
            b"6,90,A\n"
        )

        uploaded = self.upload(contents)
        self.assertEqual(uploaded.status_code, 201)

        response = self.client.get(
            f"/api/dataset/{uploaded.json['dataset_id']}/statistics"
        )

        self.assertEqual(response.status_code, 200)
        result = response.json

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["column_count"], 3)

        columns = {
            column["name"]: column
            for column in result["columns"]
        }

        self.assertEqual(columns["hours"]["type"], "numeric")
        self.assertEqual(columns["hours"]["missing_values"], 0)
        self.assertEqual(columns["hours"]["statistics"]["mean"], 4.0)
        self.assertEqual(columns["hours"]["statistics"]["median"], 4.0)
        self.assertEqual(columns["hours"]["statistics"]["min"], 2)
        self.assertEqual(columns["hours"]["statistics"]["max"], 6)

        self.assertEqual(columns["score"]["missing_values"], 1)
        self.assertEqual(columns["score"]["statistics"]["mean"], 75.0)

        self.assertEqual(columns["category"]["type"], "categorical")
        self.assertIsNone(columns["category"]["statistics"])

    def test_preview_requires_authentication(self):
        response = self.client.get("/api/dataset/1/preview")
        self.assertEqual(response.status_code, 401)

    def test_preview_hides_datasets_owned_by_another_user(self):
        self.authenticate()
        other_user = User(
            username=f"other_{uuid4().hex[:12]}",
            email=f"{uuid4().hex}@example.com",
            password_hash=generate_password_hash("other-password-123"),
        )
        db.session.add(other_user)
        db.session.flush()

        path = os.path.join(
            self.app.config["UPLOAD_FOLDER"],
            "other-user.csv",
        )
        os.makedirs(self.app.config["UPLOAD_FOLDER"], exist_ok=True)

        with open(path, "w", encoding="utf-8") as saved:
            saved.write("secret,value\nprivate,1\n")

        dataset = Dataset(
            user_id=other_user.id,
            filename="secret.csv",
            file_path=path,
            file_size=24,
            row_count=1,
            column_count=2,
        )
        db.session.add(dataset)
        db.session.commit()

        response = self.client.get(
            f"/api/dataset/{dataset.id}/preview"
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["error"], "Dataset not found")
        self.assertNotIn("secret", response.get_data(as_text=True))

    def test_preview_nonexistent_dataset_returns_404(self):
        self.authenticate()
        response = self.client.get("/api/dataset/999999/preview")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json["error"], "Dataset not found")

    def test_preview_missing_file_returns_clear_404_without_path(self):
        self.authenticate()
        uploaded = self.upload(b"feature,target\n1,yes\n2,no\n")
        dataset = db.session.get(Dataset, uploaded.json["dataset_id"])
        os.remove(dataset.file_path)

        response = self.client.get(
            f"/api/dataset/{dataset.id}/preview"
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("file is missing", response.json["error"].lower())
        self.assertNotIn(dataset.file_path, response.get_data(as_text=True))

    def test_preview_unreadable_file_returns_clear_500(self):
        self.authenticate()
        uploaded = self.upload(b"feature,target\n1,yes\n2,no\n")

        with patch(
            "routes.dataset_routes.pd.read_csv",
            side_effect=PermissionError("private path"),
        ):
            response = self.client.get(
                f"/api/dataset/{uploaded.json['dataset_id']}/preview"
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json["error"],
            "The dataset file could not be read.",
        )
        self.assertNotIn("private path", response.get_data(as_text=True))

    def test_preview_malformed_csv_returns_clear_400(self):
        self.authenticate()
        uploaded = self.upload(b"feature,target\n1,yes\n2,no\n")
        dataset = db.session.get(Dataset, uploaded.json["dataset_id"])

        with open(dataset.file_path, "wb") as saved:
            saved.write(b'feature,target\n"unterminated,yes\n')

        response = self.client.get(
            f"/api/dataset/{dataset.id}/preview"
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("malformed", response.json["error"].lower())


    def test_us11_replacement_updates_owned_dataset_without_changing_its_id(self):
        self.authenticate()
        initial = self.upload(b"feature,target\n1,yes\n2,no\n", "initial.csv")
        self.assertEqual(initial.status_code, 201)
        dataset_id = initial.json["dataset_id"]
        original = db.session.get(Dataset, dataset_id)
        old_path = original.file_path
        replacement = self.client.post(
            f"/api/dataset/{dataset_id}/replace",
            data={"file": (io.BytesIO(b"height,label\n10,A\n20,B\n30,A\n"), "replacement.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(replacement.status_code, 200, replacement.get_data(as_text=True))
        current = db.session.get(Dataset, dataset_id)
        self.assertEqual(current.id, dataset_id)
        self.assertEqual(current.filename, "replacement.csv")
        self.assertEqual(current.row_count, 3)
        self.assertEqual(current.metadata_json["columns"], ["height", "label"])
        self.assertNotEqual(current.file_path, old_path)
        self.assertFalse(os.path.exists(old_path))

    def test_us11_replacement_of_another_users_dataset_is_hidden(self):
        self.authenticate()
        uploaded = self.upload(b"feature,target\n1,yes\n2,no\n")
        other = self.app.test_client()
        token = uuid4().hex
        username = f"other_{token[:12]}"
        other.post("/api/auth/register", json={"username": username, "email": f"{token}@example.com", "password": "StrongPass123!"})
        other.post("/api/auth/login", json={"username": username, "password": "StrongPass123!"})
        response = other.post(f"/api/dataset/{uploaded.json['dataset_id']}/replace", data={
            "file": (io.BytesIO(b"feature,target\n3,yes\n4,no\n"), "stolen.csv")
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
