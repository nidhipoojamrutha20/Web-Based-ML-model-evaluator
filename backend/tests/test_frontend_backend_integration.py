import unittest
from pathlib import Path

from app import create_app
from database.db import db


class TestFrontendBackendIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class TestConfig:
            TESTING = True
            SECRET_KEY = "test-only"
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            UPLOAD_FOLDER = "test_uploads"
            SAVED_MODELS_FOLDER = "test_saved_models"

        cls.app = create_app(TestConfig)
        cls.context = cls.app.app_context()
        cls.context.push()

    @classmethod
    def tearDownClass(cls):
        db.session.remove()
        db.drop_all()
        db.engine.dispose()
        cls.context.pop()

    def setUp(self):
        self.client = self.app.test_client()

    def test_frontend_page_loads(self):
        response = self.client.get("/")
        try:
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers["Location"], "/login")
        finally:
            response.close()

    def test_login_page_loads(self):
        response = self.client.get("/login")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"login-form", response.data)
        finally:
            response.close()

    def test_dashboard_file_is_protected(self):
        response = self.client.get("/index.html")
        try:
            self.assertEqual(response.status_code, 302)
            self.assertEqual(response.headers["Location"], "/login")
        finally:
            response.close()

    def test_backend_health_endpoint(self):
        response = self.client.get("/api/health")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["status"], "healthy")
        finally:
            response.close()

    def test_frontend_javascript_is_served(self):
        response = self.client.get("/js/app.js")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"API_BASE", response.data)
        finally:
            response.close()

    def test_us08_csv_upload_form_and_confirmation_are_wired(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        script_path = frontend_dir / "js" / "app.js"
        script = script_path.read_text(encoding="utf-8")

        self.assertIn('id="upload-form"', page)
        self.assertIn('type="file"', page)
        self.assertIn('accept=".csv"', page)
        self.assertIn('id="upload-feedback"', page)
        self.assertIn('"upload-form"', script)
        self.assertIn('addEventListener("submit", uploadDataset)', script)
        self.assertIn('body: new FormData(form)', script)
        self.assertIn("uploaded successfully", script)
        self.assertIn("result.row_count", script)
        self.assertIn("result.column_count", script)

    def test_us16_split_slider_updates_and_is_sent_to_evaluation(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        script = (frontend_dir / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="split-range"', page)
        self.assertIn('value="0.2"', page)
        self.assertIn('id="split-display"', page)
        self.assertIn("updateSplitDisplay", script)
        self.assertIn('addEventListener("input", updateSplitDisplay)', script)
        self.assertIn('apiRequest("/ml/evaluate"', script)
        self.assertIn('test_size: Number(document.getElementById("split-range").value)', script)

    def test_us12_preview_is_loaded_after_upload_and_rendered_safely(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        script = (frontend_dir / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="preview-status"', page)
        self.assertIn('id="preview-sample-label"', page)
        self.assertIn('id="preview-table"', page)
        self.assertIn("<thead", page)
        self.assertIn("<tbody>", page)
        self.assertIn("await loadDatasetPreview(result.dataset_id)", script)
        self.assertIn("/preview", script)
        self.assertIn("preview.data_types[column]", script)
        self.assertIn("preview.row_count", script)
        self.assertIn("preview.column_count", script)
        self.assertIn("for (const column of preview.columns)", script)
        self.assertIn("for (const row of preview.rows)", script)
        self.assertIn("preview.sample_size", script)
        self.assertIn("cell.textContent", script)
        self.assertNotIn("innerHTML = row", script)

    def test_auth_and_upload_frontend_flows_remain_wired(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        login_page = (frontend_dir / "login.html").read_text(encoding="utf-8")
        script = (frontend_dir / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="registration-form"', login_page)
        self.assertIn('id="login-form"', login_page)
        self.assertIn('id="upload-form"', page)
        self.assertIn('addEventListener("submit", uploadDataset)', script)
        self.assertIn('addEventListener("submit", async (event) =>', script)
        self.assertIn('apiRequest("/auth/register"', script)
        self.assertIn('apiRequest("/auth/login"', script)
        self.assertIn('apiRequest("/auth/session")', script)
        self.assertIn('const endpoint = isReplacement', script)
        self.assertIn('"/dataset/upload"', script)
        self.assertIn('apiRequest(endpoint', script)

    def test_us14a_admin_panel_elements_and_wiring(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        script = (frontend_dir / "js" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="user-role-badge"', page)
        self.assertIn('id="admin-panel-btn"', page)
        self.assertIn('id="adminModal"', page)
        self.assertIn('id="admin-users-pane"', page)
        self.assertIn('id="admin-datasets-pane"', page)
        self.assertIn('id="admin-logs-pane"', page)

        self.assertIn('setupAdminPanel', script)
        self.assertIn('/admin/users', script)
        self.assertIn('/admin/datasets', script)
        self.assertIn('/admin/logs', script)
        self.assertIn('loadAdminUsers', script)
        self.assertIn('loadAdminDatasets', script)
        self.assertIn('loadAdminLogs', script)


    def test_us17_to_us20_frontend_catalog_history_comparison_and_review_are_wired(self):
        frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
        script = (frontend_dir / "js" / "app.js").read_text(encoding="utf-8")
        page = (frontend_dir / "index.html").read_text(encoding="utf-8")
        self.assertIn('apiRequest("/ml/algorithms")', script)
        self.assertIn('"/ml/compare"', script)
        self.assertIn('"/ml/instructor/evaluations"', script)
        self.assertIn('"/ml/instructor/compare"', script)
        self.assertIn("loadInstructorEvaluation", script)
        self.assertIn('classList.toggle("d-none", readOnly)', script)
        self.assertIn('id="dataset-preview-card"', page)
        self.assertIn('id="statistics-card"', page)
        self.assertIn('id="instructor-review-details"', page)
        self.assertIn("result.confusion_matrix", script)
        self.assertIn('id="run-history-body"', page)
        self.assertIn('id="instructor-mode-banner"', page)


if __name__ == "__main__":
    unittest.main()
