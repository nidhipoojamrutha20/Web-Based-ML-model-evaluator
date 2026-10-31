# Web-Based ML Model Evaluator

A lightweight web application for uploading tabular datasets (CSV), configuring supervised machine learning models, training them on the fly, and evaluating performance metrics through interactive visualizations.

---

## 🏛️ Architecture Overview

The repository is modularly structured into dedicated frontend, backend, machine learning, and database components:

```
Web-Based-ML-model-evaluator/
├── backend/                        # Python Flask application
│   ├── app.py                     # Main application factory & health check
│   ├── config.py                  # Environment & database configuration
│   ├── requirements.txt           # Python dependencies
│   ├── database/                  # SQLAlchemy ORM database initialization
│   │   ├── __init__.py
│   │   └── db.py
│   ├── models/                    # Relational data models
│   │   ├── __init__.py
│   │   ├── user.py                # User accounts & authentication
│   │   ├── dataset.py             # Dataset metadata & column tracking
│   │   └── evaluation.py          # Trained model runs & evaluation metrics
│   ├── ml/                        # Machine Learning core module
│   │   ├── __init__.py
│   │   ├── preprocessor.py        # Missing values, encoding, scaling & split
│   │   ├── trainer.py             # Scikit-learn algorithm training
│   │   └── evaluator.py           # Metrics calculation & confusion matrix
│   ├── routes/                    # REST API Blueprints
│   │   ├── __init__.py
│   │   ├── auth_routes.py         # /api/auth (register, login, logout)
│   │   ├── dataset_routes.py      # /api/dataset (upload, preview, metadata)
│   │   └── ml_routes.py           # /api/ml (algorithms, train, evaluations)
│   ├── uploads/                   # Stored uploaded CSV files
│   └── saved_models/              # Serialized trained model files (.joblib)
├── frontend/                      # Web user interface
│   ├── index.html                 # Single-page web dashboard
│   ├── css/
│   │   └── style.css              # Custom responsive styles
│   └── js/
│       └── app.js                 # REST API client & dynamic UI renderer
├── documents/                     # Project SRS, UML diagrams, test plans
├── .gitignore                     # Git ignore rules for Python, virtualenv, and data
└── README.md
```

---

## �️ Database Schema

The application uses SQLAlchemy ORM with SQLite (configurable to PostgreSQL/MySQL) and consists of four relational tables:

### Entity-Relationship Diagram

```
┌─────────────────────┐
│       users         │
├─────────────────────┤
│ id (PK)            │◄────┐
│ username (unique)   │     │
│ email (unique)      │     │
│ password_hash       │     │
│ created_at          │     │
└─────────────────────┘     │
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         │                  │                  │
┌────────▼────────┐  ┌──────▼──────────┐  ┌───▼─────────────────┐
│    datasets     │  │     models      │  │ evaluation_results  │
├─────────────────┤  ├─────────────────┤  ├─────────────────────┤
│ id (PK)        │◄─┤ id (PK)        │◄─┤ id (PK)            │
│ user_id (FK)   │  │ user_id (FK)   │  │ user_id (FK) ⚠️    │
│ filename        │  │ dataset_id (FK)│  │ model_id (FK)      │
│ file_path       │  │ model_name      │  │ metrics_json       │
│ file_size       │  │ algorithm       │  │ confusion_matrix   │
│ row_count       │  │ model_path      │  │ report_json        │
│ column_count    │  │ status          │  │ created_at         │
│ metadata_json   │  │ config_json     │  └────────────────────┘
│ created_at      │  │ created_at      │
└─────────────────┘  └─────────────────┘
```

### Table Descriptions

#### **1. users**
Stores user account information with authentication credentials.

| Column         | Type          | Constraints           | Description                    |
|----------------|---------------|-----------------------|--------------------------------|
| id             | Integer       | Primary Key           | Unique user identifier         |
| username       | String(80)    | Unique, Not Null      | User's display name            |
| email          | String(120)   | Unique, Not Null      | User's email address           |
| password_hash  | String(256)   | Not Null              | Bcrypt-hashed password         |
| created_at     | DateTime      | Not Null, Default NOW | Account creation timestamp     |

**Relationships**: One-to-many with `datasets`, `models`, and `evaluation_results`

---

#### **2. datasets**
Stores uploaded CSV dataset metadata and file references.

| Column         | Type          | Constraints           | Description                    |
|----------------|---------------|-----------------------|--------------------------------|
| id             | Integer       | Primary Key           | Unique dataset identifier      |
| user_id        | Integer       | Foreign Key, Not Null | Owner reference → users.id     |
| filename       | String(255)   | Not Null              | Original uploaded filename     |
| file_path      | String(512)   | Not Null              | Server storage path            |
| file_size      | BigInteger    | Nullable              | File size in bytes             |
| row_count      | Integer       | Nullable              | Number of data rows            |
| column_count   | Integer       | Nullable              | Number of columns              |
| metadata_json  | JSON          | Nullable              | Column names, types, stats     |
| created_at     | DateTime      | Not Null, Default NOW | Upload timestamp               |

**Relationships**: Belongs to `User`, has many `MLModel`

---

#### **3. models**
Stores trained machine learning model configurations and serialized file paths.

| Column         | Type          | Constraints           | Description                    |
|----------------|---------------|-----------------------|--------------------------------|
| id             | Integer       | Primary Key           | Unique model identifier        |
| user_id        | Integer       | Foreign Key, Not Null | Owner reference → users.id     |
| dataset_id     | Integer       | Foreign Key, Not Null | Source dataset → datasets.id   |
| model_name     | String(128)   | Not Null              | User-defined model name        |
| algorithm      | String(128)   | Not Null              | Algorithm (e.g., random_forest)|
| model_path     | String(512)   | Nullable              | Path to .joblib serialized file|
| status         | String(32)    | Not Null, Default: pending | Training status           |
| config_json    | JSON          | Nullable              | Hyperparameters, split ratio   |
| created_at     | DateTime      | Not Null, Default NOW | Model creation timestamp       |

**Relationships**: Belongs to `User` and `Dataset`, has many `EvaluationResult`

---

#### **4. evaluation_results**
Stores model performance metrics, confusion matrices, and classification reports.

| Column                | Type          | Constraints           | Description                    |
|-----------------------|---------------|-----------------------|--------------------------------|
| id                    | Integer       | Primary Key           | Unique evaluation identifier   |
| user_id               | Integer       | Foreign Key, Not Null | Owner reference → users.id ⚠️ |
| model_id              | Integer       | Foreign Key, Not Null | Evaluated model → models.id    |
| metrics_json          | JSON          | Not Null              | Accuracy, F1, MSE, RMSE, R²    |
| confusion_matrix_json | JSON          | Nullable              | Confusion matrix array         |
| report_json           | JSON          | Nullable              | Per-class precision/recall     |
| created_at            | DateTime      | Not Null, Default NOW | Evaluation timestamp           |

**Relationships**: Belongs to `User` and `MLModel`

---

### 🔐 US-06 Session Handling

Successful logins create a random server-side session record and place its opaque identifier in the signed Flask session cookie. Protected requests validate the record, user, revocation state, and 24-hour idle timeout. Logout revokes the server-side record and clears the browser cookie; replaying an earlier cookie does not restore access. Cookies are HttpOnly and SameSite=Lax. Set `SESSION_COOKIE_SECURE=true` when serving the application over HTTPS.

### 🔐 US-07 User Data Isolation

All user-specific API access is scoped to the authenticated user on the server. Dataset listing and metadata/detail requests filter by `datasets.user_id`. Model access also checks the associated dataset owner, and evaluation access checks the evaluation owner, model owner, and dataset owner together. A foreign or missing identifier returns the same not-found response. API responses omit server filesystem paths.

CSV upload (US-08 to US-10) is implemented and assigns dataset ownership from the authenticated session; a client-supplied owner ID is never trusted. Model training/evaluation creation (US-18/US-19) remains planned and must use the same owner-scoped lookups. Artifact download routes must apply the model ownership check before serving a file.

### 🔒 Schema Integrity: Owner Consistency

**Design Decision**:
- The `evaluation_results` table stores both `user_id` and `model_id` as foreign keys.
- While `user_id` could technically be derived from `models.user_id`, storing it directly enables:
  - Faster queries when filtering evaluations by user without joining through models
  - Clearer audit trails showing who created each evaluation
  - Support for potential future features (e.g., shared model evaluations)

**Integrity Protection**:
Evaluation reads require `evaluation_results.user_id`, `models.user_id`, and the associated `datasets.user_id` to match the authenticated user. This prevents inconsistent cross-owner rows from being returned. SQLite does not permit the cross-table subquery required for a `CHECK` constraint.

**Implementation Status**:
- ✅ Read/list endpoints apply owner filters for datasets, models, and evaluation results.
- ⚠️ Model training and evaluation creation routes (US-18/US-19) remain TODO. They must assign the current session's owner and validate the selected dataset/model with `get_owned_dataset` or `get_owned_model` before writing records.

**Required write-path check** (when US-18/US-19 are implemented):
```python
model = get_owned_model(model_id, g.current_user.id)
if model is None:
    return jsonify({"error": "Model not found"}), 404

evaluation = EvaluationResult(
    user_id=g.current_user.id,
    model_id=model.id,
    metrics_json=metrics,
)
```

**Note**: SQLite does not support subqueries in CHECK constraints. For production deployments using PostgreSQL or MySQL, a database-level CHECK constraint or trigger can provide additional enforcement.

---

## �🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9+ (Python 3.13 supported)
- Modern Web Browser (Chrome, Firefox, Edge)

### 2. Setup & Installation

Navigate to the `backend/` directory:
```bash
cd backend
```

Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Application

Launch the Flask backend server:
```bash
python app.py
```

The application will start on:
👉 **http://127.0.0.1:5000**

Opening **http://127.0.0.1:5000** in your browser will automatically serve the full web interface!

---

## 📡 Core REST APIs

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Service health check |
| `/api/auth/register` | `POST` | Register user account |
| `/api/auth/login` | `POST` | User login session |
| `/api/auth/logout` | `POST` | Logout and invalidate session |
| `/api/auth/session` | `GET` | Check current session status |
| `/api/dataset` | `GET` | List the authenticated user's datasets |
| `/api/dataset/<id>` | `GET` | Read owner-scoped dataset metadata |
| `/api/dataset/upload` | `POST` | Validate, store, and record authenticated CSV uploads (US-08 to US-10) |
| `/api/dataset/<id>/preview` | `GET` | Preview (planned: US-12; ownership check is active) |
| `/api/dataset/<id>/statistics` | `GET` | Statistics (planned: US-13; ownership check is active) |
| `/api/ml/algorithms` | `GET` | List supported ML algorithms |
| `/api/ml/models` | `GET` | List the authenticated user's model runs |
| `/api/ml/models/<id>` | `GET` | Read an owner-scoped model run |
| `/api/ml/evaluations` | `GET` | List the authenticated user's evaluation results |
| `/api/ml/evaluations/<id>` | `GET` | Read an owner-scoped evaluation result |
| `/api/ml/train` | `POST` | Train, persist an owner-scoped model pipeline and held-out evaluation |
| `/api/ml/evaluate` | `POST` | Retrieve a run's persisted evaluation |
| `/api/ml/compare` | `POST` | Compare compatible saved runs using persisted metrics |
| `/api/ml/models/<id>/download` | `GET` | Download the owner-scoped joblib pipeline |
| `/api/ml/evaluations/<id>/report` | `GET` | Generate and download an owner-scoped PDF report |

### Model comparison and downloads (US-20)

Training saves scikit-learn preprocessing and estimator pipelines beneath `backend/saved_models/<user-id>/` with restrictive directory/file permissions. The database stores run configuration, dataset fingerprint, feature metadata, artifact checksum, and evaluation metrics. `POST /api/ml/compare` accepts `{ "model_ids": [1, 2] }` and requires the runs to share the dataset contents, target, task, split, random seed, and preprocessing configuration. Its model rows, chart labels, and metric points follow the requested `model_ids` order; chart labels include run IDs so repeated algorithms are distinguishable. It returns persisted classification (`accuracy`, `precision`, `recall`, `f1`) or regression (`mae`, `mse`, `rmse`, `r2`) metrics and chart data. Model downloads are joblib files and are served without loading/deserializing them. Evaluation reports are generated as PDFs with ReportLab, declared in `backend/requirements.txt`.

### CSV dataset upload and validation (US-08 to US-10)

`POST /api/dataset/upload` accepts a multipart form field named `file` and requires a verified login session. CSV input must be UTF-8 (with an optional BOM), contain at least two non-empty data rows and two columns (a feature and a target), have unique non-empty headers, consistent row widths, and at least two columns containing data; entirely empty data rows and binary NUL characters are rejected. Unsupported extensions, empty or malformed files, and invalid structures return clear JSON errors. The configured 16 MiB request-body limit is enforced with HTTP 413; it includes multipart form overhead.

Accepted files are stored under `backend/uploads/` with server-generated names. The database records the original filename, stored path, file size, row and column counts, column names, upload timestamp, and the authenticated user's ID. Upload responses include the dataset ID and summary metadata. The owner-scoped `GET /api/dataset` and `GET /api/dataset/<id>` endpoints retrieve the current user's metadata. Failed database writes are rolled back and the newly created file is removed. Tests are in `backend/tests/test_dataset_upload.py`.

#### macOS setup and tests

Run these commands from the repository root to install the backend dependencies, run all backend tests, and start the app:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python app.py
```

To run only the isolated CSV upload tests after setup:

```bash
cd backend
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_dataset_upload.py' -v
```

---

## 🤖 Supported ML Algorithms

- **Classification**:
  - Logistic Regression
  - Random Forest Classifier
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
- **Regression**:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
  - KNN Regressor
