from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import current_app, g, jsonify, session

from database.db import db
from models.dataset import Dataset
from models.evaluation import EvaluationResult
from models.model import MLModel
from models.user import User
from models.user_session import UserSession


def _utcnow():
    """Return a naive UTC timestamp compatible with the SQLite DateTime columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _session_lifetime():
    lifetime = current_app.config.get(
        "PERMANENT_SESSION_LIFETIME", timedelta(days=31)
    )
    if not isinstance(lifetime, timedelta):
        lifetime = timedelta(seconds=float(lifetime))
    return lifetime


def get_authenticated_user():
    """Validate and refresh the server-side session for this request."""
    user_id = session.get("user_id")
    session_id = session.get("session_id")
    if user_id is None or not isinstance(session_id, str):
        if user_id is not None or session_id is not None:
            session.clear()
        return None

    session_record = db.session.get(UserSession, session_id)
    if (
        session_record is None
        or session_record.user_id != user_id
        or session_record.revoked_at is not None
    ):
        session.clear()
        return None

    now = _utcnow()
    if now - session_record.last_seen_at >= _session_lifetime():
        session_record.revoked_at = now
        db.session.commit()
        session.clear()
        return None

    user = db.session.get(User, user_id)
    if user is None:
        session_record.revoked_at = now
        db.session.commit()
        session.clear()
        return None

    # This timestamp is also the idle-timeout clock. The signed cookie alone is
    # not authoritative; every protected request checks this revocable record.
    session_record.last_seen_at = now
    db.session.commit()
    g.current_user = user
    g.current_user_session = session_record
    return user


def revoke_current_session():
    """Revoke the current session record and clear its browser cookie state."""
    user_id = session.get("user_id")
    session_id = session.get("session_id")
    active = False

    if user_id is not None and isinstance(session_id, str):
        session_record = db.session.get(UserSession, session_id)
        if session_record is not None and session_record.user_id == user_id:
            now = _utcnow()
            active = (
                session_record.revoked_at is None
                and now - session_record.last_seen_at < _session_lifetime()
                and db.session.get(User, user_id) is not None
            )
            if session_record.revoked_at is None:
                session_record.revoked_at = now
                db.session.commit()

    session.clear()
    return active


def owned_dataset_query(user_id):
    return Dataset.query.filter(Dataset.user_id == user_id)


def get_owned_dataset(dataset_id, user_id):
    return owned_dataset_query(user_id).filter(Dataset.id == dataset_id).first()


def owned_model_query(user_id):
    return (
        MLModel.query.join(Dataset, MLModel.dataset_id == Dataset.id)
        .filter(MLModel.user_id == user_id, Dataset.user_id == user_id)
    )


def get_owned_model(model_id, user_id):
    return owned_model_query(user_id).filter(MLModel.id == model_id).first()


def owned_evaluation_query(user_id):
    return (
        EvaluationResult.query
        .join(MLModel, EvaluationResult.model_id == MLModel.id)
        .join(Dataset, MLModel.dataset_id == Dataset.id)
        .filter(
            EvaluationResult.user_id == user_id,
            MLModel.user_id == user_id,
            Dataset.user_id == user_id,
        )
    )


def get_owned_evaluation(evaluation_id, user_id):
    return owned_evaluation_query(user_id).filter(
        EvaluationResult.id == evaluation_id
    ).first()


def login_required(view):
    """Reject API requests without a current, unexpired server-side session."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = get_authenticated_user()
        if user is None:
            return jsonify({"error": "Authentication required"}), 401
        if not getattr(user, "is_active", True):
            return jsonify({"error": "Account is deactivated. Contact an administrator."}), 403
        return view(*args, **kwargs)

    return wrapped


def role_required(*allowed_roles):
    """Reject requests if user is unauthenticated, inactive, or lacks required role."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = get_authenticated_user()
            if user is None:
                return jsonify({"error": "Authentication required"}), 401
            if not getattr(user, "is_active", True):
                return jsonify({"error": "Account is deactivated. Contact an administrator."}), 403
            user_role = getattr(user, "role", "student")
            if user_role not in allowed_roles:
                return jsonify({
                    "error": "Access denied: insufficient permissions",
                    "required_roles": list(allowed_roles),
                    "current_role": user_role
                }), 403
            return view(*args, **kwargs)
        return wrapped
    return decorator


def admin_required(view):
    """Convenience decorator restricting access strictly to Admin role."""
    return role_required("admin")(view)


def log_activity(action, user_id=None, details=None, ip_address=None):
    """Record an audit event to ActivityLog without exposing sensitive data."""
    try:
        from models.activity_log import ActivityLog
        details_str = details
        if isinstance(details, dict):
            import json
            sanitized = {
                k: v for k, v in details.items()
                if k not in ("password", "password_hash", "token", "session_id")
            }
            details_str = json.dumps(sanitized)
        elif isinstance(details_str, str):
            for sensitive_term in ("password=", "token=", "session_id="):
                if sensitive_term in details_str:
                    details_str = details_str.split(sensitive_term)[0] + f"{sensitive_term}[REDACTED]"

        log_entry = ActivityLog(
            user_id=user_id,
            action=action,
            details=details_str,
            ip_address=ip_address
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception:
        db.session.rollback()


def non_instructor_required(view):
    """Allow authenticated students/admins while enforcing instructor read-only access."""
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if getattr(g.current_user, "role", "student") == "instructor":
            return jsonify({"error": "Instructor accounts have read-only access."}), 403
        return view(*args, **kwargs)
    return wrapped
