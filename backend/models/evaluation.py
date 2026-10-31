
from database.db import db

from sqlalchemy import DDL, event



class EvaluationResult(db.Model):
    __tablename__ = "evaluation_results"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    model_id = db.Column(
        db.Integer,
        db.ForeignKey("models.id"),
        nullable=False
    )

    metrics_json = db.Column(db.JSON, nullable=False)
    confusion_matrix_json = db.Column(db.JSON, nullable=True)
    report_json = db.Column(db.JSON, nullable=True)
    shared_with_instructors = db.Column(db.Boolean, nullable=False, default=False, server_default=db.text("0"))

    created_at = db.Column(
        db.DateTime,
        nullable=False,
        server_default=db.func.current_timestamp()
    )

    user = db.relationship("User", back_populates="evaluations")
    model = db.relationship("MLModel", back_populates="evaluations")


    # SQLite does not allow subqueries in CHECK constraints. These triggers
    # enforce the cross-table ownership rule whenever an evaluation is written.


_owner_guard = """
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1
            FROM models
            WHERE models.id = NEW.model_id
              AND models.user_id = NEW.user_id
        )
        THEN RAISE(ABORT, 'evaluation_owner_matches_model_owner')
    END;
"""

event.listen(
    db.metadata,
    "after_create",
    DDL(
        f"""
        CREATE TRIGGER IF NOT EXISTS evaluation_owner_matches_model_owner_insert
        BEFORE INSERT ON evaluation_results
        BEGIN
            {_owner_guard}
        END;
        """
    ).execute_if(dialect="sqlite"),
)

event.listen(
    db.metadata,
    "after_create",
    DDL(
        """
        CREATE TRIGGER IF NOT EXISTS evaluation_owner_matches_model_owner_model_update
        BEFORE UPDATE OF user_id ON models
        WHEN EXISTS (
            SELECT 1
            FROM evaluation_results
            WHERE evaluation_results.model_id = OLD.id
              AND evaluation_results.user_id != NEW.user_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'evaluation_owner_matches_model_owner');
        END;
        """
    ).execute_if(dialect="sqlite"),
)

event.listen(
    db.metadata,
    "after_create",
    DDL(
        f"""
        CREATE TRIGGER IF NOT EXISTS evaluation_owner_matches_model_owner_update
        BEFORE UPDATE OF user_id, model_id ON evaluation_results
        BEGIN
            {_owner_guard}
        END;
        """
    ).execute_if(dialect="sqlite"),
)
