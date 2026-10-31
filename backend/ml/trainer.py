from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import joblib

class ModelTrainer:
    """Trains classification and regression models using scikit-learn."""

    SUPPORTED_ALGORITHMS = [
        "logistic_regression",
        "linear_regression",
        "decision_tree",
        "random_forest",
        "knn"
    ]

    @classmethod
    def train(cls, algorithm_name: str, task_type: str, X_train, y_train, **kwargs):
        """Fit a persisted-ready preprocessing and estimator pipeline."""
        factories = {
            ("classification", "logistic_regression"): LogisticRegression,
            ("regression", "linear_regression"): LinearRegression,
            ("classification", "decision_tree"): DecisionTreeClassifier,
            ("regression", "decision_tree"): DecisionTreeRegressor,
            ("classification", "random_forest"): RandomForestClassifier,
            ("regression", "random_forest"): RandomForestRegressor,
            ("classification", "knn"): KNeighborsClassifier,
            ("regression", "knn"): KNeighborsRegressor,
        }
        factory = factories.get((task_type, algorithm_name))
        if factory is None:
            raise ValueError("The selected algorithm is not supported for this task.")
        numeric = list(X_train.select_dtypes(include="number").columns)
        categorical = [column for column in X_train.columns if column not in numeric]
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical))
        pipeline = Pipeline([
            ("preprocessing", ColumnTransformer(transformers)),
            ("model", factory()),
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    @staticmethod
    def save(model, filepath: str):
        """Serialize model weights with joblib."""
        joblib.dump(model, filepath)
