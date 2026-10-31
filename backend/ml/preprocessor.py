# Machine Learning Preprocessing Module
#
# US-14: Implement data preprocessing (Assigned to: @aryannang123)
# TODO: US-15: Implement target column selection (Assigned to: @abhishekdeene-eng)
# TODO: US-16: Implement train-test split configuration (Assigned to: @nidhipoojamrutha20)

import math
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

class DataPreprocessor:
    """Preprocesses tabular dataset before training."""

    SUPPORTED_MISSING_STRATEGIES = ["none", "drop", "mean", "median", "mode", "constant"]
    SUPPORTED_ENCODING_STRATEGIES = ["none", "label", "onehot"]
    SUPPORTED_SCALING_STRATEGIES = ["none", "minmax", "standard"]

    @staticmethod
    def inspect(df: pd.DataFrame) -> dict:
        """Inspect columns, null counts, and data types (US-09 & US-13)."""
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a valid pandas DataFrame.")

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        null_counts = {str(c): int(df[c].isna().sum()) for c in df.columns}
        dtypes = {str(c): str(df[c].dtype) for c in df.columns}

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "numeric_columns": [str(c) for c in numeric_cols],
            "categorical_columns": [str(c) for c in categorical_cols],
            "null_counts": null_counts,
            "has_missing_values": any(v > 0 for v in null_counts.values()),
            "dtypes": dtypes,
        }

    @classmethod
    def handle_missing_values(
        cls,
        df: pd.DataFrame,
        strategy: str = "none",
        fill_value=None,
        target_column: str = None
    ) -> pd.DataFrame:
        """Impute or drop missing data (US-14)."""
        if strategy not in cls.SUPPORTED_MISSING_STRATEGIES:
            raise ValueError(
                f"Unsupported missing value strategy '{strategy}'. "
                f"Supported options: {', '.join(cls.SUPPORTED_MISSING_STRATEGIES)}"
            )

        if strategy == "none":
            return df.copy()

        df_cleaned = df.copy()

        if strategy == "drop":
            df_cleaned = df_cleaned.dropna()
            if len(df_cleaned) < 2:
                raise ValueError("Dropping missing values left fewer than 2 rows in the dataset.")
            return df_cleaned

        for col in df_cleaned.columns:
            if df_cleaned[col].isna().sum() == 0:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])

            if strategy == "mean":
                if is_numeric:
                    mean_val = df_cleaned[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                else:
                    mode_series = df_cleaned[col].mode(dropna=True)
                    mode_val = mode_series.iloc[0] if not mode_series.empty else "missing"
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)

            elif strategy == "median":
                if is_numeric:
                    median_val = df_cleaned[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                else:
                    mode_series = df_cleaned[col].mode(dropna=True)
                    mode_val = mode_series.iloc[0] if not mode_series.empty else "missing"
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)

            elif strategy == "mode":
                mode_series = df_cleaned[col].mode(dropna=True)
                if not mode_series.empty:
                    fill_val = mode_series.iloc[0]
                else:
                    fill_val = 0 if is_numeric else "missing"
                df_cleaned[col] = df_cleaned[col].fillna(fill_val)

            elif strategy == "constant":
                if fill_value is not None:
                    val = fill_value
                    if is_numeric:
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            val = 0
                    df_cleaned[col] = df_cleaned[col].fillna(val)
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(0 if is_numeric else "missing")

        return df_cleaned

    @classmethod
    def encode_features(
        cls,
        df: pd.DataFrame,
        strategy: str = "none",
        target_column: str = None
    ) -> pd.DataFrame:
        """Encode categorical features (US-14)."""
        if strategy not in cls.SUPPORTED_ENCODING_STRATEGIES:
            raise ValueError(
                f"Unsupported encoding strategy '{strategy}'. "
                f"Supported options: {', '.join(cls.SUPPORTED_ENCODING_STRATEGIES)}"
            )

        if strategy == "none":
            return df.copy()

        df_encoded = df.copy()

        # Identify categorical columns to encode
        cat_cols = [
            c for c in df_encoded.columns
            if not pd.api.types.is_numeric_dtype(df_encoded[c])
        ]

        if not cat_cols:
            return df_encoded

        # Check for unhandled missing values in categorical columns before encoding
        for col in cat_cols:
            if df_encoded[col].isna().any():
                df_encoded[col] = df_encoded[col].fillna("missing")

        if strategy == "label":
            for col in cat_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        elif strategy == "onehot":
            feature_cat_cols = [c for c in cat_cols if c != target_column]

            if feature_cat_cols:
                df_encoded = pd.get_dummies(
                    df_encoded,
                    columns=feature_cat_cols,
                    drop_first=False,
                    dtype=int
                )

            # If target column is categorical, encode it with LabelEncoder to keep it as a single target column
            if target_column and target_column in df_encoded.columns and not pd.api.types.is_numeric_dtype(df_encoded[target_column]):
                le = LabelEncoder()
                df_encoded[target_column] = le.fit_transform(df_encoded[target_column].astype(str))

        return df_encoded

    @classmethod
    def scale_features(
        cls,
        df: pd.DataFrame,
        strategy: str = "none",
        target_column: str = None
    ) -> pd.DataFrame:
        """Apply feature scaling (MinMax or Standard) to numeric feature columns (US-14)."""
        if strategy not in cls.SUPPORTED_SCALING_STRATEGIES:
            raise ValueError(
                f"Unsupported scaling strategy '{strategy}'. "
                f"Supported options: {', '.join(cls.SUPPORTED_SCALING_STRATEGIES)}"
            )

        if strategy == "none":
            return df.copy()

        df_scaled = df.copy()

        feature_cols = [c for c in df_scaled.columns if c != target_column]
        if not feature_cols:
            raise ValueError("No feature columns available to scale.")

        # Reject incompatible configurations: scaling attempted on non-numeric features
        non_numeric_features = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(df_scaled[c])
        ]
        if non_numeric_features:
            raise ValueError(
                f"Cannot apply {strategy} scaling to non-numeric column(s): "
                f"{', '.join(non_numeric_features)}. Please encode categorical features first."
            )

        # Ensure no infinite or remaining null values
        for col in feature_cols:
            if df_scaled[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains missing values. Please handle missing values before scaling."
                )
            if np.isinf(df_scaled[col]).any():
                raise ValueError(
                    f"Column '{col}' contains infinite values which cannot be scaled."
                )

        if strategy == "minmax":
            scaler = MinMaxScaler()
            df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

        elif strategy == "standard":
            scaler = StandardScaler()
            df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

        return df_scaled

    @classmethod
    def preprocess(cls, df: pd.DataFrame, config: dict, target_column: str = None) -> tuple[pd.DataFrame, dict]:
        """Apply full preprocessing pipeline according to configuration (US-14)."""
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid dataset DataFrame.")

        if df.empty or len(df) < 2:
            raise ValueError("The dataset must contain at least 2 rows to preprocess.")

        if target_column is not None and target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' does not exist in dataset.")

        missing_strategy = str(config.get("missing_values", "none")).lower()
        encoding_strategy = str(config.get("encoding", "none")).lower()
        scaling_strategy = str(config.get("scaling", "none")).lower()
        fill_value = config.get("fill_value", None)

        rows_before = len(df)
        cols_before = len(df.columns)

        # 1. Missing values
        df_processed = cls.handle_missing_values(
            df,
            strategy=missing_strategy,
            fill_value=fill_value,
            target_column=target_column
        )

        # 2. Categorical encoding
        df_processed = cls.encode_features(
            df_processed,
            strategy=encoding_strategy,
            target_column=target_column
        )

        # 3. Feature scaling
        df_processed = cls.scale_features(
            df_processed,
            strategy=scaling_strategy,
            target_column=target_column
        )

        if len(df_processed) < 2:
            raise ValueError("Preprocessing resulted in fewer than 2 rows.")

        if len(df_processed.columns) < 2:
            raise ValueError("Preprocessing resulted in fewer than 2 columns.")

        summary = {
            "missing_values": missing_strategy,
            "encoding": encoding_strategy,
            "scaling": scaling_strategy,
            "target_column": target_column,
            "rows_before": rows_before,
            "rows_after": len(df_processed),
            "columns_before": cols_before,
            "columns_after": len(df_processed.columns),
            "new_columns": [str(c) for c in df_processed.columns],
        }

        return df_processed, summary

    @staticmethod
    def split(X, y, test_size=0.2, random_state=42, stratify=None):
        """Split aligned samples, defaulting to 80% train and 20% test."""
        if isinstance(test_size, bool) or not isinstance(test_size, Real):
            raise ValueError("test_size must be a finite number greater than 0 and less than 1.")
        test_size = float(test_size)
        if not math.isfinite(test_size) or not 0 < test_size < 1:
            raise ValueError("test_size must be a finite number greater than 0 and less than 1.")

        try:
            sample_count = len(X)
            target_count = len(y)
        except TypeError as exc:
            raise ValueError("Features and target must be sized collections.") from exc
        if sample_count != target_count:
            raise ValueError("Features and target must contain the same number of samples.")
        if sample_count < 2:
            raise ValueError("At least 2 samples are required to create training and testing partitions.")

        test_count = math.ceil(sample_count * test_size)
        train_count = sample_count - test_count
        if test_count < 1 or train_count < 1:
            raise ValueError(
                "The selected split leaves no samples in the training or testing partition. "
                "Choose a different ratio or provide more data."
            )

        if stratify is not None:
            if len(stratify) != sample_count:
                raise ValueError("Stratification labels must match the number of samples.")
            class_count = len(pd.unique(stratify))
            if test_count < class_count or train_count < class_count:
                raise ValueError(
                    "The selected split does not leave enough training and testing samples "
                    "to represent every target class. Choose a different ratio or provide more data."
                )

        try:
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            raise ValueError(
                "The selected split cannot produce valid training and testing partitions. "
                "Check the sample count and target class distribution."
            ) from exc

    @staticmethod
    def build_pipeline(X_train, estimator, scale_numeric=False):
        """Build preprocessing that is fitted only on the training partition."""
        numeric_columns = list(X_train.select_dtypes(include="number").columns)
        categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
        transformers = []
        if numeric_columns:
            numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            numeric_indices = [list(X_train.columns).index(c) for c in numeric_columns]
            transformers.append(("numeric", Pipeline(numeric_steps), numeric_indices))
        if categorical_columns:
            categorical_steps = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ])
            categorical_indices = [list(X_train.columns).index(c) for c in categorical_columns]
            transformers.append(("categorical", categorical_steps, categorical_indices))
        if not transformers:
            raise ValueError("Select at least one usable feature column for training.")
        return Pipeline([
            ("preprocessor", ColumnTransformer(transformers)),
            ("estimator", estimator),
        ])
