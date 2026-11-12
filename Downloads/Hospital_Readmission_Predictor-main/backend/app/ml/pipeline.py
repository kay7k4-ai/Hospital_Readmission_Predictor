"""
Feature engineering and preprocessing pipeline definitions for Hospital 30-Day Readmission Predictor.
Ensures strict featurization ordering (fitted only on training split).
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ordinal mappings
AGE_MAP = {
    "0-10": 5.0,
    "11-20": 15.0,
    "21-30": 25.0,
    "31-40": 35.0,
    "41-50": 45.0,
    "51-60": 55.0,
    "61-70": 65.0,
    "71-80": 75.0,
    "81-90": 85.0,
    "91-100": 95.0,
}

SEVERITY_MAP = {
    "Minor": 1.0,
    "Moderate": 2.0,
    "Extreme": 3.0,
}

# Raw column definitions
RAW_CATEGORICAL_COLS = [
    "Hospital_type_code",
    "Hospital_region_code",
    "Department",
    "Ward_Type",
    "Ward_Facility_Code",
    "Type of Admission",
]

RAW_NUMERICAL_COLS = [
    "Available Extra Rooms in Hospital",
    "Bed Grade",
    "Visitors with Patient",
    "Admission_Deposit",
    "length_of_stay_days",
    "encounter_number",
    "previous_encounters",
    "days_since_previous_discharge",
    "previous_30d_encounters",
    "previous_90d_encounters",
    "previous_365d_encounters",
    "previous_hospital_days",
    "previous_avg_los",
    "previous_max_los",
    "previous_readmission_count",
    "previous_emergency_count",
    "previous_urgent_count",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs clinically sound derived features from raw hospital encounter inputs:
    - Age band midpoint & numerical severity encoding
    - Utilization acceleration and ratios
    - First admission indicator
    - Log-transformed discharge gap
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Handle age and severity mapping
        if "Age" in df.columns:
            df["age_num"] = df["Age"].astype(str).map(AGE_MAP).fillna(55.0)
        elif "age_num" not in df.columns:
            df["age_num"] = 55.0

        if "Severity of Illness" in df.columns:
            df["severity_num"] = df["Severity of Illness"].astype(str).map(SEVERITY_MAP).fillna(2.0)
        elif "severity_num" not in df.columns:
            df["severity_num"] = 2.0

        # Previous encounters & utilization ratios
        prev_enc = df.get("previous_encounters", pd.Series(0, index=df.index)).fillna(0).astype(float)
        df["is_first_admission"] = (prev_enc == 0).astype(float)

        prev_readm = df.get("previous_readmission_count", pd.Series(0, index=df.index)).fillna(0).astype(float)
        df["readmission_ratio"] = prev_readm / (prev_enc + 1.0)

        prev_emerg = df.get("previous_emergency_count", pd.Series(0, index=df.index)).fillna(0).astype(float)
        df["emergency_ratio"] = prev_emerg / (prev_enc + 1.0)

        los = df.get("length_of_stay_days", pd.Series(5.0, index=df.index)).fillna(5.0).astype(float)
        prev_avg_los = df.get("previous_avg_los", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
        df["los_deviation"] = los - prev_avg_los
        df["los_ratio"] = los / (prev_avg_los + 1.0)

        deposit = df.get("Admission_Deposit", pd.Series(4000.0, index=df.index)).fillna(4000.0).astype(float)
        df["deposit_per_day"] = deposit / (los + 1.0)

        visitors = df.get("Visitors with Patient", pd.Series(2, index=df.index)).fillna(2).astype(float)
        df["visitors_per_day"] = visitors / (los + 1.0)

        prev_30d = df.get("previous_30d_encounters", pd.Series(0, index=df.index)).fillna(0).astype(float)
        prev_90d = df.get("previous_90d_encounters", pd.Series(0, index=df.index)).fillna(0).astype(float)
        df["recent_accel"] = prev_30d / (prev_90d + 1.0)

        df["high_utilizer"] = (prev_enc >= 3).astype(float)

        days_since = df.get("days_since_previous_discharge", pd.Series(np.nan, index=df.index))
        days_since_vals = pd.to_numeric(days_since, errors="coerce").fillna(999.0).clip(lower=0.0).to_numpy(dtype=float)
        df["days_since_prev_log"] = np.log1p(days_since_vals)

        return df


def build_preprocessor() -> Tuple[Pipeline, List[str], List[str]]:
    """
    Builds the ColumnTransformer for numerical and categorical features.
    """
    engineered_num_cols = RAW_NUMERICAL_COLS + [
        "age_num",
        "severity_num",
        "is_first_admission",
        "readmission_ratio",
        "emergency_ratio",
        "los_deviation",
        "los_ratio",
        "deposit_per_day",
        "visitors_per_day",
        "recent_accel",
        "high_utilizer",
        "days_since_prev_log",
    ]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, engineered_num_cols),
            ("cat", cat_pipeline, RAW_CATEGORICAL_COLS),
        ]
    )

    full_pipeline = Pipeline(
        steps=[
            ("feature_engineer", ClinicalFeatureEngineer()),
            ("preprocessor", col_transformer),
        ]
    )

    return full_pipeline, engineered_num_cols, RAW_CATEGORICAL_COLS

