# pyrefly: ignore[missing-import]
"""
Hospital 30-Day Readmission Predictor - FastAPI Application.
Implements:
- US-001: Score a Patient
- US-002: Top 5 SHAP factors with directional impact
- US-003: Cost-based Risk Tiering (Low / Medium / High)
- US-004: Explicit Uncertainty Flagging ("needs review")
- US-005: Batch Ward Scoring with CSV or JSON upload
- US-006 & US-009: Model comparison, calibration, confusion matrix metrics
- US-007: Filterable error cases (False Positives / False Negatives with SHAP)
- US-008: Subgroup fairness metrics (Age, Illness Severity)
- US-010: Plain-Language Score Summaries
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import io
import pandas as pd
# pyrefly: ignore[missing-import]
from fastapi import FastAPI, File, HTTPException, UploadFile
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field

try:
    from backend.app.ml.predictor import get_predictor
except ModuleNotFoundError:
    from app.ml.predictor import get_predictor

app = FastAPI(
    title="Hospital 30-Day Readmission Predictor API",
    description="Production ML API providing explainable, cost-calibrated readmission risk predictions.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "ml" / "artifacts"


class PatientRecord(BaseModel):
    # Core clinical variables
    age: Optional[str] = Field("51-60", description="Age bracket e.g. 51-60 or numeric midpoint")
    severity_of_illness: Optional[str] = Field("Moderate", description="Minor, Moderate, Extreme")
    type_of_admission: Optional[str] = Field("Emergency", description="Emergency, Trauma, Urgent")
    department: Optional[str] = Field("gynecology", description="gynecology, anesthesia, radiotherapy, surgery, TB & Chest disease")
    ward_type: Optional[str] = Field("R", description="Ward type code R, S, Q, P, T, U")
    ward_facility_code: Optional[str] = Field("F", description="Ward facility code A-F")
    bed_grade: Optional[float] = Field(2.0, description="Bed grade 1.0 - 4.0")
    length_of_stay: Optional[float] = Field(5.0, description="Current length of stay in days")
    
    # Historical utilization features
    num_prior_admissions: Optional[int] = Field(0, description="Total previous hospital encounters")
    previous_emergency_count: Optional[int] = Field(0, description="Prior emergency encounters")
    previous_readmission_count: Optional[int] = Field(0, description="Prior 30-day readmissions")
    previous_hospital_days: Optional[float] = Field(0.0, description="Total historical hospital days")
    previous_avg_los: Optional[float] = Field(0.0, description="Prior average length of stay")
    previous_max_los: Optional[float] = Field(0.0, description="Prior max length of stay")
    days_since_previous_discharge: Optional[float] = Field(None, description="Days since last discharge")
    previous_30d_encounters: Optional[int] = Field(0, description="Encounters in last 30 days")
    previous_90d_encounters: Optional[int] = Field(0, description="Encounters in last 90 days")
    previous_365d_encounters: Optional[int] = Field(0, description="Encounters in last 365 days")
    
    # Hospital context
    available_extra_rooms: Optional[int] = Field(2, description="Available extra hospital rooms")
    visitors_with_patient: Optional[int] = Field(2, description="Number of visitors accompanying patient")
    admission_deposit: Optional[float] = Field(4500.0, description="Admission deposit in dollars")
    hospital_type_code: Optional[str] = Field("c", description="Hospital type code a-g")
    hospital_region_code: Optional[str] = Field("X", description="Hospital region code X, Y, Z")
    has_follow_up_scheduled: Optional[bool] = Field(False, description="Post-discharge follow-up scheduled")


class FactorContribution(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    direction: str


class RiskResponse(BaseModel):
    risk_probability: float
    risk_tier: str
    decision_threshold: float
    needs_review: bool
    uncertainty_reason: Optional[str] = None
    top_factors: List[str]  # String list for backward-compatibility with App.tsx
    factor_details: List[FactorContribution]
    plain_language_summary: str
    model_version: str


def _patient_to_raw_dict(p: PatientRecord) -> Dict[str, Any]:
    """Maps PatientRecord attributes to raw model feature column names."""
    # Convert age string if integer passed
    age_str = str(p.age)
    if "-" not in age_str and age_str.isdigit():
        num_age = int(age_str)
        tens = (num_age // 10) * 10
        age_str = f"{tens}-{tens+10}"

    return {
        "Age": age_str,
        "Severity of Illness": p.severity_of_illness or "Moderate",
        "Type of Admission": p.type_of_admission or "Emergency",
        "Department": p.department or "gynecology",
        "Ward_Type": p.ward_type or "R",
        "Ward_Facility_Code": p.ward_facility_code or "F",
        "Bed Grade": p.bed_grade or 2.0,
        "length_of_stay_days": p.length_of_stay or 5.0,
        "encounter_number": (p.num_prior_admissions or 0) + 1,
        "previous_encounters": p.num_prior_admissions or 0,
        "previous_emergency_count": p.previous_emergency_count or 0,
        "previous_readmission_count": p.previous_readmission_count or 0,
        "previous_hospital_days": p.previous_hospital_days or 0.0,
        "previous_avg_los": p.previous_avg_los or 0.0,
        "previous_max_los": p.previous_max_los or 0.0,
        "previous_30d_encounters": p.previous_30d_encounters or 0,
        "previous_90d_encounters": p.previous_90d_encounters or 0,
        "previous_365d_encounters": p.previous_365d_encounters or 0,
        "previous_urgent_count": 0,
        "days_since_previous_discharge": p.days_since_previous_discharge,
        "Available Extra Rooms in Hospital": p.available_extra_rooms or 2,
        "Visitors with Patient": p.visitors_with_patient or 2,
        "Admission_Deposit": p.admission_deposit or 4500.0,
        "Hospital_type_code": p.hospital_type_code or "c",
        "Hospital_region_code": p.hospital_region_code or "X",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Hospital 30-Day Readmission Predictor",
        "model_loaded": ARTIFACTS_DIR.exists() and (ARTIFACTS_DIR / "best_model.joblib").exists(),
    }


@app.post("/predict", response_model=RiskResponse)
def predict_readmission_risk(record: PatientRecord):
    """
    US-001, US-002, US-003, US-004, US-010:
    Calculates cost-calibrated readmission risk, tier, uncertainty flag, SHAP factors, and plain summary.
    """
    try:
        predictor = get_predictor()
        raw_dict = _patient_to_raw_dict(record)
        pred = predictor.predict_single(raw_dict)

        top_factors_str = [
            f"{f['display_name']} ({'+' if f['direction'] == 'increases_risk' else '-'}{abs(f['shap_value']):.3f})"
            for f in pred["top_factors"]
        ]

        factor_details = [
            FactorContribution(
                feature=f["feature"],
                display_name=f["display_name"],
                shap_value=f["shap_value"],
                direction=f["direction"],
            )
            for f in pred["top_factors"]
        ]

        return RiskResponse(
            risk_probability=pred["risk_probability"],
            risk_tier=pred["risk_tier"],
            decision_threshold=pred["decision_threshold"],
            needs_review=pred["needs_review"],
            uncertainty_reason=pred["uncertainty_reason"],
            top_factors=top_factors_str,
            factor_details=factor_details,
            plain_language_summary=pred["plain_language_summary"],
            model_version=pred["model_version"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")


@app.post("/predict/batch")
async def batch_predict_readmission(file: Optional[UploadFile] = File(None), records: Optional[List[PatientRecord]] = None):
    """
    US-005: Batch score a ward from uploaded CSV or JSON records.
    Per-row error isolation ensures invalid rows do not fail the whole batch.
    """
    predictor = get_predictor()

    if file is not None:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
        results = predictor.predict_batch(df)
        return {
            "total_submitted": len(df),
            "successful_predictions": len([r for r in results if r.get("status") != "SCORING_FAILED"]),
            "patients": results,
        }
    elif records is not None:
        dict_records = [_patient_to_raw_dict(r) for r in records]
        df = pd.DataFrame(dict_records)
        results = predictor.predict_batch(df)
        return {
            "total_submitted": len(records),
            "successful_predictions": len([r for r in results if r.get("status") != "SCORING_FAILED"]),
            "patients": results,
        }
    else:
        raise HTTPException(status_code=400, detail="Must provide either a CSV file or list of patient records.")


@app.get("/model/metrics")
def get_model_metrics():
    """
    US-006 & US-009: Exposes model comparison benchmark, calibration curve, and confusion matrix.
    """
    try:
        with open(ARTIFACTS_DIR / "model_comparison.json") as f:
            comparison = json.load(f)
        with open(ARTIFACTS_DIR / "calibration_curve.json") as f:
            calibration = json.load(f)
        with open(ARTIFACTS_DIR / "confusion_matrix.json") as f:
            confusion = json.load(f)
        with open(ARTIFACTS_DIR / "metadata.json") as f:
            metadata = json.load(f)

        return {
            "comparison": comparison,
            "calibration": calibration,
            "confusion_matrix": confusion,
            "thresholds": metadata.get("thresholds"),
            "model_version": metadata.get("model_version"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read evaluation artifacts: {str(e)}")


@app.get("/model/fairness")
def get_subgroup_fairness():
    """
    US-008: Subgroup fairness metrics by Age and Severity of Illness.
    """
    try:
        with open(ARTIFACTS_DIR / "subgroup_fairness.json") as f:
            fairness = json.load(f)
        return fairness
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read fairness artifact: {str(e)}")


@app.get("/model/error-cases")
def get_error_cases():
    """
    US-007: Filterable false positive and false negative audit cases with SHAP breakdowns.
    """
    try:
        with open(ARTIFACTS_DIR / "error_cases.json") as f:
            error_cases = json.load(f)
        return {"total_cases": len(error_cases), "cases": error_cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read error cases: {str(e)}")


@app.get("/model/info")
def get_model_info():
    """Returns model metadata, active thresholds, and configuration."""
    try:
        with open(ARTIFACTS_DIR / "metadata.json") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {str(e)}")
