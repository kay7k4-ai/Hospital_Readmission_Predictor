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

import io
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
# pyrefly: ignore[missing-import]
from fastapi import FastAPI, File, HTTPException, UploadFile
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore[missing-import]
from fastapi.responses import Response
# pyrefly: ignore[missing-import]
from pydantic import BaseModel, Field, ValidationError

try:
    from backend.app.ml.predictor import get_predictor
except ModuleNotFoundError:
    from app.ml.predictor import get_predictor

logger = logging.getLogger("readmission.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Hospital 30-Day Readmission Predictor API",
    description="ML API providing explainable, cost-calibrated readmission risk predictions.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "ml" / "artifacts"

AGE_BRACKETS = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]


class PatientRecord(BaseModel):
    """Validated input for one patient. Invalid values (e.g. severity 'Banana') are rejected with a 422."""

    # Core clinical variables
    age: str = Field("51-60", description="Age bracket like 51-60, or a plain age like 57")
    severity_of_illness: Literal["Minor", "Moderate", "Extreme"] = "Moderate"
    type_of_admission: Literal["Emergency", "Trauma", "Urgent"] = "Emergency"
    department: Literal["gynecology", "anesthesia", "radiotherapy", "surgery", "TB & Chest disease"] = "gynecology"
    ward_type: Literal["P", "Q", "R", "S", "T", "U"] = "R"
    ward_facility_code: Literal["A", "B", "C", "D", "E", "F"] = "F"
    bed_grade: float = Field(2.0, ge=1.0, le=4.0, description="Bed grade 1.0 - 4.0")
    length_of_stay: float = Field(5.0, ge=0.0, le=365.0, description="Current length of stay in days")

    # Historical utilization features
    num_prior_admissions: int = Field(0, ge=0, le=100, description="Total previous hospital encounters")
    previous_emergency_count: int = Field(0, ge=0, le=100, description="Prior emergency encounters")
    previous_urgent_count: int = Field(0, ge=0, le=100, description="Prior urgent encounters")
    previous_readmission_count: int = Field(0, ge=0, le=100, description="Prior 30-day readmissions")
    previous_hospital_days: float = Field(0.0, ge=0.0, description="Total historical hospital days")
    previous_avg_los: float = Field(0.0, ge=0.0, description="Prior average length of stay")
    previous_max_los: float = Field(0.0, ge=0.0, description="Prior max length of stay")
    days_since_previous_discharge: Optional[float] = Field(None, ge=0.0, description="Days since last discharge (blank if first visit)")
    previous_30d_encounters: int = Field(0, ge=0, le=100, description="Encounters in last 30 days")
    previous_90d_encounters: int = Field(0, ge=0, le=100, description="Encounters in last 90 days")
    previous_365d_encounters: int = Field(0, ge=0, le=100, description="Encounters in last 365 days")

    # Hospital context
    available_extra_rooms: int = Field(2, ge=0, le=100, description="Available extra hospital rooms")
    visitors_with_patient: int = Field(2, ge=0, le=100, description="Number of visitors accompanying patient")
    admission_deposit: float = Field(4500.0, ge=0.0, description="Admission deposit in dollars")
    hospital_type_code: Literal["a", "b", "c", "d", "e", "f", "g"] = "c"
    hospital_region_code: Literal["X", "Y", "Z"] = "X"

    # Workflow note only: NOT a model input (the model was not trained on follow-up data).
    has_follow_up_scheduled: bool = Field(False, description="Recorded for the coordinator; not used by the model")


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
    top_factors: List[str]  # Readable strings, e.g. "Prior Emergency Visits (+0.120)"
    factor_details: List[FactorContribution]
    plain_language_summary: str
    model_version: str


def _normalise_age(age: Any) -> str:
    """Accepts '51-60' or a plain number like 57 and returns a valid bracket."""
    text = str(age).strip()
    if text in AGE_BRACKETS:
        return text
    try:
        value = int(float(text))
    except ValueError:
        raise ValueError(f"age must be a bracket like '51-60' or a number, got '{age}'")
    if not 0 <= value <= 100:
        raise ValueError("age must be between 0 and 100")
    idx = min(max((value - 1) // 10, 0), 9) if value > 10 else 0
    return AGE_BRACKETS[idx]


def _patient_to_raw_dict(p: PatientRecord) -> Dict[str, Any]:
    """Maps PatientRecord attributes to raw model feature column names."""
    return {
        "Age": _normalise_age(p.age),
        "Severity of Illness": p.severity_of_illness,
        "Type of Admission": p.type_of_admission,
        "Department": p.department,
        "Ward_Type": p.ward_type,
        "Ward_Facility_Code": p.ward_facility_code,
        "Bed Grade": p.bed_grade,
        "length_of_stay_days": p.length_of_stay,
        "encounter_number": p.num_prior_admissions + 1,
        "previous_encounters": p.num_prior_admissions,
        "previous_emergency_count": p.previous_emergency_count,
        "previous_readmission_count": p.previous_readmission_count,
        "previous_hospital_days": p.previous_hospital_days,
        "previous_avg_los": p.previous_avg_los,
        "previous_max_los": p.previous_max_los,
        "previous_30d_encounters": p.previous_30d_encounters,
        "previous_90d_encounters": p.previous_90d_encounters,
        "previous_365d_encounters": p.previous_365d_encounters,
        "previous_urgent_count": p.previous_urgent_count,
        "days_since_previous_discharge": p.days_since_previous_discharge,
        "Available Extra Rooms in Hospital": p.available_extra_rooms,
        "Visitors with Patient": p.visitors_with_patient,
        "Admission_Deposit": p.admission_deposit,
        "Hospital_type_code": p.hospital_type_code,
        "Hospital_region_code": p.hospital_region_code,
    }


def _format_response(pred: Dict[str, Any]) -> RiskResponse:
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
        pred = predictor.predict_single(_patient_to_raw_dict(record))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

    # Audit log: no patient identifiers, just the outcome.
    logger.info(
        "prediction prob=%.4f tier=%s needs_review=%s model=%s",
        pred["risk_probability"], pred["risk_tier"], pred["needs_review"], pred["model_version"],
    )
    return _format_response(pred)


# ---------------------------------------------------------------- batch scoring

BATCH_TEMPLATE_ROWS = [
    {"patient_ref": "P-001", "age": "71-80", "severity_of_illness": "Extreme", "type_of_admission": "Emergency",
     "department": "gynecology", "length_of_stay": 9, "num_prior_admissions": 4, "previous_emergency_count": 3,
     "previous_readmission_count": 2, "previous_365d_encounters": 4, "previous_hospital_days": 30,
     "previous_avg_los": 7.5, "previous_max_los": 12},
    {"patient_ref": "P-002", "age": "31-40", "severity_of_illness": "Minor", "type_of_admission": "Urgent",
     "department": "radiotherapy", "length_of_stay": 3, "num_prior_admissions": 0},
    {"patient_ref": "P-003", "age": "51-60", "severity_of_illness": "Moderate", "type_of_admission": "Trauma",
     "department": "surgery", "length_of_stay": 6, "num_prior_admissions": 1, "previous_emergency_count": 1,
     "previous_hospital_days": 5, "previous_avg_los": 5, "previous_max_los": 5, "days_since_previous_discharge": 40},
]


def _score_records(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Scores each row on its own: one bad row never fails the whole batch."""
    predictor = get_predictor()
    patients: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        ref = row.pop("patient_ref", None) or f"row-{i + 1}"
        try:
            record = PatientRecord(**row)
            pred = predictor.predict_single(_patient_to_raw_dict(record))
            patients.append({
                "row_index": i, "patient_ref": ref, "status": "OK",
                "risk_probability": pred["risk_probability"], "risk_tier": pred["risk_tier"],
                "needs_review": pred["needs_review"],
                "top_factors": [f["display_name"] for f in pred["top_factors"][:3]],
                "plain_language_summary": pred["plain_language_summary"],
            })
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            patients.append({"row_index": i, "patient_ref": ref, "status": "SCORING_FAILED", "error": msg})
        except Exception as e:
            patients.append({"row_index": i, "patient_ref": ref, "status": "SCORING_FAILED", "error": str(e)})

    ok = [p for p in patients if p["status"] == "OK"]
    ok.sort(key=lambda p: p["risk_probability"], reverse=True)
    failed = [p for p in patients if p["status"] != "OK"]
    return {
        "total_submitted": len(rows),
        "successful_predictions": len(ok),
        "failed_predictions": len(failed),
        "tier_counts": {t: sum(1 for p in ok if p["risk_tier"] == t) for t in ("High", "Medium", "Low")},
        "patients": ok + failed,  # highest risk first, failures at the bottom
    }


@app.get("/predict/batch/template")
def batch_template():
    """Downloadable sample CSV showing the accepted column names."""
    csv_text = pd.DataFrame(BATCH_TEMPLATE_ROWS).to_csv(index=False)
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_template.csv"},
    )


@app.post("/predict/batch")
async def batch_predict_csv(file: UploadFile = File(...)):
    """
    US-005: Batch score a ward from an uploaded CSV.
    Columns use the same names as the single-patient form (see /predict/batch/template);
    missing optional columns fall back to defaults. Rows are ranked highest risk first.
    """
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no rows.")
    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="CSV too large: max 5000 rows per upload.")

    rows: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        # Blank cells (NaN) are dropped so the field defaults apply.
        rows.append({k: v for k, v in rec.items()
                     if not (isinstance(v, float) and math.isnan(v))})
    return _score_records(rows)


@app.post("/predict/batch-json")
def batch_predict_json(records: List[PatientRecord]):
    """US-005: Batch score a list of patient records sent as JSON."""
    if not records:
        raise HTTPException(status_code=400, detail="Provide at least one patient record.")
    return _score_records([r.model_dump() for r in records])


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
