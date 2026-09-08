from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Hospital 30-Day Readmission Predictor API")


class PatientRecord(BaseModel):
    age: int
    num_prior_admissions: int
    num_medications: int
    length_of_stay: int
    has_follow_up_scheduled: bool


class RiskResponse(BaseModel):
    risk_probability: float
    risk_tier: str
    top_factors: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=RiskResponse)
def predict_readmission_risk(record: PatientRecord):
    # Placeholder logic — real model (US-001, US-002, US-003) not wired up yet.
    # Replace with trained model inference + SHAP explanation.
    dummy_score = 0.42
    tier = "Medium" if dummy_score >= 0.3 else "Low"
    return RiskResponse(
        risk_probability=dummy_score,
        risk_tier=tier,
        top_factors=["num_prior_admissions", "length_of_stay"],
    )
