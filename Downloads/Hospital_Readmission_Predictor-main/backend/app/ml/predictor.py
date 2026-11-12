"""
Production Inference Engine for Hospital 30-Day Readmission Predictor.
Fulfills:
- US-001: Score a Patient
- US-002: See Why (Top 5 SHAP factors with direction)
- US-003: Cost-based Risk Tiering (Low / Medium / High)
- US-004: Explicit Uncertainty Flagging ("needs review")
- US-005: Batch Scoring for Ward Triage
- US-010: Plain-Language Score Explanation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

FEATURE_DISPLAY_NAMES = {
    "previous_emergency_count": "Prior Emergency Visits",
    "previous_encounters": "Total Prior Encounters",
    "previous_readmission_count": "Prior 30-Day Readmissions",
    "previous_hospital_days": "Total Prior Hospital Days",
    "previous_avg_los": "Prior Average Length of Stay",
    "previous_max_los": "Prior Maximum Length of Stay",
    "days_since_previous_discharge": "Days Since Last Discharge",
    "length_of_stay_days": "Current Length of Stay",
    "Admission_Deposit": "Admission Deposit",
    "Available Extra Rooms in Hospital": "Available Hospital Rooms",
    "Bed Grade": "Bed Grade",
    "Visitors with Patient": "Visitors with Patient",
    "age_num": "Patient Age",
    "severity_num": "Illness Severity",
    "readmission_ratio": "Historical Readmission Rate",
    "emergency_ratio": "Historical Emergency Admission Ratio",
    "los_deviation": "Stay Duration Deviation from Baseline",
    "deposit_per_day": "Deposit-to-Stay Ratio",
    "recent_accel": "Recent Encounter Acceleration (30d vs 90d)",
    "high_utilizer": "High Healthcare Utilization History",
    "is_first_admission": "First-Time Hospital Admission",
}


class ReadmissionPredictor:
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.model = joblib.load(self.artifacts_dir / "best_model.joblib")
            self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.joblib")
            self.explainer = joblib.load(self.artifacts_dir / "explainer.joblib")
            with open(self.artifacts_dir / "metadata.json") as f:
                self.metadata = json.load(f)

            self.thresholds = self.metadata.get("thresholds", {
                "low_upper": 0.145,
                "decision_threshold": 0.195,
                "high_lower": 0.275,
                "uncertainty_band": [0.170, 0.220],
            })
            self.feature_names = self.metadata.get("feature_names", [])
        except Exception as e:
            raise RuntimeError(f"Failed to load ML artifacts from {self.artifacts_dir}: {e}")

    def _determine_tier(self, prob: float) -> str:
        if prob < self.thresholds["low_upper"]:
            return "Low"
        elif prob < self.thresholds["high_lower"]:
            return "Medium"
        else:
            return "High"

    def _check_uncertainty(self, prob: float) -> tuple[bool, Optional[str]]:
        low_b, high_b = self.thresholds["uncertainty_band"]
        if low_b <= prob <= high_b:
            return True, f"Borderline probability ({prob:.1%}) within [{low_b:.1%} - {high_b:.1%}] margin — clinical review recommended."
        return False, None

    def _clean_feature_name(self, raw_name: str) -> str:
        # Remove OHE prefixes like cat__Department_
        if "__" in raw_name:
            raw_name = raw_name.split("__")[-1]
        for prefix in ["Department_", "Type of Admission_", "Ward_Type_", "Ward_Facility_Code_", "Hospital_type_code_", "Hospital_region_code_"]:
            if raw_name.startswith(prefix):
                return raw_name.replace(prefix, "") + f" ({prefix.rstrip('_')})"
        return FEATURE_DISPLAY_NAMES.get(raw_name, raw_name.replace("_", " ").title())

    def _generate_plain_language_summary(
        self,
        risk_tier: str,
        prob: float,
        top_factors: List[Dict[str, Any]],
        needs_review: bool,
    ) -> str:
        """
        US-010: Generate plain language summary explaining score drivers.
        """
        risk_drivers = [f for f in top_factors if f["direction"] == "increases_risk"]
        protective_factors = [f for f in top_factors if f["direction"] == "decreases_risk"]

        parts = [f"Patient evaluated at {risk_tier} Risk ({prob:.1%} readmission probability)."]

        if risk_drivers:
            driver_phrases = [d["display_name"] for d in risk_drivers[:3]]
            parts.append(f"Elevated risk is primarily driven by {', '.join(driver_phrases)}.")

        if protective_factors:
            protect_phrases = [p["display_name"] for p in protective_factors[:2]]
            parts.append(f"Mitigated in part by favorable {', '.join(protect_phrases)}.")

        if needs_review:
            parts.append("Note: Score falls within the uncertainty band; coordinator clinical judgment is strongly advised.")

        return " ".join(parts)

    def predict_single(self, record: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
        """
        Scores a single patient record, returns risk probability, tier, uncertainty, SHAP, and summary.
        """
        if isinstance(record, dict):
            row = pd.DataFrame([record])
        elif isinstance(record, pd.Series):
            row = pd.DataFrame([record.to_dict()])
        else:
            row = pd.DataFrame(record)

        # Preprocess features
        X_proc = self.preprocessor.transform(row)

        # Predict probability
        prob = float(self.model.predict_proba(X_proc)[0, 1])
        tier = self._determine_tier(prob)
        needs_review, uncertainty_msg = self._check_uncertainty(prob)

        # Compute SHAP explanation
        shap_res = self.explainer(X_proc)
        shap_vals = shap_res.values[0]

        top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
        top_factors = []
        for idx in top_indices:
            raw_feat = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            display_name = self._clean_feature_name(raw_feat)
            shap_val = float(shap_vals[idx])
            top_factors.append({
                "feature": raw_feat,
                "display_name": display_name,
                "shap_value": round(shap_val, 4),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
            })

        plain_language = self._generate_plain_language_summary(
            risk_tier=tier,
            prob=prob,
            top_factors=top_factors,
            needs_review=needs_review,
        )

        return {
            "risk_probability": round(prob, 4),
            "risk_tier": tier,
            "decision_threshold": self.thresholds["decision_threshold"],
            "needs_review": needs_review,
            "uncertainty_reason": uncertainty_msg,
            "top_factors": top_factors,
            "plain_language_summary": plain_language,
            "model_version": self.metadata.get("model_version", "1.0.0"),
        }

    def predict_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        US-005: Triage an entire ward (CSV/DataFrame) with row-level error resilience.
        """
        results = []
        for idx, row in df.iterrows():
            try:
                single_res = self.predict_single(row)
                single_res["row_index"] = int(idx)
                single_res["case_id"] = row.get("case_id", idx)
                single_res["patient_id"] = row.get("patientid", None)
                results.append(single_res)
            except Exception as e:
                results.append({
                    "row_index": int(idx),
                    "case_id": row.get("case_id", idx),
                    "error": str(e),
                    "status": "SCORING_FAILED",
                })
        return results


# Module-level singleton instance
_predictor_instance: Optional[ReadmissionPredictor] = None


def get_predictor() -> ReadmissionPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ReadmissionPredictor()
    return _predictor_instance

