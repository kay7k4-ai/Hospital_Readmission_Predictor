# Product Requirements Document (PRD)
## Hospital 30-Day Readmission Predictor

---

## 1. Product Goal

Turn a patient's discharge record into a reliable, explainable, and reviewable 30-day
readmission risk assessment — delivered through a full-stack web application that a
care coordinator can act on and an ML analyst can audit.

---

## 2. Feature Priorities

| Feature | Priority |
|---|---|
| Upload patient record (single form entry) | Must |
| Batch upload (CSV) | Must |
| Risk prediction (readmission probability) | Must |
| Risk tiering (Low / Medium / High) | Must |
| Feature contribution explanation (SHAP) | Must |
| Confidence / uncertainty flagging | Must |
| Patient Risk Dashboard (React) | Must |
| Evaluation dashboard (AUC, calibration, confusion matrix) | Should |
| Compare multiple ML models (LogReg vs RF vs XGBoost) | Should |
| Batch scoring for full ward / export to CSV | Could |
| Prediction history / audit trail per patient | Could |

---

## 3. User Stories

**US-001 — Score a Patient**
As a care coordinator, I want to submit a patient's discharge data so the system
predicts their 30-day readmission risk.
*Acceptance:* form validates required fields; shows loading state; returns a score or
a clear error.

**US-002 — See Why**
As a coordinator, I want the top contributing risk factors shown alongside the score
so I can act on the specific cause, not just a number.
*Acceptance:* top 5 SHAP features displayed with direction (increases/decreases risk).

**US-003 — Read Risk Tier**
As a coordinator, I want the probability converted into Low/Medium/High so I know
urgency without interpreting raw numbers.
*Acceptance:* tier thresholds are cost-based, not default 0.5, and documented.

**US-004 — Flag Uncertainty**
As a coordinator, I want low-confidence predictions clearly flagged so I know when to
rely on clinical judgment instead of the model.
*Acceptance:* predictions inside a defined uncertainty band are visually distinct and
labeled "needs review."

**US-005 — Batch Score a Ward**
As a coordinator, I want to upload a CSV of multiple patients so I can triage an
entire ward at once.
*Acceptance:* CSV validated against schema; per-row errors reported without failing
the whole batch.

**US-010 — Understand the Score in Plain Language**
As a coordinator, I want the risk explanation written as a plain sentence (e.g. "High
number of recent lab procedures is pushing risk up"), not raw feature names and SHAP
values, so I can act on it without ML background.
*Acceptance:* every prediction includes a generated plain-language summary alongside
the SHAP chart.

**US-011 — Track Follow-Up**
As a coordinator, I want to mark a patient as "followed up" after I've acted on their
risk flag so the team knows the case is handled and it's not reviewed twice.
*Acceptance:* status toggle persists per patient and is visible in the dashboard and
batch view.

**US-012 — Find a Patient**
As a coordinator, I want to search or filter patients within a scored batch so I can
quickly locate a specific case without scrolling a full ward list.
*Acceptance:* search by patient ID/name field and filter by risk tier and follow-up
status.

**US-006 — Evaluate Model**
I want calibration and discrimination metrics (AUC, Brier score, precision/recall at
chosen threshold) so I can judge trustworthiness, not just accuracy.
*Acceptance:* metrics dashboard reads from held-out temporal test set.

**US-007 — Inspect Errors**
I want to browse false-positive and false-negative cases so I can identify failure
patterns.
*Acceptance:* filterable table of misclassified cases with their SHAP explanations.

**US-008 — Check Fairness**
I want subgroup performance (age band, gender, etc.) so I can catch systematic bias
before deployment.
*Acceptance:* per-subgroup AUC/recall shown side by side with overall metrics.

**US-009 — Compare Models**
I want to compare multiple trained models (Logistic Regression, Random Forest,
XGBoost) side by side on AUC, calibration, and inference speed so I can justify which
model goes into production, not just pick the highest-accuracy one.
*Acceptance:* comparison view shows all three models on the same held-out temporal
test set with metrics displayed together.

---

## 4. Product States

`RECORD_SUBMITTED → VALIDATED → FEATURES_BUILT → RISK_SCORED → EXPLAINED →
REVIEW_REQUIRED / COMPLETED`
Failure: `SCORING_FAILED`

---

## 5. Product Principles

- **Evidence before interpretation** — every score ships with its top contributing
  features, never a bare number.
- **Component-level metrics** — model performance and app performance are measured
  and reported separately.
- **Explicit uncertainty** — low-confidence predictions are flagged, not silently
  presented as equally reliable.
- **Reproducible configurations** — model version, threshold, and training data
  snapshot are logged with every prediction.
- **Privacy by design** — no real patient PII; de-identified/synthetic data only.
