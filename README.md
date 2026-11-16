# Hospital 30-Day Readmission Predictor

A full-stack web application that predicts a patient's 30-day hospital readmission
risk from their record, with explainable, reviewable output instead of a bare
probability score.

Built as an OJT (On-the-Job Training) project, data science track.

---

## Overview

Care coordinators today rely on manual chart review to judge which discharged
patients are likely to be readmitted. This project turns that judgment call into a
supported decision: a patient's record goes in, and a risk score comes out, along
with the specific factors driving that score and a flag when the model itself is
uncertain, so a human stays in the loop for the calls that matter.

---

## Features

- **Score a patient:** enter details in a form and get a risk percentage, a Low / Medium / High tier and a plain-language summary
- **Explainable output:** the top factors behind each score, shown as risk-raising or risk-lowering (SHAP)
- **Needs-review flag:** scores close to the decision threshold are marked as uncertain
- **Cost-based threshold:** the decision threshold weighs a missed readmission as costlier than an unnecessary follow-up, instead of using a flat 0.5
- **Ward batch scoring:** upload a CSV of patients and get them ranked highest risk first; a bad row never blocks the rest
- **Model insights:** comparison of Logistic Regression, Random Forest and XGBoost, confusion matrix, and performance by age group and illness severity
- **Input validation:** invalid values (for example an unknown severity) are rejected with a clear message

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React (JavaScript), Vite |
| Backend | Python, FastAPI |
| ML | scikit-learn, XGBoost, SHAP |

---

## Dataset

**AV Healthcare Analytics II** (Analytics Vidhya) — ~318,000 hospital encounters across
~92,000 unique patients, with admission details such as department, ward type,
severity of illness, type of admission, age group, and admission deposit.

The original dataset has no dates and no readmission label, so the following were
added on top of it:

- **Synthetic admission/discharge dates:** generated for each encounter; not real hospital records.
- **`readmitted_30_days` target:** derived from the gap between a patient's consecutive encounters on that synthetic timeline.
- **Admission-history features:** `previous_encounters`, `previous_30d/90d/365d_encounters`, `previous_avg_los`, `previous_readmission_count`, etc., computed from each patient's own encounter history.

> **Limitation:** because the dates and target are synthetic, the model has little
> real predictive power on this data (test AUC about 0.55). Results demonstrate the
> pipeline and approach, not clinical performance. With a dataset that has real
> dates and readmission labels, the same pipeline can be retrained.

Source: Analytics Vidhya, Healthcare Analytics II hackathon dataset.

---

## Repository Structure

```
Hospital_Readmission_Predictor/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── raw/                      # training data (synthetic dates + derived target)
├── notebooks/
│   ├── 01_eda.ipynb              # exploratory data analysis
│   └── 02_model_training_and_evaluation.ipynb
├── docs/
│   └── PRD.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py               # FastAPI app: /predict, /predict/batch, /model/*
│       └── ml/
│           ├── pipeline.py       # feature engineering + preprocessing
│           ├── train.py          # model training and evaluation
│           ├── predictor.py      # inference + SHAP explanations
│           └── artifacts/        # trained model and evaluation results
└── frontend/
    ├── package.json
    └── src/
        ├── App.jsx
        └── components/           # patient form, result card, batch upload, model insights
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`. Check `/health` to confirm the model loaded.
Interactive API docs are at `http://localhost:8000/docs`.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:5173`. It talks to the backend through a dev proxy, so
start the backend first.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Score one patient |
| POST | `/predict/batch` | Score a ward from an uploaded CSV |
| GET | `/predict/batch/template` | Download a sample CSV |
| POST | `/predict/batch-json` | Score a list of patients sent as JSON |
| GET | `/model/metrics` | Model comparison, calibration, confusion matrix |
| GET | `/model/fairness` | Performance by age group and severity |
| GET | `/model/error-cases` | Sample false positives and false negatives |
| GET | `/health` | Service and model status |

---

## License

This project is licensed under the MIT License. See [`LICENSE`](./LICENSE) for
details.
