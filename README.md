# Hospital 30-Day Readmission Predictor

A full-stack web application that predicts a patient's 30-day hospital readmission
risk from their discharge data — with explainable, reviewable output instead of a
bare probability score.

Built as an OJT (On-the-Job Training) project, data science track.

---

## Overview

Care coordinators today rely on manual chart review to judge which discharged
patients are likely to be readmitted. This project turns that judgment call into a
supported decision: a patient's discharge record goes in, and a risk score comes
out — along with the specific factors driving that score and a flag when the model
itself is uncertain, so a human stays in the loop for the calls that matter.

---

## Features

- Score a single patient or upload a batch (CSV) of patients
- Risk tiering (Low / Medium / High) based on a cost-aware threshold, not a flat 0.5
- Explainable output — top contributing factors per prediction (SHAP)
- Confidence flagging for predictions the model itself is unsure about
- Evaluation dashboard — AUC, calibration, confusion matrix
- Model comparison — Logistic Regression vs Random Forest vs XGBoost

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| ML | scikit-learn, XGBoost, SHAP |
| Database | PostgreSQL (SQLite for local dev) |
| Deployment | Vercel (frontend), Render/Railway (backend) |

---

## Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)** — ~100,000 patient encounters with a
labeled readmission outcome (`<30`, `>30`, `NO`) and clinical features including
prior admissions, medications, procedures, and length of stay.

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## Repository Structure

```
readmission-predictor/
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   └── raw/
│       ├── diabetic_data.csv
│       └── IDS_mapping.csv
├── backend/
│   ├── requirements.txt
│   └── app/
│       └── main.py            # FastAPI app — health check + prediction endpoint
└── frontend/
    ├── package.json
    └── src/
        └── App.tsx             # Patient Risk Dashboard entry point
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

API runs at `http://localhost:8000`. Check `/health` to confirm it's up.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App runs at `http://localhost:5173` by default.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](./LICENSE) for
details.
