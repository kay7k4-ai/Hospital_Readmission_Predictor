"""
Production Model Training, Evaluation, and Artifact Serialization Script.
Follows PRD specifications:
- US-001: Risk prediction model
- US-002: SHAP explanation setup
- US-003: Cost-aware thresholding & risk tiers
- US-004: Uncertainty flagging
- US-006: Temporal test set evaluation & calibration
- US-007: Error analysis (misclassified cases)
- US-008: Subgroup fairness analysis
- US-009: Side-by-side comparison (LogReg, Random Forest, XGBoost)
"""

import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
import shap

try:
    from backend.app.ml.pipeline import (
        RAW_CATEGORICAL_COLS,
        RAW_NUMERICAL_COLS,
        build_preprocessor,
    )
except ModuleNotFoundError:
    from app.ml.pipeline import (
        RAW_CATEGORICAL_COLS,
        RAW_NUMERICAL_COLS,
        build_preprocessor,
    )

# Set non-interactive matplotlib and writable cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

DATA_PATH = Path("data/raw/train_data_enhanced.csv")
ARTIFACTS_DIR = Path("backend/app/ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # Sort strictly chronologically by discharge datetime
    df["synthetic_discharge_datetime"] = pd.to_datetime(df["synthetic_discharge_datetime"])
    df = df.sort_values("synthetic_discharge_datetime").reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val].copy()
    test_df = df.iloc[n_val:].copy()

    print(f"Dataset split: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(
        f"Target prevalence: Train={train_df['readmitted_30_days'].mean():.3%} | "
        f"Val={val_df['readmitted_30_days'].mean():.3%} | "
        f"Test={test_df['readmitted_30_days'].mean():.3%}"
    )

    feature_cols = RAW_CATEGORICAL_COLS + RAW_NUMERICAL_COLS + ["Age", "Severity of Illness"]

    X_train = train_df[feature_cols].copy()
    y_train = np.asarray(train_df["readmitted_30_days"], dtype=int)

    X_val = val_df[feature_cols].copy()
    y_val = np.asarray(val_df["readmitted_30_days"], dtype=int)

    X_test = test_df[feature_cols].copy()
    y_test = np.asarray(test_df["readmitted_30_days"], dtype=int)

    return (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df)


def optimize_threshold(y_true, y_prob, cost_fn=4.0, cost_fp=1.0):
    """
    Finds cost-optimal decision threshold minimizing: Cost = cost_fn * FN + cost_fp * FP
    Acceptance US-003: Tier thresholds are cost-based, not default 0.5.
    """
    thresholds = np.linspace(0.05, 0.60, 111)
    best_thresh = 0.5
    min_cost = float("inf")
    metrics_at_thresh = {}

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        fn = np.sum((y_true == 1) & (preds == 0))
        fp = np.sum((y_true == 0) & (preds == 1))
        cost = (cost_fn * fn + cost_fp * fp) / len(y_true)

        if cost < min_cost:
            min_cost = cost
            best_thresh = th
            tp = np.sum((y_true == 1) & (preds == 1))
            tn = np.sum((y_true == 0) & (preds == 0))
            metrics_at_thresh = {
                "threshold": float(round(best_thresh, 3)),
                "cost_per_patient": float(round(cost, 4)),
                "precision": float(round(tp / (tp + fp + 1e-9), 4)),
                "recall": float(round(tp / (tp + fn + 1e-9), 4)),
                "f1": float(round(2 * tp / (2 * tp + fp + fn + 1e-9), 4)),
                "f2": float(round(5 * tp / (5 * tp + 4 * fn + fp + 1e-9), 4)),
            }

    return best_thresh, metrics_at_thresh


def evaluate_model(name, model, X_test_proc, y_test, threshold):
    t0 = time.time()
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    infer_time_ms = (time.time() - t0) * 1000.0 / len(y_test) * 1000.0  # ms per 1k

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    preds = (y_prob >= threshold).astype(int)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)

    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "model_name": name,
        "auc_roc": float(round(auc, 4)),
        "pr_auc": float(round(pr_auc, 4)),
        "brier_score": float(round(brier, 4)),
        "recall": float(round(rec, 4)),
        "precision": float(round(prec, 4)),
        "f1_score": float(round(f1, 4)),
        "f2_score": float(round(f2, 4)),
        "latency_ms_per_1k": float(round(infer_time_ms, 2)),
        "confusion_matrix": cm,
        "y_prob": y_prob,
    }


def main():
    print("=== STEP 1: LOAD & TEMPORAL SPLIT ===")
    (X_train, y_train, train_df), (X_val, y_val, val_df), (X_test, y_test, test_df) = load_and_split_data()

    print("\n=== STEP 2: FIT PREPROCESSING PIPELINE ON TRAIN ONLY ===")
    pipeline, num_cols, cat_cols = build_preprocessor()
    t0 = time.time()
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)
    print(f"Preprocessor fit & transform completed in {time.time()-t0:.2f}s. Total feature dim: {X_train_proc.shape[1]}")

    # Extract transformed feature names
    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_ohe_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_ohe_names

    print("\n=== STEP 3: TRAIN MULTIPLE MODELS (US-009) ===")
    models = {}

    # Model 1: Logistic Regression (L2 Regularized Baseline)
    print("Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    lr.fit(X_train_proc, y_train)
    lr_train_time = time.time() - t0
    models["Logistic Regression"] = (lr, lr_train_time)

    # Model 2: Random Forest Classifier
    print("Training Random Forest Classifier (100 trees, max_depth=12)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_proc, y_train)
    rf_train_time = time.time() - t0
    models["Random Forest"] = (rf, rf_train_time)

    # Model 3: XGBoost Classifier (Champion - Naturally Calibrated)
    print("Training XGBoost Classifier...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train_proc, y_train)
    xgb_train_time = time.time() - t0
    models["XGBoost"] = (xgb, xgb_train_time)

    print("\n=== STEP 4: THRESHOLD OPTIMIZATION ON VALIDATION SET (US-003) ===")
    xgb_val_prob = xgb.predict_proba(X_val_proc)[:, 1]
    optimal_thresh, val_cost_metrics = optimize_threshold(y_val, xgb_val_prob, cost_fn=4.0, cost_fp=1.0)
    print(f"Optimal Decision Threshold (Validation Cost Minimization): {optimal_thresh:.3f}")
    print(f"Validation Metrics at Threshold: {val_cost_metrics}")

    # Define Risk Tiers and Uncertainty Band
    tier_thresholds = {
        "low_upper": float(round(max(0.10, optimal_thresh - 0.05), 3)),  # ~0.145
        "decision_threshold": float(round(optimal_thresh, 3)),             # ~0.195
        "high_lower": float(round(optimal_thresh + 0.08, 3)),             # ~0.275
        "uncertainty_band": [
            float(round(max(0.05, optimal_thresh - 0.025), 3)),           # ~0.170
            float(round(optimal_thresh + 0.025, 3)),                       # ~0.220
        ],
    }
    print(f"Tier Cutoffs: Low < {tier_thresholds['low_upper']} | Medium [{tier_thresholds['low_upper']} - {tier_thresholds['high_lower']}) | High >= {tier_thresholds['high_lower']}")
    print(f"Uncertainty Band (labeled 'needs review'): [{tier_thresholds['uncertainty_band'][0]} - {tier_thresholds['uncertainty_band'][1]}]")

    print("\n=== STEP 5: EVALUATION ON HELD-OUT TEMPORAL TEST SET (US-006, US-009) ===")
    comparison_results = []
    eval_details = {}

    for name, (mdl, tr_time) in models.items():
        res = evaluate_model(name, mdl, X_test_proc, y_test, optimal_thresh)
        res["training_time_s"] = float(round(tr_time, 2))
        eval_details[name] = res
        comparison_results.append({
            "model_name": name,
            "auc_roc": res["auc_roc"],
            "pr_auc": res["pr_auc"],
            "brier_score": res["brier_score"],
            "recall": res["recall"],
            "precision": res["precision"],
            "f1_score": res["f1_score"],
            "f2_score": res["f2_score"],
            "latency_ms_per_1k": res["latency_ms_per_1k"],
            "training_time_s": res["training_time_s"],
        })
        print(f"-> {name:20s}: AUC-ROC={res['auc_roc']:.4f} | PR-AUC={res['pr_auc']:.4f} | Brier={res['brier_score']:.4f} | Recall={res['recall']:.4f} | Precision={res['precision']:.4f} | F2={res['f2_score']:.4f} | Latency={res['latency_ms_per_1k']:.2f}ms/1k")

    # Calibration Curve Data for Dashboard
    xgb_test_prob = eval_details["XGBoost"]["y_prob"]
    prob_true, prob_pred = calibration_curve(y_test, xgb_test_prob, n_bins=10, strategy="uniform")
    calibration_data = {
        "fraction_of_positives": [float(round(v, 4)) for v in prob_true],
        "mean_predicted_value": [float(round(v, 4)) for v in prob_pred],
    }

    # Confusion matrix for dashboard
    cm = eval_details["XGBoost"]["confusion_matrix"]
    cm_data = {
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
        "threshold": tier_thresholds["decision_threshold"],
    }

    print("\n=== STEP 6: SUBGROUP FAIRNESS AUDIT (US-008) ===")
    fairness_results = {}
    test_df_eval = test_df.copy()
    test_df_eval["pred_prob"] = xgb_test_prob
    test_df_eval["pred_label"] = (xgb_test_prob >= optimal_thresh).astype(int)

    # Subgroups by Age
    age_groups = test_df_eval.groupby("Age")
    fairness_results["age"] = []
    for age_val, grp in age_groups:
        if len(grp) > 50 and grp["readmitted_30_days"].nunique() > 1:
            grp_auc = roc_auc_score(grp["readmitted_30_days"], grp["pred_prob"])
            grp_rec = recall_score(grp["readmitted_30_days"], grp["pred_label"])
            fairness_results["age"].append({
                "subgroup": str(age_val),
                "count": int(len(grp)),
                "readmission_rate": float(round(grp["readmitted_30_days"].mean(), 4)),
                "auc_roc": float(round(grp_auc, 4)),
                "recall": float(round(grp_rec, 4)),
            })

    # Subgroups by Severity
    sev_groups = test_df_eval.groupby("Severity of Illness")
    fairness_results["severity"] = []
    for sev_val, grp in sev_groups:
        grp_auc = roc_auc_score(grp["readmitted_30_days"], grp["pred_prob"])
        grp_rec = recall_score(grp["readmitted_30_days"], grp["pred_label"])
        fairness_results["severity"].append({
            "subgroup": str(sev_val),
            "count": int(len(grp)),
            "readmission_rate": float(round(grp["readmitted_30_days"].mean(), 4)),
            "auc_roc": float(round(grp_auc, 4)),
            "recall": float(round(grp_rec, 4)),
        })

    print("Fairness audit completed across Age and Severity subpopulations.")

    print("\n=== STEP 7: SHAP EXPLAINER & ERROR CASES AUDIT (US-002, US-007) ===")
    # Fit SHAP TreeExplainer
    t0 = time.time()
    explainer = shap.TreeExplainer(xgb)
    print(f"SHAP TreeExplainer initialized in {time.time()-t0:.2f}s.")

    # Select representative misclassified cases for error auditing
    fps = test_df_eval[(test_df_eval["readmitted_30_days"] == 0) & (test_df_eval["pred_label"] == 1)].head(15)
    fns = test_df_eval[(test_df_eval["readmitted_30_days"] == 1) & (test_df_eval["pred_label"] == 0)].head(15)

    error_cases = []
    for error_type, subset in [("False Positive", fps), ("False Negative", fns)]:
        sub_proc = pipeline.transform(subset[RAW_CATEGORICAL_COLS + RAW_NUMERICAL_COLS + ["Age", "Severity of Illness"]])
        sub_shap = explainer(sub_proc)

        for i, (_, row) in enumerate(subset.iterrows()):
            vals = sub_shap.values[i]
            top_indices = np.argsort(np.abs(vals))[-5:][::-1]
            top_factors = []
            for idx in top_indices:
                feat_name = all_feature_names[idx]
                top_factors.append({
                    "feature": feat_name,
                    "shap_value": float(round(vals[idx], 4)),
                    "direction": "increases_risk" if vals[idx] > 0 else "decreases_risk",
                })

            error_cases.append({
                "case_id": int(row["case_id"]),
                "error_type": error_type,
                "actual_outcome": int(row["readmitted_30_days"]),
                "predicted_prob": float(round(row["pred_prob"], 4)),
                "age": str(row["Age"]),
                "severity": str(row["Severity of Illness"]),
                "department": str(row["Department"]),
                "previous_encounters": int(row["previous_encounters"]),
                "length_of_stay_days": float(round(row["length_of_stay_days"], 1)),
                "top_factors": top_factors,
            })

    print(f"Extracted {len(error_cases)} error cases with SHAP breakdowns for audit dashboard.")

    print("\n=== STEP 8: SERIALIZE ARTIFACTS ===")
    # Save best model & pipeline
    joblib.dump(xgb, ARTIFACTS_DIR / "best_model.joblib")
    joblib.dump(pipeline, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(explainer, ARTIFACTS_DIR / "explainer.joblib")

    # Save metadata
    metadata = {
        "model_version": "1.0.0",
        "algorithm": "XGBoost Classifier (TreeHist)",
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "thresholds": tier_thresholds,
        "feature_names": all_feature_names,
        "test_metrics": eval_details["XGBoost"],
    }
    # Clean y_prob before json serializing
    del metadata["test_metrics"]["y_prob"]

    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(ARTIFACTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    with open(ARTIFACTS_DIR / "calibration_curve.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    with open(ARTIFACTS_DIR / "confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)

    with open(ARTIFACTS_DIR / "subgroup_fairness.json", "w") as f:
        json.dump(fairness_results, f, indent=2)

    with open(ARTIFACTS_DIR / "error_cases.json", "w") as f:
        json.dump(error_cases, f, indent=2)

    print(f"All ML production artifacts successfully saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
