import { useEffect, useState } from "react";
import { getFairness, getMetrics, getModelInfo } from "../api.js";

const pct = (v) => `${(v * 100).toFixed(1)}%`;

function Confusion({ cm }) {
  const total = cm.true_negatives + cm.false_positives + cm.false_negatives + cm.true_positives;
  const cell = (label, value, cls) => (
    <div className={`cm-cell ${cls}`}>
      <b>{value.toLocaleString()}</b>
      <span>{label}</span>
      <small>{pct(value / total)}</small>
    </div>
  );
  return (
    <div className="cm">
      {cell("Correctly cleared", cm.true_negatives, "good")}
      {cell("Flagged, not readmitted", cm.false_positives, "meh")}
      {cell("Missed readmission", cm.false_negatives, "bad")}
      {cell("Correctly flagged", cm.true_positives, "good")}
    </div>
  );
}

function SubgroupTable({ title, rows }) {
  return (
    <div>
      <h3>{title}</h3>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Group</th><th>Patients</th><th>Readmit rate</th><th>AUC</th><th>Recall</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.subgroup}>
                <td>{r.subgroup}</td>
                <td>{r.count.toLocaleString()}</td>
                <td>{pct(r.readmission_rate)}</td>
                <td>{r.auc_roc?.toFixed(3) ?? "–"}</td>
                <td>{pct(r.recall)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function ModelInsights() {
  const [metrics, setMetrics] = useState(null);
  const [fairness, setFairness] = useState(null);
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([getMetrics(), getFairness(), getModelInfo()])
      .then(([m, f, i]) => {
        setMetrics(m);
        setFairness(f);
        setInfo(i);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="banner error" role="alert">{error}</div>;
  if (!metrics) return <p className="muted">Loading model details…</p>;

  const t = metrics.thresholds;
  return (
    <div className="stack">
      <section className="card">
        <h2>Model comparison</h2>
        <p className="muted">
          Three models were trained on the same split. Logistic Regression and Random Forest show
          very high recall only because they flag almost every patient; compare them on AUC and Brier
          score instead.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>AUC</th><th>PR-AUC</th><th>Brier (lower is better)</th><th>Recall</th><th>Precision</th></tr>
            </thead>
            <tbody>
              {metrics.comparison.map((m) => (
                <tr key={m.model_name} className={m.model_name === info?.test_metrics?.model_name ? "best" : ""}>
                  <td>{m.model_name}{m.model_name === info?.test_metrics?.model_name ? " (deployed)" : ""}</td>
                  <td>{m.auc_roc.toFixed(3)}</td>
                  <td>{m.pr_auc.toFixed(3)}</td>
                  <td>{m.brier_score.toFixed(3)}</td>
                  <td>{pct(m.recall)}</td>
                  <td>{pct(m.precision)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid two">
        <section className="card">
          <h2>Confusion matrix (test set)</h2>
          <Confusion cm={metrics.confusion_matrix} />
          <p className="muted small">Decision threshold {t.decision_threshold}</p>
        </section>

        <section className="card">
          <h2>Risk tiers</h2>
          <ul className="plain">
            <li><span className="tier low">Low</span> below {pct(t.low_upper)}</li>
            <li><span className="tier medium">Medium</span> {pct(t.low_upper)} to {pct(t.high_lower)}</li>
            <li><span className="tier high">High</span> above {pct(t.high_lower)}</li>
          </ul>
          <p className="muted small">
            Scores between {pct(t.uncertainty_band[0])} and {pct(t.uncertainty_band[1])} are marked
            "needs review" because the model is least certain there.
          </p>
        </section>
      </div>

      <section className="card">
        <h2>Fairness by subgroup</h2>
        <p className="muted">Does the model perform similarly across patient groups?</p>
        <div className="grid two">
          {fairness.age && <SubgroupTable title="By age group" rows={fairness.age} />}
          {fairness.severity && <SubgroupTable title="By severity of illness" rows={fairness.severity} />}
        </div>
      </section>
    </div>
  );
}
