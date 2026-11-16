// Shows one prediction: probability vs threshold, tier, review flag, SHAP drivers, plain summary.

function RiskMeter({ probability, threshold }) {
  const MAX = 0.5; // scale the bar to 0–50% so differences are visible
  const pct = (v) => `${Math.min(v / MAX, 1) * 100}%`;
  return (
    <div className="meter" aria-label={`Risk ${(probability * 100).toFixed(1)} percent`}>
      <div className="meter-fill" style={{ width: pct(probability) }} />
      <div className="meter-mark" style={{ left: pct(threshold) }} title="Decision threshold">
        <span>threshold {(threshold * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

function Drivers({ factors }) {
  const maxAbs = Math.max(...factors.map((f) => Math.abs(f.shap_value)), 0.0001);
  return (
    <ul className="drivers">
      {factors.map((f) => {
        const up = f.direction === "increases_risk";
        return (
          <li key={f.feature}>
            <span className="driver-name">{f.display_name}</span>
            <span className="driver-track">
              <span
                className={`driver-bar ${up ? "up" : "down"}`}
                style={{ width: `${(Math.abs(f.shap_value) / maxAbs) * 100}%` }}
              />
            </span>
            <span className={`driver-val ${up ? "up" : "down"}`}>
              {up ? "raises" : "lowers"} risk
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export default function ResultCard({ result, error, loading }) {
  if (error) {
    return (
      <section className="card result">
        <h2>Risk assessment</h2>
        <div className="banner error" role="alert">{error}</div>
      </section>
    );
  }
  if (!result) {
    return (
      <section className="card result empty">
        <h2>Risk assessment</h2>
        <p className="muted">
          {loading
            ? "Scoring…"
            : "Fill in the patient details (or pick a sample) and press Calculate risk. The score, the factors behind it and a plain-language summary appear here."}
        </p>
      </section>
    );
  }

  const tier = result.risk_tier.toLowerCase();
  return (
    <section className="card result">
      <h2>Risk assessment</h2>

      <div className="score-row">
        <div className="score">{(result.risk_probability * 100).toFixed(1)}%</div>
        <span className={`tier ${tier}`}>{result.risk_tier} risk</span>
      </div>

      <RiskMeter probability={result.risk_probability} threshold={result.decision_threshold} />

      {result.needs_review && (
        <div className="banner warn">
          <strong>Needs review.</strong> {result.uncertainty_reason}
        </div>
      )}

      <p className="summary">{result.plain_language_summary}</p>

      <h3>Top factors behind this score</h3>
      <Drivers factors={result.factor_details} />

      <p className="muted small">Model version {result.model_version}</p>
    </section>
  );
}
