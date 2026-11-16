import { useState } from "react";
import { scoreBatchCsv } from "../api.js";

function toCsv(patients) {
  const header = ["patient_ref", "status", "risk_probability", "risk_tier", "needs_review", "top_factors", "error"];
  const rows = patients.map((p) =>
    [
      p.patient_ref,
      p.status,
      p.risk_probability ?? "",
      p.risk_tier ?? "",
      p.needs_review ?? "",
      (p.top_factors || []).join(" | "),
      p.error ?? "",
    ]
      .map((v) => `"${String(v).replace(/"/g, '""')}"`)
      .join(",")
  );
  return [header.join(","), ...rows].join("\n");
}

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setData(null);
    try {
      setData(await scoreBatchCsv(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const download = () => {
    const blob = new Blob([toCsv(data.patients)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "ward_risk_scores.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <section className="card">
      <div className="card-head">
        <h2>Score a whole ward</h2>
        <a className="chip" href="/predict/batch/template" download>
          Download sample CSV
        </a>
      </div>
      <p className="muted">
        Upload a CSV with one patient per row. Only the columns you include are used; anything
        missing falls back to a sensible default. Patients come back ranked highest risk first, and
        a bad row never blocks the rest.
      </p>

      <div className="upload-row">
        <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)} />
        <button className="primary" onClick={run} disabled={!file || loading}>
          {loading ? "Scoring…" : "Score ward"}
        </button>
      </div>

      {error && <div className="banner error" role="alert">{error}</div>}

      {data && (
        <>
          <div className="tiles">
            <div className="tile"><b>{data.total_submitted}</b><span>submitted</span></div>
            <div className="tile high"><b>{data.tier_counts.High}</b><span>high</span></div>
            <div className="tile medium"><b>{data.tier_counts.Medium}</b><span>medium</span></div>
            <div className="tile low"><b>{data.tier_counts.Low}</b><span>low</span></div>
            <div className="tile bad"><b>{data.failed_predictions}</b><span>failed</span></div>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th><th>Patient</th><th>Risk</th><th>Tier</th><th>Top factors</th><th>Flag</th>
                </tr>
              </thead>
              <tbody>
                {data.patients.map((p, i) => (
                  <tr key={p.row_index} className={p.status === "OK" ? "" : "failed"}>
                    <td>{i + 1}</td>
                    <td>{p.patient_ref}</td>
                    {p.status === "OK" ? (
                      <>
                        <td>{(p.risk_probability * 100).toFixed(1)}%</td>
                        <td><span className={`tier ${p.risk_tier.toLowerCase()}`}>{p.risk_tier}</span></td>
                        <td>{p.top_factors.join(", ")}</td>
                        <td>{p.needs_review ? "Needs review" : ""}</td>
                      </>
                    ) : (
                      <td colSpan={4}>Could not score: {p.error}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <button className="chip" onClick={download}>Download results CSV</button>
        </>
      )}
    </section>
  );
}
