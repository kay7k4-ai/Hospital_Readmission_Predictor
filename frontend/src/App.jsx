import { useEffect, useState } from "react";
import PatientForm from "./components/PatientForm.jsx";
import ResultCard from "./components/ResultCard.jsx";
import BatchUpload from "./components/BatchUpload.jsx";
import ModelInsights from "./components/ModelInsights.jsx";
import { getHealth, scorePatient } from "./api.js";

const TABS = [
  { id: "single", label: "Score a patient" },
  { id: "batch", label: "Ward batch" },
  { id: "model", label: "Model insights" },
];

function App() {
  const [tab, setTab] = useState("single");
  const [apiUp, setApiUp] = useState(null); // null = checking
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getHealth()
      .then((h) => setApiUp(Boolean(h.model_loaded)))
      .catch(() => setApiUp(false));
  }, []);

  const handleScore = async (patient) => {
    setLoading(true);
    setError("");
    try {
      setResult(await scorePatient(patient));
    } catch (e) {
      setResult(null);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Readmission Risk Dashboard</h1>
          <p className="subtitle">30-day hospital readmission risk for care coordinators</p>
        </div>
        <span className={`status ${apiUp === null ? "wait" : apiUp ? "ok" : "down"}`}>
          {apiUp === null ? "Checking API…" : apiUp ? "Model online" : "API offline"}
        </span>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "tab active" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "single" && (
          <div className="grid two">
            <PatientForm onSubmit={handleScore} loading={loading} />
            <ResultCard result={result} error={error} loading={loading} />
          </div>
        )}
        {tab === "batch" && <BatchUpload />}
        {tab === "model" && <ModelInsights />}
      </main>

      <footer className="footnote">
        Demo data note: the training dataset uses synthetic admission dates and a derived
        readmission label, so scores demonstrate the pipeline, not clinical accuracy. Output
        supports, and never replaces, clinical judgment.
      </footer>
    </div>
  );
}

export default App;
