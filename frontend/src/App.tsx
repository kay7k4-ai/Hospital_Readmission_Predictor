import { useState } from "react";

interface RiskResponse {
  risk_probability: number;
  risk_tier: string;
  top_factors: string[];
}

function App() {
  const [result, setResult] = useState<RiskResponse | null>(null);

  // Placeholder — wired to backend /predict endpoint (see US-001).
  const handleScore = () => {
    setResult({
      risk_probability: 0.42,
      risk_tier: "Medium",
      top_factors: ["num_prior_admissions", "length_of_stay"],
    });
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Patient Risk Dashboard</h1>
      <p>Hospital 30-Day Readmission Predictor — scaffold in progress.</p>
      <button onClick={handleScore}>Score Sample Patient</button>
      {result && (
        <div style={{ marginTop: "1rem" }}>
          <p>Risk Tier: {result.risk_tier}</p>
          <p>Probability: {result.risk_probability}</p>
          <p>Top factors: {result.top_factors.join(", ")}</p>
        </div>
      )}
    </div>
  );
}

export default App;
