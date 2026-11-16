import { useState } from "react";

const DEFAULTS = {
  age: "51-60",
  severity_of_illness: "Moderate",
  type_of_admission: "Emergency",
  department: "gynecology",
  ward_type: "R",
  ward_facility_code: "F",
  bed_grade: 2,
  length_of_stay: 5,
  num_prior_admissions: 0,
  previous_emergency_count: 0,
  previous_urgent_count: 0,
  previous_readmission_count: 0,
  previous_hospital_days: 0,
  previous_avg_los: 0,
  previous_max_los: 0,
  days_since_previous_discharge: "",
  previous_30d_encounters: 0,
  previous_90d_encounters: 0,
  previous_365d_encounters: 0,
  available_extra_rooms: 2,
  visitors_with_patient: 2,
  admission_deposit: 4500,
  hospital_type_code: "c",
  hospital_region_code: "X",
};

const PRESETS = {
  "Frequent returner": {
    age: "71-80",
    severity_of_illness: "Extreme",
    type_of_admission: "Emergency",
    length_of_stay: 9,
    num_prior_admissions: 5,
    previous_emergency_count: 4,
    previous_readmission_count: 3,
    previous_hospital_days: 38,
    previous_avg_los: 7.6,
    previous_max_los: 14,
    days_since_previous_discharge: 12,
    previous_30d_encounters: 1,
    previous_90d_encounters: 2,
    previous_365d_encounters: 5,
  },
  "First-time, minor": {
    age: "31-40",
    severity_of_illness: "Minor",
    type_of_admission: "Urgent",
    length_of_stay: 3,
    num_prior_admissions: 0,
    visitors_with_patient: 4,
  },
};

const AGES = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"];
const NUMERIC = new Set([
  "bed_grade", "length_of_stay", "num_prior_admissions", "previous_emergency_count",
  "previous_urgent_count", "previous_readmission_count", "previous_hospital_days",
  "previous_avg_los", "previous_max_los", "previous_30d_encounters", "previous_90d_encounters",
  "previous_365d_encounters", "available_extra_rooms", "visitors_with_patient", "admission_deposit",
]);

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span className="label">{label}</span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

export default function PatientForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS);

  const set = (name) => (e) => setForm({ ...form, [name]: e.target.value });

  const select = (name, options) => (
    <select value={form[name]} onChange={set(name)}>
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );

  const num = (name, props = {}) => (
    <input type="number" min="0" step="any" value={form[name]} onChange={set(name)} {...props} />
  );

  const submit = (e) => {
    e.preventDefault();
    const payload = {};
    for (const [key, value] of Object.entries(form)) {
      if (key === "days_since_previous_discharge") {
        payload[key] = value === "" ? null : Number(value);
      } else if (NUMERIC.has(key)) {
        payload[key] = Number(value);
      } else {
        payload[key] = value;
      }
    }
    onSubmit(payload);
  };

  return (
    <form className="card" onSubmit={submit}>
      <div className="card-head">
        <h2>Patient details</h2>
        <div className="presets">
          {Object.keys(PRESETS).map((name) => (
            <button
              type="button"
              key={name}
              className="chip"
              onClick={() => setForm({ ...DEFAULTS, ...PRESETS[name] })}
            >
              {name}
            </button>
          ))}
          <button type="button" className="chip ghost" onClick={() => setForm(DEFAULTS)}>
            Reset
          </button>
        </div>
      </div>

      <fieldset>
        <legend>Current admission</legend>
        <div className="fields">
          <Field label="Age group">{select("age", AGES)}</Field>
          <Field label="Severity of illness">{select("severity_of_illness", ["Minor", "Moderate", "Extreme"])}</Field>
          <Field label="Type of admission">{select("type_of_admission", ["Emergency", "Trauma", "Urgent"])}</Field>
          <Field label="Department">
            {select("department", ["gynecology", "anesthesia", "radiotherapy", "surgery", "TB & Chest disease"])}
          </Field>
          <Field label="Length of stay (days)">{num("length_of_stay", { max: 365 })}</Field>
          <Field label="Bed grade (1–4)">{num("bed_grade", { min: 1, max: 4 })}</Field>
        </div>
      </fieldset>

      <fieldset>
        <legend>Prior hospital history</legend>
        <div className="fields">
          <Field label="Prior admissions">{num("num_prior_admissions")}</Field>
          <Field label="Prior emergency visits">{num("previous_emergency_count")}</Field>
          <Field label="Prior 30-day readmissions">{num("previous_readmission_count")}</Field>
          <Field label="Admissions in last 365 days">{num("previous_365d_encounters")}</Field>
          <Field label="Admissions in last 90 days">{num("previous_90d_encounters")}</Field>
          <Field label="Admissions in last 30 days">{num("previous_30d_encounters")}</Field>
          <Field label="Total prior hospital days">{num("previous_hospital_days")}</Field>
          <Field label="Prior average stay (days)">{num("previous_avg_los")}</Field>
          <Field label="Prior longest stay (days)">{num("previous_max_los")}</Field>
          <Field label="Days since last discharge" hint="Leave blank for a first visit">
            {num("days_since_previous_discharge")}
          </Field>
        </div>
      </fieldset>

      <details>
        <summary>Hospital and ward details</summary>
        <div className="fields">
          <Field label="Ward type">{select("ward_type", ["P", "Q", "R", "S", "T", "U"])}</Field>
          <Field label="Ward facility">{select("ward_facility_code", ["A", "B", "C", "D", "E", "F"])}</Field>
          <Field label="Hospital type">{select("hospital_type_code", ["a", "b", "c", "d", "e", "f", "g"])}</Field>
          <Field label="Hospital region">{select("hospital_region_code", ["X", "Y", "Z"])}</Field>
          <Field label="Available extra rooms">{num("available_extra_rooms")}</Field>
          <Field label="Visitors with patient">{num("visitors_with_patient")}</Field>
          <Field label="Admission deposit ($)">{num("admission_deposit")}</Field>
        </div>
      </details>

      <button className="primary" type="submit" disabled={loading}>
        {loading ? "Scoring…" : "Calculate risk"}
      </button>
    </form>
  );
}
