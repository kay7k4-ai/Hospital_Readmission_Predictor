// Small wrapper around fetch so every component handles errors the same way.

// FastAPI sends validation errors as a list; turn them into one readable line.
function readableError(data, fallback) {
  if (!data || !data.detail) return fallback;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`)
      .join("; ");
  }
  return fallback;
}

async function request(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new Error("Cannot reach the API. Is the backend running on port 8000?");
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) throw new Error(readableError(data, `Request failed (${res.status})`));
  return data;
}

export const scorePatient = (patient) =>
  request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patient),
  });

export const scoreBatchCsv = (file) => {
  const form = new FormData();
  form.append("file", file);
  return request("/predict/batch", { method: "POST", body: form });
};

export const getMetrics = () => request("/model/metrics");
export const getFairness = () => request("/model/fairness");
export const getModelInfo = () => request("/model/info");
export const getHealth = () => request("/health");
