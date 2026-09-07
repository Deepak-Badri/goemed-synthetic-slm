import { useState } from "react";
import "./App.css";

// ── Constants ──────────────────────────────────────────────────────────────
const API_URL = "http://3.218.72.111:8000";

const LABEL_META = {
  hypertension:       { display: "Hypertension",              color: "#D85A30", icon: "🫀", why: "Driven by systolic/diastolic BP thresholds (ACC/AHA 2017: SBP ≥ 130 or DBP ≥ 80) plus BP medication status." },
  diabetes:           { display: "Type 2 Diabetes",           color: "#BA7517", icon: "🩸", why: "ADA diagnostic criteria: HbA1c ≥ 6.5% or fasting glucose ≥ 126 mg/dL. FINDRISC score also contributes via BMI, waist, family history, and activity level." },
  cvd_risk:           { display: "Cardiovascular Disease",    color: "#2E6DA4", icon: "💓", why: "Average of Framingham Risk Score (Circulation 2008) and ACC/AHA ASCVD Pooled Cohort Equations (JACC 2013). Key inputs: age, sex, cholesterol, BP, smoking status." },
  ckd:                { display: "Chronic Kidney Disease",    color: "#7F77DD", icon: "🔬", why: "2021 race-free CKD-EPI equation (NEJM 2021). eGFR < 60 mL/min/1.73m² = CKD. Primary inputs: serum creatinine, age, sex." },
  osa:                { display: "Sleep Apnea (OSA)",         color: "#1D9E75", icon: "😴", why: "STOP-BANG score ≥ 3 (Anesthesiology 2008). Inputs: snoring, fatigue, observed apnea, BP, BMI > 35, age > 50, neck circumference, male sex." },
  depression:         { display: "Depression / Anxiety",      color: "#639922", icon: "🧠", why: "Modeled from PHQ-9 risk factors (JGIM 2001): female sex, younger age, low physical activity, and chronic disease burden." },
  copd:               { display: "COPD / Asthma",             color: "#595855", icon: "🫁", why: "Driven by smoking pack-years and age. Risk escalates sharply above 20 pack-years. Based on GOLD COPD guidelines." },
  metabolic_syndrome: { display: "Metabolic Syndrome",        color: "#BA7517", icon: "⚖️",  why: "ATP-III criteria (3 of 5): elevated waist, elevated BP, elevated fasting glucose, low HDL, elevated triglycerides proxy. All inputs from patient profile." },
  hypothyroidism:     { display: "Hypothyroidism",            color: "#2E6DA4", icon: "🦋", why: "Prevalence-based model: female sex and age > 60 are the primary risk markers. NHANES prevalence: ~4.6% overall, ~8.6% in older women." },
  prediabetes:        { display: "Prediabetes",               color: "#D85A30", icon: "📊", why: "ADA criteria: HbA1c 5.7–6.4% or fasting glucose 100–125 mg/dL. Does not require formal diabetes diagnosis." },
  colorectal_cancer:  { display: "Colorectal Cancer Risk",    color: "#7F77DD", icon: "🎗️",  why: "SEER 2018-2022 age/sex/race-specific 10-year incidence rates. Risk modifiers: family history (+50%), smoking (+30%), obesity (+20%), low activity (+15%)." },
};

const EXAMPLE_PATIENTS = [
  {
    label: "High-risk — 58yo Black male",
    data: { age: 58, sex: "M", race: "nh_black", bmi: 31.4, sbp: 148, dbp: 92, total_cholesterol: 210, hdl_cholesterol: 38, hba1c: 7.2, fasting_glucose: 142, smoker: false, bp_treated: true, family_history_dm: true, family_history_cvd: true, creatinine: 1.3 }
  },
  {
    label: "Low-risk — 28yo White female",
    data: { age: 28, sex: "F", race: "nh_white", bmi: 22.5, sbp: 110, dbp: 70, total_cholesterol: 165, hdl_cholesterol: 65, hba1c: 5.1, smoker: false, physical_activity_low: false }
  },
  {
    label: "Moderate-risk — 45yo Hispanic female",
    data: { age: 45, sex: "F", race: "hispanic", bmi: 28.9, sbp: 128, dbp: 82, total_cholesterol: 195, hdl_cholesterol: 48, hba1c: 5.9, fasting_glucose: 108, smoker: false, physical_activity_low: true, family_history_dm: true }
  },
  {
    label: "Elderly high-risk — 72yo male, smoker",
    data: { age: 72, sex: "M", race: "nh_black", bmi: 34.2, sbp: 162, dbp: 96, total_cholesterol: 240, hdl_cholesterol: 32, hba1c: 8.1, fasting_glucose: 188, smoker: true, pack_years: 35, bp_treated: true, family_history_dm: true, family_history_cvd: true, creatinine: 1.8 }
  },
];

const DEFAULT_JSON = JSON.stringify({
  age: 58, sex: "M", race: "nh_black",
  bmi: 31.4, sbp: 148, dbp: 92,
  total_cholesterol: 210, hdl_cholesterol: 38,
  hba1c: 7.2, fasting_glucose: 142,
  smoker: false, bp_treated: true,
  family_history_dm: true, family_history_cvd: true,
  creatinine: 1.3
}, null, 2);

// ── Risk bar component ─────────────────────────────────────────────────────
function RiskBar({ label, data, expanded, onToggle }) {
  const meta  = LABEL_META[label] || {};
  const prob  = data.probability;
  const level = data.risk_level;
  const pct   = Math.round(prob * 100);

  const levelColor = level === "high" ? "#D85A30" : level === "moderate" ? "#BA7517" : "#1D9E75";
  const levelBg    = level === "high" ? "#FAECE7" : level === "moderate" ? "#FAEEDA" : "#E1F5EE";

  return (
    <div className="risk-bar-wrap" onClick={onToggle}>
      <div className="risk-bar-header">
        <div className="risk-bar-left">
          <span className="risk-icon">{meta.icon}</span>
          <span className="risk-label">{meta.display || label}</span>
        </div>
        <div className="risk-bar-right">
          <span className="risk-pct" style={{ color: levelColor }}>{pct}%</span>
          <span className="risk-badge" style={{ color: levelColor, background: levelBg }}>{level.toUpperCase()}</span>
          <span className="risk-chevron" style={{ transform: expanded ? "rotate(180deg)" : "none" }}>▾</span>
        </div>
      </div>

      <div className="risk-track">
        <div
          className="risk-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${meta.color}99, ${meta.color})`,
            transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
          }}
        />
        <div className="risk-thresh" style={{ left: "50%" }} title="50% threshold" />
        <div className="risk-thresh" style={{ left: "70%" }} title="70% threshold" />
      </div>

      {expanded && (
        <div className="risk-reasoning">
          <div className="reasoning-label">Clinical Reasoning</div>
          <p>{meta.why}</p>
          <div className="reasoning-formula">
            <span className="formula-key">Raw probability:</span>
            <span className="formula-val">{prob.toFixed(4)}</span>
            <span className="formula-key" style={{ marginLeft: 16 }}>Risk level:</span>
            <span className="formula-val" style={{ color: levelColor }}>{level} {pct >= 70 ? "(≥ 70%)" : pct >= 40 ? "(40–69%)" : "(< 40%)"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Summary strip ─────────────────────────────────────────────────────────
function SummaryStrip({ response }) {
  const counts = Object.values(response.predictions).reduce((acc, p) => {
    acc[p.risk_level] = (acc[p.risk_level] || 0) + 1;
    return acc;
  }, {});
  const summaryColor = response.risk_summary === "high" ? "#D85A30"
    : response.risk_summary === "moderate" ? "#BA7517" : "#1D9E75";

  return (
    <div className="summary-strip">
      <div className="summary-overall">
        <span className="summary-label">Overall Risk</span>
        <span className="summary-value" style={{ color: summaryColor }}>
          {response.risk_summary.toUpperCase()}
        </span>
      </div>
      <div className="summary-counts">
        {["high","moderate","low"].map(level => (
          <div key={level} className="summary-count">
            <span className="count-num" style={{ color: level === "high" ? "#D85A30" : level === "moderate" ? "#BA7517" : "#1D9E75" }}>
              {counts[level] || 0}
            </span>
            <span className="count-label">{level}</span>
          </div>
        ))}
      </div>
      <div className="summary-factors">
        <span className="summary-label">Top Risks</span>
        <span className="summary-factors-list">
          {response.top_risk_factors.slice(0,3).join(" · ")}
        </span>
      </div>
      <div className="summary-timing">
        <span className="summary-label">Inference</span>
        <span className="summary-value">{Math.round(response.inference_time_ms)}ms</span>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [jsonInput, setJsonInput]     = useState(DEFAULT_JSON);
  const [response, setResponse]       = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [expanded, setExpanded]       = useState({});
  const [tab, setTab]                 = useState("bars"); // bars | json
  const [jsonError, setJsonError]     = useState(null);
  const [apiMode, setApiMode]         = useState(true); // true = call API, false = mock

  // Sort predictions by probability descending
  const sortedPredictions = response
    ? Object.entries(response.predictions).sort((a, b) => b[1].probability - a[1].probability)
    : [];

  function validateJson(text) {
    try { JSON.parse(text); setJsonError(null); return true; }
    catch (e) { setJsonError(e.message); return false; }
  }

  function handleJsonChange(e) {
    setJsonInput(e.target.value);
    validateJson(e.target.value);
  }

  function loadExample(ex) {
    const str = JSON.stringify(ex.data, null, 2);
    setJsonInput(str);
    setJsonError(null);
    setResponse(null);
    setError(null);
  }

  function toggleExpand(label) {
    setExpanded(prev => ({ ...prev, [label]: !prev[label] }));
  }

  function expandAll() {
    const all = {};
    sortedPredictions.forEach(([l]) => { all[l] = true; });
    setExpanded(all);
  }

  function collapseAll() { setExpanded({}); }

  async function runPrediction() {
    if (!validateJson(jsonInput)) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    setExpanded({});

    try {
      const payload = JSON.parse(jsonInput);

      if (!apiMode) {
        // Mock mode — simulate API response
        await new Promise(r => setTimeout(r, 800));
        const mock = {
          predictions: {},
          risk_summary: "moderate",
          top_risk_factors: ["Hypertension", "Type 2 Diabetes", "CVD Risk"],
          model_version: "bio-clinicalbert-goemed-v1 (mock)",
          inference_time_ms: 712,
          disclaimer: "Mock response — start the API for real predictions.",
        };
        Object.keys(LABEL_META).forEach(label => {
          const prob = Math.random();
          mock.predictions[label] = {
            label,
            display_name: LABEL_META[label].display,
            probability: parseFloat(prob.toFixed(4)),
            risk_level: prob >= 0.70 ? "high" : prob >= 0.40 ? "moderate" : "low",
          };
        });
        setResponse(mock);
        setTab("bars");
        setLoading(false);
        return;
      }

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API error ${res.status}: ${text}`);
      }

      const data = await res.json();
      setResponse(data);
      setTab("bars");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-mark">G</span>
            <div>
              <div className="logo-name">GoEMed</div>
              <div className="logo-sub">Health Risk Intelligence</div>
            </div>
          </div>
          <div className="header-right">
            <div className="api-toggle">
              <span className="toggle-label">Mode:</span>
              <button
                className={`toggle-btn ${apiMode ? "active" : ""}`}
                onClick={() => setApiMode(true)}
              >Live API</button>
              <button
                className={`toggle-btn ${!apiMode ? "active" : ""}`}
                onClick={() => setApiMode(false)}
              >Mock</button>
            </div>
            <div className={`api-status ${apiMode ? "live" : "mock"}`}>
              <span className="status-dot" />
              {apiMode ? "api.goemed.local:8000" : "mock mode"}
            </div>
          </div>
        </div>
      </header>

      <main className="main">

        {/* ── Left panel: Input ── */}
        <section className="panel panel-input">
          <div className="panel-header">
            <h2 className="panel-title">Patient Profile</h2>
            <span className="panel-hint">JSON input</span>
          </div>

          {/* Example presets */}
          <div className="examples">
            <div className="examples-label">Load example</div>
            <div className="examples-grid">
              {EXAMPLE_PATIENTS.map((ex, i) => (
                <button key={i} className="example-btn" onClick={() => loadExample(ex)}>
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          {/* Schema quick-ref */}
          <div className="schema-ref">
            <div className="schema-title">Required fields</div>
            <div className="schema-fields">
              {["age (int)", "sex (M|F)"].map(f => (
                <span key={f} className="schema-pill required">{f}</span>
              ))}
            </div>
            <div className="schema-title" style={{ marginTop: 6 }}>Optional fields</div>
            <div className="schema-fields">
              {["race", "bmi", "sbp", "dbp", "total_cholesterol", "hdl_cholesterol",
                "hba1c", "fasting_glucose", "creatinine", "smoker (bool)",
                "pack_years", "bp_treated (bool)", "physical_activity_low (bool)",
                "family_history_dm (bool)", "family_history_cvd (bool)",
                "snoring (bool)", "alcohol_use (bool)", "waist_cm"
              ].map(f => (
                <span key={f} className="schema-pill">{f}</span>
              ))}
            </div>
            <div className="schema-title" style={{ marginTop: 6 }}>Race values</div>
            <div className="schema-fields">
              {["nh_white","nh_black","hispanic","nh_asian","other_multiracial"].map(r => (
                <span key={r} className="schema-pill race">{r}</span>
              ))}
            </div>
          </div>

          {/* JSON editor */}
          <div className="editor-wrap">
            <textarea
              className={`editor ${jsonError ? "editor-error" : ""}`}
              value={jsonInput}
              onChange={handleJsonChange}
              spellCheck={false}
              rows={22}
            />
            {jsonError && <div className="editor-error-msg">⚠ {jsonError}</div>}
          </div>

          <button
            className="predict-btn"
            onClick={runPrediction}
            disabled={loading || !!jsonError}
          >
            {loading ? (
              <><span className="spinner" /> Analyzing patient profile…</>
            ) : (
              <><span className="predict-icon">⚡</span> Run Prediction</>
            )}
          </button>

          {error && (
            <div className="error-box">
              <div className="error-title">Prediction failed</div>
              <div className="error-msg">{error}</div>
              <div className="error-hint">
                Make sure the API is running: <code>aws ecs update-service --cluster goemed-inference-cluster --service goemed-slm-service --desired-count 1 --region us-east-1</code>
                <br/>Or switch to <strong>Mock</strong> mode to test the UI.
              </div>
            </div>
          )}
        </section>

        {/* ── Right panel: Output ── */}
        <section className="panel panel-output">
          {!response && !loading && (
            <div className="empty-state">
              <div className="empty-icon">🏥</div>
              <div className="empty-title">No prediction yet</div>
              <div className="empty-body">
                Enter a patient profile on the left and click <strong>Run Prediction</strong> to see health risk scores for 11 conditions.
              </div>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="loading-ring" />
              <div className="empty-title" style={{ marginTop: 20 }}>Running Bio_ClinicalBERT…</div>
              <div className="empty-body">Downloading model from S3, tokenizing prompt, running inference.</div>
            </div>
          )}

          {response && (
            <>
              <SummaryStrip response={response} />

              <div className="output-tabs">
                <button className={`tab ${tab === "bars" ? "tab-active" : ""}`} onClick={() => setTab("bars")}>
                  Risk Bars
                </button>
                <button className={`tab ${tab === "json" ? "tab-active" : ""}`} onClick={() => setTab("json")}>
                  Raw JSON
                </button>
                {tab === "bars" && (
                  <div className="expand-controls">
                    <button className="expand-btn" onClick={expandAll}>Expand all</button>
                    <button className="expand-btn" onClick={collapseAll}>Collapse all</button>
                  </div>
                )}
              </div>

              {tab === "bars" && (
                <div className="bars-list">
                  <div className="bars-legend">
                    <div className="legend-item"><span className="legend-line" style={{ left: "50%" }} />50%</div>
                    <div className="legend-item"><span className="legend-line" style={{ left: "70%" }} />70%</div>
                  </div>
                  {sortedPredictions.map(([label, data]) => (
                    <RiskBar
                      key={label}
                      label={label}
                      data={data}
                      expanded={!!expanded[label]}
                      onToggle={() => toggleExpand(label)}
                    />
                  ))}
                  <div className="disclaimer">
                    {response.disclaimer}
                  </div>
                </div>
              )}

              {tab === "json" && (
                <div className="json-wrap">
                  <button
                    className="copy-btn"
                    onClick={() => navigator.clipboard.writeText(JSON.stringify(response, null, 2))}
                  >Copy JSON</button>
                  <pre className="json-output">{JSON.stringify(response, null, 2)}</pre>
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}
