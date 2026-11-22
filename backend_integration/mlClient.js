// Node.js client for the Python ML service (app/model_api.py).
// CommonJS, Node >= 18 (global fetch). Does NOT replace calculateRisk.js:
// the rule-based score is always computed and returned as the baseline / fallback.
const calculateRisk = require("../calculateRisk");

const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";
const TIMEOUT_MS = Number(process.env.ML_API_TIMEOUT_MS || 3000);

// The ML model needs these 16 fields. The current Student schema only holds
// marks and attendance, so a student record can only be sent when the remaining
// fields have been collected. Missing values are NEVER invented here.
const REQUIRED_FEATURES = [
  "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic",
  "Semester", "Relation", "raisedhands", "VisITedResources", "AnnouncementsView",
  "Discussion", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays",
];

async function predictPerformance(student, mlFeatures) {
  // Baseline: existing rule-based score, always available.
  const rule = calculateRisk(student.marks, student.attendance);
  const baseline = { ...rule, method: "rule_based" };

  const missing = REQUIRED_FEATURES.filter((f) => mlFeatures?.[f] === undefined || mlFeatures?.[f] === null);
  if (missing.length) {
    return { ...baseline, ml: null, source: "rule_based_fallback",
             reason: "ML features unavailable for this student", missing_features: missing };
  }
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    const res = await fetch(`${ML_API_URL}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mlFeatures), signal: ctrl.signal,
    });
    clearTimeout(t);
    const body = await res.json();
    if (!res.ok) {
      return { ...baseline, ml: null, source: "rule_based_fallback",
               reason: "ML service rejected the record", details: body.detail };
    }
    return { ...baseline, ml: body, source: "ml_with_rule_baseline" };
  } catch (err) {
    return { ...baseline, ml: null, source: "rule_based_fallback",
             reason: `ML service unreachable: ${err.message}` };
  }
}

module.exports = { predictPerformance, REQUIRED_FEATURES };
