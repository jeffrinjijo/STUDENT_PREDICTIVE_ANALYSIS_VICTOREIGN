// Run: node tests/test_mlClient.js   (no network needed - uses a stub server)
const http = require("http");
const assert = require("assert");
process.env.ML_API_URL = "http://127.0.0.1:18765";
process.env.ML_API_TIMEOUT_MS = "1000";
const { predictPerformance, REQUIRED_FEATURES } = require("../backend_integration/mlClient");

const feats = Object.fromEntries(REQUIRED_FEATURES.map((f) => [f, "x"]));
const server = http.createServer((req, res) => {
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ predicted_class: "L", predicted_label: "Low", probabilities: { L: 0.8, M: 0.15, H: 0.05 } }));
});

(async () => {
  // 1. missing features -> fallback, never fabricated
  let r = await predictPerformance({ marks: 40, attendance: 50 }, { gender: "M" });
  assert.strictEqual(r.source, "rule_based_fallback"); assert.strictEqual(r.missing_features.length, 15);
  assert.ok(Math.abs(r.risk_score - 0.56) < 1e-9 && r.risk_level === "MEDIUM"); // rule-based result unchanged
  // 2. service down -> fallback
  r = await predictPerformance({ marks: 90, attendance: 95 }, feats);
  assert.strictEqual(r.source, "rule_based_fallback"); assert.strictEqual(r.risk_level, "LOW");
  // 3. service up -> ml result + rule baseline both returned
  await new Promise((ok) => server.listen(18765, ok));
  r = await predictPerformance({ marks: 40, attendance: 50 }, feats);
  assert.strictEqual(r.source, "ml_with_rule_baseline"); assert.strictEqual(r.ml.predicted_class, "L"); assert.strictEqual(r.method, "rule_based");
  server.close(); console.log("mlClient: 3/3 checks passed");
})().catch((e) => { console.error(e); process.exit(1); });
