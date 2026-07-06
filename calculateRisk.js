function calculateRisk(marks, attendance) {
  const marksNorm = marks / 100;
  const attendanceNorm = attendance / 100;

  const risk_score = 1 - (0.6 * marksNorm + 0.4 * attendanceNorm);

  let risk_level = "LOW";
  if (risk_score >= 0.6) risk_level = "HIGH";
  else if (risk_score >= 0.35) risk_level = "MEDIUM";

  return { risk_score, risk_level };
}

module.exports = calculateRisk;
