/* Continuous 0-100 aggregate points from mean holdout MAE and mean F1. */
const MAX_METRIC_POINTS = 50;
// Linear bounds leave substantial headroom beyond top holdout runs (~91.6 MAE, ~0.523 F1)
// so strong candidates remain clearly separated rather than saturating at 100 points.
const MAE_ZERO_SCORE = 120;
const MAE_FULL_SCORE = 70;
const F1_ZERO_SCORE = 0;
const F1_FULL_SCORE = 0.8;

function hundredths(value) {
  const lower = Math.floor(value * 100);
  // Compare in the original units so a decimal midpoint like 1.005 rounds up.
  return value >= (lower + 0.5) / 100 ? lower + 1 : lower;
}

function linearMetricPoints(value, zeroScore, fullScore) {
  const ratio = (value - zeroScore) / (fullScore - zeroScore);
  const clamped = Math.max(0, Math.min(1, ratio));
  return clamped * MAX_METRIC_POINTS;
}

function checkpointPoints(mae, f1) {
  if (!Number.isFinite(mae) || !Number.isFinite(f1)) {
    return {mae_points: null, f1_points: null, aggregate: null};
  }
  const maeHundredths = hundredths(linearMetricPoints(mae, MAE_ZERO_SCORE, MAE_FULL_SCORE));
  const f1Hundredths = hundredths(linearMetricPoints(f1, F1_ZERO_SCORE, F1_FULL_SCORE));
  return {
    mae_points: maeHundredths / 100,
    f1_points: f1Hundredths / 100,
    aggregate: (maeHundredths + f1Hundredths) / 100,
  };
}
