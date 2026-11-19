/* Checkpoint 1 test-set MAE and F1, in the published November 11 leaderboard order. */
const checkpointSubmissions = [
  // 1_1_0 through 1_1_9
  [171.93839, 0.098923], [96.95125, 0.445033], [97.31445, 0.445033], [99.00964, 0.451696],
  [107.23870, 0.001383], [107.16688, 0.002766], [107.21171, 0.002766], [107.03598, 0.000000],
  [196.77396, 0.096838], [128.55843, 0.513989],
  // 1_5_1 through 1_5_10
  [107.03598, 0.000000], [107.93998, 0.091603], [107.03665, 0.160920], [200.36118, 0.091603],
  [107.03682, 0.238115], [107.04278, 0.091603], [107.16908, 0.238115], [101.00935, 0.425721],
  [103.44257, 0.513067], [109.60888, 0.091603],
  // 1_2_1 through 1_2_10
  [107.31277, 0.014785], [107.11688, 0.000000], [107.03598, 0.000000], [109.31995, 0.447984],
  [110.15569, 0.432592], [107.47890, 0.189702], [103.40172, 0.161461], [102.74353, 0.284698],
  [106.13918, 0.047872], [103.77985, 0.271047],
  // 1_4_1 through 1_4_10
  [105.18871, 0.160852], [113.95814, 0.458621], [106.23223, 0.213142], [105.54404, 0.150628],
  [106.57152, 0.102144], [104.89737, 0.160852], [121.65900, 0.458621], [105.52168, 0.213142],
  [105.40175, 0.150628], [106.43271, 0.102144],
  // 1_3_1 through 1_3_10
  [107.03598, 0.000000], [107.26374, 0.006873], [107.13040, 0.009622], [107.26374, 0.006873],
  [107.32549, 0.005498], [107.06932, 0.001386], [107.20374, 0.019048], [107.10265, 0.001383],
  [107.15598, 0.004141], [107.09215, 0.019048],
];

function hundredths(value) {
  const lower = Math.floor(value * 100);
  // Compare in the original units so a decimal midpoint like 1.005 rounds up.
  return value >= (lower + 0.5) / 100 ? lower + 1 : lower;
}

function checkpointRanks(values) {
  const sorted = values.sort((a,b) => a-b);
  const ranks = [];
  for (let start = 0; start < sorted.length;) {
    let end = start + 1;
    while (end < sorted.length && sorted[end] === sorted[start]) end++;
    // Tied submissions occupy places start + 1 through end on the published scale.
    ranks.push({value: sorted[start], points: (start + 1 + end) / 2});
    start = end;
  }
  return ranks;
}

// Negating MAE orders both metrics from worst to best.
const checkpointMae = checkpointRanks(checkpointSubmissions.map(([mae]) => -mae));
const checkpointF1 = checkpointRanks(checkpointSubmissions.map(([,f1]) => f1));

function interpolatedPoints(value, reference) {
  let low = 0, high = reference.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (reference[middle].value < value) low = middle + 1;
    else high = middle;
  }
  if (low === reference.length) return checkpointSubmissions.length;
  const higher = reference[low];
  if (value === higher.value) return higher.points;
  if (low === 0) return 1;
  const lower = reference[low - 1];
  const fraction = (value - lower.value) / (higher.value - lower.value);
  return lower.points + fraction * (higher.points - lower.points);
}

function checkpointPoints(mae, f1) {
  const maeHundredths = hundredths(interpolatedPoints(-mae, checkpointMae));
  const f1Hundredths = hundredths(interpolatedPoints(f1, checkpointF1));
  return {
    mae_points: maeHundredths / 100,
    f1_points: f1Hundredths / 100,
    aggregate: (maeHundredths + f1Hundredths) / 100,
  };
}
