/* Inspect arithmetic seed means and export an existing training configuration. */
const study = JSON.parse(document.getElementById('study-data').textContent);
const byId = id => document.getElementById(id);
const number = value => value.toFixed(4);
const percent = value => (100 * value).toFixed(2) + '%';
let visible = [];
function render() {
  const minimumPrecision = Number(byId('precision').value) / 100;
  const minimumRecall = Number(byId('recall').value) / 100;
  const maximumMae = byId('mae').value === '' ? Infinity : Number(byId('mae').value);
  let rows = study.records.filter(row => row.precision >= minimumPrecision && row.recall >= minimumRecall
    && row.mae <= maximumMae && (!byId('every-seed').checked
      || (row.worst_precision >= minimumPrecision && row.worst_recall >= minimumRecall
        && row.worst_mae <= maximumMae)));
  const order = byId('order').value;
  rows.sort((a, b) => order === 'mae' ? a.mae - b.mae
    : b[order] - a[order] || a.mae - b.mae);
  if (byId('one-per-classifier').checked) {
    const seen = new Set();
    rows = rows.filter(row => !seen.has(row.candidate) && seen.add(row.candidate));
  }
  visible = rows;
  byId('count').textContent = rows.length + ' retained configurations match. All metrics average seeds 41, 42, and 43.';
  byId('rows').replaceChildren();
  for (const row of rows) {
    const tr = document.createElement('tr');
    const values = [row.candidate_ids.join(', '), row.seeds.join(', '), row.classifier.family,
      number(row.mae), number(row.baseline_mae), number(row.mae_difference), percent(row.relative_gain),
      percent(row.precision), percent(row.recall), number(row.mae_std), number(row.worst_mae), row.severity_method,
      row.threshold.toFixed(3), row.severity_scale, row.amount_cap ?? 'None'];
    for (const value of values) {
      const cell = document.createElement('td'); cell.textContent = String(value); tr.append(cell);
    }
    const cell = document.createElement('td'), button = document.createElement('button');
    button.textContent = 'View'; button.addEventListener('click', () => details(row));
    cell.append(button); tr.append(cell); byId('rows').append(tr);
  }
}
function configuration(row, seed) {
  return {schema_version:study.metadata.config_schema_version, classifier:row.classifier,
    threshold:row.threshold, evaluation_percent:study.metadata.evaluation_percent,
    random_seed:seed, severity_method:row.severity_method, ridge_alpha:study.metadata.ridge_alpha,
    severity_scale:row.severity_scale, amount_cap:row.amount_cap, n_splits:5};
}
function download(name, content, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function details(row) {
  byId('dialog-title').textContent = 'Settings ' + row.candidate_ids.join(', ') + ' — seeds ' + row.seeds.join(', ');
  byId('seed-rows').replaceChildren();
  for (const score of row.per_seed) {
    const tr = document.createElement('tr');
    const baseline = study.metadata.baselines.find(item => item.seed === score.seed).mae;
    for (const value of [score.seed, number(score.mae), percent(score.precision), percent(score.recall),
      number(baseline), number(baseline - score.mae), percent(score.relative_gain), score.tp, score.fp]) {
      const td = document.createElement('td'); td.textContent = String(value); tr.append(td);
    }
    byId('seed-rows').append(tr);
  }
  byId('configuration').textContent = JSON.stringify(row, null, 2);
  byId('downloads').replaceChildren();
  for (const seed of row.seeds) {
    const button = document.createElement('button'); button.textContent = 'Download seed ' + seed;
    button.addEventListener('click', () => download('seed-study-' + row.candidate + '-seed-' + seed + '.json',
      JSON.stringify(configuration(row, seed), null, 2) + '\n'));
    byId('downloads').append(button);
  }
  byId('dialog').showModal();
}
for (const id of ['precision', 'recall', 'mae', 'every-seed', 'one-per-classifier', 'order'])
  byId(id).addEventListener('input', render);
byId('close').addEventListener('click', () => byId('dialog').close());
byId('export').addEventListener('click', () => download('three-seed-selected-results.json',
  JSON.stringify({metadata:study.metadata, records:visible}, null, 2)));
render();
