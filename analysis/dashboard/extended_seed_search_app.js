/* Unrestricted category ranking with per-configuration split and Ridge settings. */
const study = JSON.parse(document.getElementById('study-data').textContent);
const byId = id => document.getElementById(id);
const number = value => value.toFixed(4);
const percent = value => (100 * value).toFixed(2) + '%';
let shown = [];
for (const value of [...new Set(study.records.map(row => row.evaluation_percent))].sort((a,b) => a-b)) {
  const option = document.createElement('option'); option.value = value; option.textContent = value + '%';
  byId('evaluation').append(option);
}
function compare(a, b, category) {
  if (category === 'mae') return a.mae-b.mae || b.precision-a.precision || b.recall-a.recall || a.candidate-b.candidate;
  if (category === 'precision') return b.precision-a.precision || b.recall-a.recall || a.mae-b.mae || a.candidate-b.candidate;
  if (category === 'recall') return b.recall-a.recall || b.precision-a.precision || a.mae-b.mae || a.candidate-b.candidate;
  return b.aggregate-a.aggregate || a.mae_ratio-b.mae_ratio || a.mae-b.mae || a.candidate-b.candidate;
}
function render() {
  const category = byId('category').value;
  const evaluation = Number(byId('evaluation').value);
  const minimumPrecision = Number(byId('precision').value)/100;
  const minimumRecall = Number(byId('recall').value)/100;
  const rows = study.records.filter(row => (byId('scope').value !== 'new' || row.candidate_ids.some(id=>id>96))
    && (!evaluation || row.evaluation_percent===evaluation)
    && row.precision>=minimumPrecision && row.recall>=minimumRecall).sort((a,b)=>compare(a,b,category));
  const selected = [], seen = new Set();
  for (const row of rows) if (!seen.has(row.candidate)) { seen.add(row.candidate); selected.push(row); }
  const limit = byId('limit').value;
  shown = limit==='all' ? selected : selected.slice(0,Number(limit));
  byId('count').textContent = shown.length + ' shown of ' + selected.length + ' matching training configurations. Metrics average seeds 41, 42, and 43.';
  byId('rows').replaceChildren();
  for (const row of shown) {
    const tr = document.createElement('tr');
    const values = [row.candidate_ids.join(', '),row.evaluation_percent+'%',row.seeds.join(', '),
      row.classifier.family,number(row.mae),number(row.baseline_mae),number(row.mae_difference),
      percent(row.relative_gain),percent(row.precision),percent(row.recall),row.aggregate.toFixed(3),
      row.flagged_percent.toFixed(2)+'%'];
    for (const value of values) { const td=document.createElement('td'); td.textContent=String(value);tr.append(td); }
    const td=document.createElement('td'),button=document.createElement('button');button.textContent='View';
    button.addEventListener('click',()=>details(row));td.append(button);tr.append(td);byId('rows').append(tr);
  }
}
function download(name, value) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=name;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function details(row) {
  byId('dialog-title').textContent='Settings '+row.candidate_ids.join(', ')+' · seeds '+row.seeds.join(', ')+' · '+row.evaluation_percent+'% evaluation';
  byId('seed-rows').replaceChildren();
  for (const score of row.per_seed) {
    const tr=document.createElement('tr');
    for (const value of [score.seed,number(score.mae),percent(score.precision),percent(score.recall),
      number(score.baseline_mae),number(score.mae_difference),percent(score.relative_gain),score.tp,score.fp]) {
      const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }
    byId('seed-rows').append(tr);
  }
  byId('configuration').textContent=JSON.stringify(row,null,2);byId('downloads').replaceChildren();
  for (const seed of row.seeds) {
    const button=document.createElement('button');button.textContent='Download seed '+seed;
    button.addEventListener('click',()=>download('extended-'+row.candidate+'-seed-'+seed+'.json',{
      schema_version:study.metadata.config_schema_version,classifier:row.classifier,threshold:row.threshold,
      evaluation_percent:row.evaluation_percent,random_seed:seed,severity_method:row.severity_method,
      ridge_alpha:row.ridge_alpha,severity_scale:row.severity_scale,amount_cap:row.amount_cap,n_splits:5}));
    byId('downloads').append(button);
  }
  byId('dialog').showModal();
}
for (const id of ['category','scope','evaluation','precision','recall','limit']) byId(id).addEventListener('input',render);
byId('close').addEventListener('click',()=>byId('dialog').close());
byId('export').addEventListener('click',()=>download('expanded-seed-search-results.json',{metadata:study.metadata,records:shown}));
render();
