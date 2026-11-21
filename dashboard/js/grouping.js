/* Shared settings identities, seed deduplication, and baseline-aware metrics. */
let groups, groupByRun, metadata, alphaValues, scaleValues;
function stableKey(value) {
  if (Array.isArray(value)) return '['+value.map(stableKey).join(',')+']';
  if (value && typeof value==='object') return '{'+Object.keys(value).sort().map(key=>JSON.stringify(key)+':'+stableKey(value[key])).join(',')+'}';
  return JSON.stringify(value);
}
function classifierSettings(run) {
  const settings=run.settings ? {...run.settings} : {family:'hist',iterations:run.max_iter,rate:run.learning_rate,
    leaves:run.max_leaf_nodes,leaf:run.min_samples_leaf,l2:run.l2_regularization,
    oversampling:run.oversampling_percent,smoothing:run.smoothing,early_stopping:'auto'};
  delete settings.name;delete settings.amount_cap;settings.amount_history ??= false;
  if (!settings.extended_amount_history) delete settings.extended_amount_history;
  return settings;
}
function initializeGroups(runs,records) {
  const indexed=new Map();metadata=new Map();groupByRun=new Map();
  function groupFor(run) {
    const settings=classifierSettings(run),key=stableKey({settings,evaluation:run.evaluation_percent,
      severity:run.severity_method || 'weighted_profile',cap:run.amount_cap ?? null,folds:run.folds ?? 5,
      // Matched reranking cohorts use their own fitted partitions and flag quotas.
      ...(run.evaluation_cohort ? {cohort:run.evaluation_cohort} : {}),
      // A submitted configuration keeps its documented holdouts even in All sources.
      ...(run.submission ? {submission:run.submission} : {})});
    if (!indexed.has(key)) indexed.set(key,{id:indexed.size+1,key,runs:[],records:[],settings});
    return indexed.get(key);
  }
  for (const run of runs) {
    metadata.set(run.fit_id,run);const group=groupFor(run);group.runs.push(run);groupByRun.set(run.fit_id,group);
  }
  for (const record of records) groupFor(record).records.push(record);
  groups=[...indexed.values()];
}
function pointKey(threshold,alpha,scale) {
  return [threshold,alpha,scale].map(value=>Number(value.toPrecision(12))).join('/');
}
function pointAt(offset) {
  return pointKey(data.getUint16(offset+2,true)/1000,alphaValues[data.getUint8(offset+4)],scaleValues[data.getUint8(offset+5)]);
}
function favouriteKey(group,point,seeds) {
  // Display IDs change when sources are rebuilt; settings and seed cohorts identify the saved result.
  return stableKey({group:group.key,point:point===null ? null : pointKey(...point),seeds});
}
function observation(reference) {
  if (typeof reference!=='number') return reference;
  const run=metadata.get(data.getUint16(reference,true)),tp=data.getUint16(reference+6,true),fp=data.getUint16(reference+8,true);
  return {seed:run.seed ?? 42,run_id:run.fit_id,label:'Run '+run.fit_id,source:'historical',
    mae:data.getFloat64(reference+10,true),baseline_mae:run.zero_mae,precision:tp+fp ? tp/(tp+fp) : 0,
    recall:tp/run.claims,tp,fp,rows:run.holdout_rows,claims:run.claims};
}
function claimF1({precision, recall}) {
  return precision + recall ? 2 * precision * recall / (precision + recall) : 0;
}
function aggregatePoint(group,references,point,filters) {
  const observations=references.map(observation).map(score=>({...score,f1:claimF1(score)}));
  const unique=new Map();let conflicts=0;
  for (const score of observations) {
    const previous=unique.get(score.seed);
    if (previous && ['mae','baseline_mae','precision','recall'].some(key=>Math.abs(previous[key]-score[key])>1e-8)) conflicts++;
    if (!previous) unique.set(score.seed,score);
  }
  const perSeed=[...unique.values()].sort((a,b)=>a.seed-b.seed).map(score=>({...score,
    mae_difference:score.baseline_mae-score.mae,relative_gain:1-score.mae/score.baseline_mae}));
  const mean=key=>perSeed.every(row=>row[key]!=null) ? perSeed.reduce((sum,row)=>sum+row[key],0)/perSeed.length : null;
  const first=group.runs[0] || group.records[0],precision=mean('precision'),recall=mean('recall'),gain=mean('relative_gain');
  const mae=mean('mae'),f1=mean('f1');
  const sourceIds=[...new Set(observations.map(row=>row.source))];
  const labels=[...new Set(observations.map(row=>row.label))];
  return {...first,fit_id:group.id,settings:group.settings,hasMetrics:true,labels,source_ids:sourceIds,
    run_ids:[...new Set(observations.map(row=>row.run_id).filter(Boolean))],
    observations,conflicts,per_seed:perSeed,seeds:perSeed.map(row=>row.seed),seed_count:perSeed.length,
    favourite_key:favouriteKey(group,point,perSeed.map(row=>row.seed)),
    seed:unique.has(42) ? 42 : perSeed[0].seed,mae,mean_mae:mae,
    min_mae:Math.min(...observations.map(score=>score.mae)),
    baseline_mae:mean('baseline_mae'),all_zero_mean_mae:mean('baseline_mae'),mae_difference:mean('mae_difference'),
    // Average per-seed F1, since F1 computed from mean precision/recall differs.
    relative_gain:gain,precision,recall,f1,...checkpointPoints(mae,f1),tp:mean('tp'),fp:mean('fp'),
    flagged_percent:perSeed.every(row=>row.rows!=null) ? perSeed.reduce((sum,row)=>sum+100*(row.tp+row.fp)/row.rows,0)/perSeed.length : null,
    threshold:point[0],alpha:point[1],scale:point[2],reproduction:group.records.find(row=>row.reproduction)?.reproduction};
}
function matches(row,filters) {
  return (filters.maeMax === null || row.mae <= filters.maeMax)
    && (filters.precisionMin === null || row.precision * 100 >= filters.precisionMin)
    && (filters.recallMin === null || row.recall * 100 >= filters.recallMin)
    && (filters.f1Min == null || (row.f1 ?? claimF1(row)) >= filters.f1Min);
}
function better(row,best) {
  if (!best) return true;
  return row.mae<best.mae || (row.mae===best.mae && (row.precision>best.precision
    || (row.precision===best.precision && row.recall>best.recall)));
}
