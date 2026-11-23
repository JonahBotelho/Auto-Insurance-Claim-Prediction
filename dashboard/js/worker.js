/* Query original binary scores together with every retained study observation. */
importScripts("checkpoint_points.js" + self.location.search, "grouping.js" + self.location.search, "data.js");

const dataDirectory = new URL("../data/", self.location.href);
let data, historicalLoad, historicalCount;
function allowedSeed(seed,filters) {return filters.seed===null || filters.seed===undefined || seed===filters.seed;}
function historicalScore(offset,run) {
  const tp=data.getUint16(offset+6,true),fp=data.getUint16(offset+8,true),mae=data.getFloat64(offset+10,true);
  const precision=tp+fp ? tp/(tp+fp) : 0,recall=tp/run.claims;
  return {mae,precision,recall};
}
function historicalPoint(group,offset,filters) {
  return aggregatePoint(group,[offset],pointAt(offset).split('/').map(Number),filters);
}
function fixedReruns(active,filters,favourites) {
  const rows=[];
  for (const group of active) for (const record of group.records) {
    if (record.source!==filters.source || (filters.runId && record.candidate!==filters.runId)) continue;
    const scores=record.per_seed.filter(score=>allowedSeed(score.seed,filters)).map(score=>({...score,
      label:record.label,source:record.source,record_id:record.uid,candidate:record.candidate}));
    if (!scores.length) continue;
    const row=aggregatePoint(group,scores,[record.threshold,record.ridge_alpha,record.scale],filters);
    const idOffset={five:100000,tuned:300000}[filters.source] ?? 0;
    row.fit_id=idOffset+record.candidate;row.candidate=record.candidate;
    if (row.seed_count>=filters.minimumSeeds && matches(row,filters)
        && (!favourites || favourites.has(row.favourite_key))) rows.push(row);
  }
  return rows;
}
function bookmarkedPoints(favourites) {
  const indexed=new Map();
  for (const key of favourites || []) {
    try {
      const saved=JSON.parse(key);
      if (typeof saved.group!=='string' || !Array.isArray(saved.seeds)
          || !(saved.point===null || typeof saved.point==='string')) continue;
      if (!indexed.has(saved.group)) indexed.set(saved.group,new Map());
      const points=indexed.get(saved.group);
      if (!points.has(saved.point)) points.set(saved.point,[]);
      points.get(saved.point).push(saved.seeds);
    } catch { /* Ignore obsolete or invalid browser-storage entries. */ }
  }
  return indexed;
}
function query(filters) {
  const favourites=filters.favouritesOnly ? new Set(filters.favouriteKeys || []) : null;
  const bookmarked=bookmarkedPoints(favourites);
  if (favourites && !bookmarked.size) return [];
  const active=new Set(groups.filter(group=>{
    const first=group.runs[0] || group.records[0];
    return (!favourites || bookmarked.has(group.key))
      && (!filters.evaluation || first.evaluation_percent===filters.evaluation)
      && (!filters.runId || group.runs.some(run=>run.fit_id===filters.runId)
        || group.records.some(row=>row.candidate===filters.runId || row.historical_run===filters.runId));
  }));
  // Every selected fixed point stays visible, including points sharing a classifier.
  if (['five','tuned','claim-history-tuning','claim-f1-tuning'].includes(filters.source))
    return fixedReruns(active,filters,favourites);
  const best=new Map(),points=new Map(),savedRows=[];
  function addPoint(group,key,reference) {
    if (favourites && !bookmarked.get(group.key)?.has(key)) return;
    if (!points.has(group)) points.set(group,new Map());
    const options=points.get(group);
    if (!options.has(key)) options.set(key,[]);
    options.get(key).push(reference);
  }
  if (!filters.source || filters.source==='historical') {
    for (let offset=0;offset<data.byteLength;offset+=18) {
      const id=data.getUint16(offset,true),group=groupByRun.get(id),run=metadata.get(id);
      if (!active.has(group) || !allowedSeed(run.seed ?? 42,filters)) continue;
      if (favourites) {addPoint(group,pointAt(offset),offset);continue;}
      // Stream singleton historical groups to avoid millions of JS objects.
      if (group.runs.length===1 && (!group.records.length || filters.source==='historical')) {
        if (filters.minimumSeeds>1) continue;
        const score=historicalScore(offset,run);
        if (!matches(score,filters)) continue;
        if (better(score,best.get(group))) best.set(group,historicalPoint(group,offset,filters));
      } else addPoint(group,pointAt(offset),offset);
    }
  }
  for (const group of active) for (const row of group.records) {
    if (filters.source && row.source!==filters.source) continue;
    const key=pointKey(row.threshold,row.ridge_alpha,row.scale);
    for (const score of row.per_seed) if (allowedSeed(score.seed,filters))
      addPoint(group,key,{...score,label:row.label,source:row.source,record_id:row.uid,
                          run_id:row.historical_run,candidate:row.candidate});
  }
  for (const [group,options] of points) {
    let coverage=0;
    if (filters.coverage==='maximum') for (const references of options.values())
      coverage=Math.max(coverage,new Set(references.map(ref=>typeof ref==='number' ? (metadata.get(data.getUint16(ref,true)).seed ?? 42) : ref.seed)).size);
    for (const [key,references] of options) {
      // A saved three-seed result remains available alongside its five-seed counterpart.
      const cohorts=favourites ? bookmarked.get(group.key).get(key) : [null];
      for (const seeds of cohorts) {
        const selected=seeds ? references.filter(reference=>seeds.includes(observation(reference).seed)) : references;
        if (!selected.length) continue;
        const row=aggregatePoint(group,selected,key.split('/').map(Number),filters);
        if (row.seed_count<Math.max(filters.minimumSeeds || 1,coverage) || !matches(row,filters)) continue;
        if (favourites) {if (favourites.has(row.favourite_key)) savedRows.push(row);}
        else if (better(row,best.get(group))) best.set(group,row);
      }
    }
  }
  const rows=favourites ? savedRows : [...best.values()];
  if (filters.includeMissing && filters.minimumSeeds===1 && (!filters.source || filters.source==='historical')) {
    for (const group of active) if (!best.has(group) && group.runs.length && group.runs.every(run=>run.qualifying===0)) {
      const first=group.runs[0];
      const key=favouriteKey(group,null,[]);
      if (favourites && !favourites.has(key)) continue;
      rows.push({...first,fit_id:group.id,favourite_key:key,hasMetrics:false,run_ids:group.runs.map(run=>run.fit_id),
        labels:group.runs.map(run=>'Run '+run.fit_id),source_ids:['historical'],
        seeds:[],seed_count:0,per_seed:[],observations:[],settings:group.settings});
    }
  }
  // Several bookmarked points may share a settings group; View needs distinct display IDs.
  return favourites ? rows.map((row,index)=>({...row,fit_id:400000+index})) : rows;
}
async function loadHistoricalScores() {
  const archive = await loadHistoricalArchive(dataDirectory);
  const stream = archive.stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  if (buffer.byteLength !== historicalCount * 18) throw new Error('Historical score data is incomplete.');
  data=new DataView(buffer);
}
self.onmessage=async ({data:message})=>{
  try {
    if (message.type==='init') {
      alphaValues=message.alphas;scaleValues=message.scales;
      historicalCount=message.count;
      initializeGroups(message.runs,message.records);
      self.postMessage({type:'ready'});return;
    }
    if (message.type==='query') {
      if (!message.filters.source || message.filters.source==='historical') {
        historicalLoad ??= loadHistoricalScores().catch(error => {
          historicalLoad = undefined;
          throw new Error('Unable to load historical scores. ' + error.message);
        });
        await historicalLoad;
      }
      const rows=query(message.filters);
      self.postMessage({type:'results',requestId:message.requestId,rows});
    }
  } catch (error) {self.postMessage({type:'error',requestId:message.requestId,message:String(error.message || error)});}
};
