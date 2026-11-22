/* Lossless source exports and source selection for the dashboard. */
function downloadPayload(name,value) {
  const blob=value instanceof Blob ? value : new Blob([JSON.stringify(value,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function setupSourceActions() {
  function button(label,action) {
    const control=document.createElement('button');control.textContent=label;control.className='secondary';
    control.addEventListener('click',action);elements.sourceDownloads.append(control);
  }
  button('Original historical metadata',()=>downloadPayload('historical.json',dataset));
  button('Original historical scores (.gz)',async ()=>{
    try {
      const archive = await loadHistoricalArchive(dataDirectory);
      downloadPayload('historical_scores.gz', archive);
    } catch (error) {
      showError('Unable to download historical scores. ' + error.message);
    }
  });
  for (const source of unified.sources) {
    if (!source.archived) {
      const option=document.createElement('option');option.value=source.id;option.textContent=source.name;elements.source.append(option);
    }
    button(source.name+' (JSON)',()=>downloadPayload(source.id+'-complete-source.json',source));
  }
  button('Archive checksums',()=>downloadPayload('dashboard-source-manifest.json',unified.archive_manifest));
  const requested=new URLSearchParams(location.search).get('source');
  elements.source.value=[...elements.source.options].some(option=>option.value===requested) ? requested : defaultSource;
  if (elements.source.value===f1Study) setSort('f1');
  elements.source.addEventListener('change',()=>{
    if (elements.source.value===f1Study) setSort('f1');
    query();
  });
}
