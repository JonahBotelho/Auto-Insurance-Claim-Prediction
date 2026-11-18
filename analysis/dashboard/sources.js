/* Lossless source exports and source selection for the single dashboard. */
function downloadPayload(name,value,type='application/json') {
  const content=typeof value==='string' || value instanceof Uint8Array ? value : JSON.stringify(value,null,2);
  const url=URL.createObjectURL(new Blob([content],{type})),link=document.createElement('a');
  link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function setupSourceActions() {
  function button(label,action) {
    const control=document.createElement('button');control.textContent=label;control.className='secondary';
    control.addEventListener('click',action);elements.sourceDownloads.append(control);
  }
  button('Original historical metadata',()=>downloadPayload('historical.json',dataset));
  button('Original historical scores (.gz)',()=>{
    const bytes=Uint8Array.from(atob(elements['score-data'].textContent.trim()),character=>character.charCodeAt(0));
    downloadPayload('historical_scores.gz',bytes,'application/gzip');
  });
  for (const source of unified.sources) {
    const option=document.createElement('option');option.value=source.id;option.textContent=source.name;elements.source.append(option);
    button(source.name+' (JSON)',()=>downloadPayload(source.id+'-complete-source.json',source));
  }
  button('Archive checksums',()=>downloadPayload('dashboard-source-manifest.json',unified.archive_manifest));
  const requested=new URLSearchParams(location.search).get('source');
  elements.source.value=[...elements.source.options].some(option=>option.value===requested) ? requested : defaultSource;
  if (elements.source.value===f1Study) setPriority('f1');
  elements.source.addEventListener('change',()=>{
    if (elements.source.value===f1Study) setPriority('f1');
    query();
  });
}
