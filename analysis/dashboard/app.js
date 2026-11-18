"use strict";
const dataset = JSON.parse(document.getElementById("run-data").textContent);
const unified = JSON.parse(document.getElementById("unified-data").textContent);
const elements = Object.fromEntries(Array.from(document.querySelectorAll("[id]")).map(element => [element.id, element]));
const metricInputs = ["maeMin","maeMax","precisionMin","precisionMax","recallMin","recallMax"];
const f1Study = "claim-f1-tuning";
const defaultSource = unified.sources.some(source => source.id === f1Study) ? f1Study : "";
let rows = [], page = 0, pageSize = 50, sortKey = "mae", sortDirection = 1;
let ready = false, historicalRequested = false, requestId = 0, debounce;
const numbers = new Intl.NumberFormat("en-US");
const percent = value => (value * 100).toFixed(2) + "%";
const decimal = (value, digits = 4) => Number(value).toFixed(digits);
const missing = value => value === undefined || value === null;
for (const [value, label] of [
  [dataset.runs.length,"historical runs"],
  [dataset.runs.filter(run => run.qualifying > 0).length,"with original scores"],
  [dataset.runs.filter(run => run.qualifying === 0).length,"with discarded original scores"],
  [dataset.scoreCount,"saved configurations"],
  [unified.records.length,"retained study points"],
]) {
  const span = document.createElement("span"), strong = document.createElement("strong");
  strong.textContent = numbers.format(value);
  span.append(strong, " " + label); elements.datasetSummary.append(span);
}
for (const value of [...new Set([...dataset.runs,...unified.records].map(run => run.evaluation_percent))].sort((a,b) => a-b)) {
  const option = document.createElement("option");
  option.value = value; option.textContent = value + "% evaluation / " + (100-value) + "% training";
  elements.evaluation.append(option);
}
for (const value of [...new Set([...dataset.runs.map(run => run.seed ?? 42),...unified.records.flatMap(row=>row.per_seed.map(score=>score.seed))])].sort((a,b) => a-b)) {
  const option = document.createElement("option"); option.value = value; option.textContent = value;
  elements.seed.append(option);
}
const workerUrl = URL.createObjectURL(new Blob([elements["worker-source"].textContent], {type:"text/javascript"}));
const worker = new Worker(workerUrl);
worker.postMessage({type:"init", runs:dataset.runs, records:unified.records, alphas:dataset.alphas, scales:dataset.scales, count:dataset.scoreCount});

function showError(message) {
  elements.filterError.textContent = message; elements.filterError.hidden = false;
}
function readFilters() {
  const filters = {};
  for (const name of metricInputs) {
    const input = elements[name], value = input.value === "" ? null : input.valueAsNumber;
    input.removeAttribute("aria-invalid");
    if (input.validity.badInput || (value !== null && (!Number.isFinite(value) || value < 0
        || (name !== "maeMin" && name !== "maeMax" && value > 100)))) {
      input.setAttribute("aria-invalid","true");
      throw new Error("Enter valid nonnegative values. Precision and recall must be between 0 and 100.");
    }
    filters[name] = value;
  }
  for (const prefix of ["mae","precision","recall"]) {
    if (filters[prefix+"Min"] !== null && filters[prefix+"Max"] !== null
        && filters[prefix+"Min"] > filters[prefix+"Max"])
      throw new Error("Each minimum must be less than or equal to its maximum.");
  }
  const id = elements.runId.value === "" ? null : elements.runId.valueAsNumber;
  if (elements.runId.validity.badInput || (id !== null && (!Number.isInteger(id) || id < 1)))
    throw new Error("Run ID must be a positive whole number.");
  filters.runId = id;
  filters.evaluation = elements.evaluation.value ? Number(elements.evaluation.value) : null;
  filters.seed = elements.seed.value ? Number(elements.seed.value) : null;
  filters.minimumSeeds = Number(elements.minimumSeeds.value);
  filters.source = elements.source.value;
  filters.coverage = elements.coverage.value;
  filters.includeMissing = elements.includeMissing.checked;
  filters.favouritesOnly = elements.favouritesOnly.checked;
  filters.favouriteKeys = filters.favouritesOnly ? [...favouriteKeys] : [];
  const bounds = metricInputs.some(key => filters[key] !== null);
  elements.includeMissing.disabled = bounds;
  elements.includeMissing.parentElement.classList.toggle("disabled", bounds);
  elements.includeMissing.title = bounds ? "Metric filters exclude unavailable values. Clear metric filters to include these runs." : "";
  return filters;
}
function query() {
  if (!ready) return;
  clearTimeout(debounce);
  try {
    const filters = readFilters();
    elements.filterError.hidden = true;
    elements.resultCount.textContent = "Finding matching configurations...";
    const request = {type:"query",filters,priority:elements.priority.value,requestId:++requestId};
    if (!historicalRequested && (!filters.source || filters.source === "historical")) {
      request.payload = elements["score-data"].textContent.trim();
      historicalRequested = true;
    }
    worker.postMessage(request);
  } catch (error) {
    requestId++; showError(error.message); elements.download.disabled = true;
    elements.resultCount.textContent = "Fix the invalid filters to update results.";
  }
}
function queueQuery() { clearTimeout(debounce); debounce = setTimeout(query,150); }
worker.onmessage = ({data: message}) => {
  if (message.type === "ready") {
    ready = true; elements.download.disabled = false;
    const summary = document.createElement("span"); summary.textContent = numbers.format(message.groups) + " settings groups";
    elements.datasetSummary.append(summary);
    URL.revokeObjectURL(workerUrl); query();
  } else if (message.type === "results" && message.requestId === requestId) {
    rows = message.rows;
    page = 0; orderRows(); render();
  } else if (message.type === "error" && (message.requestId === undefined || message.requestId === requestId)) {
    showError(message.message);
    elements.resultCount.textContent = "Unable to load matching configurations.";
  }
};
worker.onerror = event => showError("Unable to load the local score data. Use a current Chrome or Edge browser. " + event.message);

function orderRows() {
  rows.sort((left,right) => {
    const a = left[sortKey], b = right[sortKey];
    if (missing(a) && missing(b)) return left.fit_id-right.fit_id;
    if (missing(a)) return 1;
    if (missing(b)) return -1;
    return (a-b) * sortDirection || (sortKey === "f1" ? left.mae-right.mae : 0) || left.fit_id-right.fit_id;
  });
  for (const button of document.querySelectorAll("[data-sort]")) {
    const active = button.dataset.sort === sortKey;
    button.parentElement.setAttribute("aria-sort", active ? (sortDirection === 1 ? "ascending" : "descending") : "none");
    button.querySelector("span").textContent = active ? String.fromCharCode(sortDirection === 1 ? 9650 : 9660) : "";
  }
}
function cell(text, className = "") {
  const element = document.createElement("td");
  element.textContent = text; element.className = className; return element;
}
function render() {
  if (elements.pageSize.value === "all") pageSize = Math.max(1, rows.length);
  const scored = rows.filter(row => row.hasMetrics).length;
  const countLabel = elements.source.value === f1Study ? " matching runs / " : " matching settings groups / ";
  elements.resultCount.textContent = numbers.format(rows.length) + (elements.favouritesOnly.checked ? " favourited results / " : countLabel)
    + numbers.format(scored) + " scored / " + numbers.format(rows.length-scored) + " without saved metrics";
  elements.tableBody.replaceChildren();
  const maximumPage = Math.max(0,Math.ceil(rows.length/pageSize)-1);
  page = Math.min(page,maximumPage);
  const start = page * pageSize, fragment = document.createDocumentFragment();
  for (const row of rows.slice(start,start+pageSize)) {
    const tr = document.createElement("tr"); tr.dataset.run = row.fit_id;
    const favourite = cell("", "favourite-cell"); favourite.append(favouriteButton(row));
    const id = cell(row.labels.slice(0,3).join(", ")+(row.labels.length>3 ? " +"+(row.labels.length-3) : ""), "run-label");
    id.title = row.labels.join(", ");
    const sourceNote=document.createElement('span');sourceNote.className='availability';
    sourceNote.textContent=row.source_ids.join(' / ');id.append(sourceNote);
    if (row.status) {
      const note = document.createElement("span");
      note.className = "availability";
      note.textContent = row.status + " / " + (row.name || "Historical fit"); id.append(note);
    }
    if (!row.hasMetrics) {
      const note = document.createElement("span");
      note.className = "availability"; note.textContent = "No saved qualifying metrics"; id.append(note);
    }
    tr.append(favourite,id,cell(row.evaluation_percent + "%"),cell(row.hasMetrics ? row.seed_count + " (" + row.seeds.join(", ") + ")" : "—"));
    for (const name of ["mae","min_mae","baseline_mae","mae_difference","relative_gain","precision","recall","f1","aggregate","threshold"]) {
      const value = row[name];
      const isPercent = ["precision","recall","relative_gain"].includes(name);
      const display = missing(value) ? String.fromCharCode(8212) : isPercent ? percent(value) : decimal(value,name === "threshold" ? 3 : 4);
      const td = cell(display, missing(value) ? "missing" : "score" + (name === sortKey ? " metric-primary" : ""));
      if (!missing(value)) td.title = String(isPercent ? value * 100 : value);
      tr.append(td);
    }
    tr.append(cell(row.hasMetrics ? numbers.format(row.tp) : String.fromCharCode(8212)),
      cell(row.hasMetrics ? numbers.format(row.fp) : String.fromCharCode(8212)),cell(row.settings.oversampling!=null ? row.settings.oversampling + "%" : "Blend"));
    const action = cell(""), button = document.createElement("button");
    button.className = "view-button"; button.dataset.openRun = row.fit_id; button.textContent = "View";
    button.setAttribute("aria-label","View settings for run " + row.fit_id);
    action.append(button); tr.append(action); fragment.append(tr);
  }
  if (!rows.length) {
    const message = elements.favouritesOnly.checked ? "No favourites match these filters. Star a row or widen the filters."
      : "No saved configurations match these filters. Clear or widen the filters.";
    const tr = document.createElement("tr"), td = cell(message,"empty");
    td.colSpan = 18; tr.append(td); fragment.append(tr);
  }
  elements.tableBody.append(fragment);
  elements.pageInfo.textContent = rows.length ? numbers.format(start+1) + "-" + numbers.format(Math.min(start+pageSize,rows.length)) + " of " + numbers.format(rows.length) : "0 results";
  elements.previous.disabled = page === 0; elements.next.disabled = page >= maximumPage;
  elements.download.disabled = !ready || rows.length === 0;
}
function openRun(id) {
  const row = rows.find(item => item.fit_id === id);
  if (row) showRunDetails(row);
}
function exportCsv() {
  const fields = ["labels","source_ids","run_ids","seed_count","seeds","status","name","severity_method","evaluation_percent","mean_mae","min_mae","all_zero_mean_mae","mae_difference","relative_gain","precision","recall","f1","aggregate","threshold","alpha","scale","tp","fp",
    "max_iter","learning_rate","max_leaf_nodes","min_samples_leaf","l2_regularization","oversampling_percent","smoothing","hasMetrics"];
  const lines = [fields.join(",")];
  const escape = value => '"' + String(Array.isArray(value) ? value.join(';') : value).replaceAll('"','""') + '"';
  for (const row of rows) lines.push(fields.map(key => missing(row[key]) ? "" : escape(row[key])).join(","));
  const url = URL.createObjectURL(new Blob([lines.join("\r\n")],{type:"text/csv;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = "filtered_model_runs.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}
for (const key of [...metricInputs,"runId"]) elements[key].addEventListener("input",queueQuery);
elements.evaluation.addEventListener("change",query); elements.seed.addEventListener("change",query);
elements.minimumSeeds.addEventListener("change",query);
elements.coverage.addEventListener("change",query);
elements.includeMissing.addEventListener("change",query);
function setPriority(key) {
  elements.priority.value = sortKey = key;
  sortDirection = key === "mae" ? 1 : -1;
}
elements.priority.addEventListener("change",() => {setPriority(elements.priority.value); query();});
elements.reset.addEventListener("click",() => {
  for (const key of [...metricInputs,"runId"]) elements[key].value = "";
  elements.evaluation.value = ""; elements.seed.value = ""; elements.minimumSeeds.value = "1"; elements.source.value=defaultSource; elements.coverage.value="available"; elements.includeMissing.checked = true; elements.favouritesOnly.checked = false;
  setPriority(defaultSource === f1Study ? "f1" : "mae"); query();
});
elements.previousLimits.addEventListener("click",() => {
  for (const key of metricInputs) elements[key].value = "";
  elements.precisionMin.value = 45; elements.recallMin.value = 30; query();
});
for (const button of document.querySelectorAll("[data-sort]")) button.addEventListener("click",() => {
  const key = button.dataset.sort;
  sortDirection = key === sortKey ? -sortDirection : ["precision","recall","f1","tp","mae_difference","relative_gain","seed_count","aggregate"].includes(key) ? -1 : 1;
  sortKey = key; page = 0; orderRows(); render();
});
elements.pageSize.addEventListener("change",() => {
  pageSize = elements.pageSize.value === "all" ? Math.max(1, rows.length) : Number(elements.pageSize.value);
  page = 0; render();
});
elements.previous.addEventListener("click",() => {page--;render();}); elements.next.addEventListener("click",() => {page++;render();});
elements.tableBody.addEventListener("click",event => {
  const favourite = event.target.closest("[data-favourite]");
  if (favourite) {toggleFavourite(favourite.dataset.favourite);return;}
  const button = event.target.closest("[data-open-run]"); if (button) openRun(Number(button.dataset.openRun));
});
elements.closeDialog.addEventListener("click",() => elements.runDialog.close());
setupReproductionActions();
setupSourceActions();
setupFavourites();
elements.download.addEventListener("click",exportCsv);
