"use strict";
const dataset = JSON.parse(document.getElementById("run-data").textContent);
const elements = Object.fromEntries(Array.from(document.querySelectorAll("[id]")).map(element => [element.id, element]));
const metricInputs = ["maeMin","maeMax","precisionMin","precisionMax","recallMin","recallMax"];
let rows = [], page = 0, pageSize = 50, sortKey = "mae", sortDirection = 1;
let ready = false, requestId = 0, debounce;
const numbers = new Intl.NumberFormat("en-US");
const percent = value => (value * 100).toFixed(2) + "%";
const decimal = (value, digits = 4) => Number(value).toFixed(digits);
const missing = value => value === undefined || value === null;
for (const [value, label] of [
  [dataset.runs.length,"runs"],
  [dataset.runs.filter(run => run.qualifying > 0).length,"with saved scores"],
  [dataset.runs.filter(run => run.qualifying === 0).length,"without saved metrics"],
  [dataset.scoreCount,"saved configurations"],
]) {
  const span = document.createElement("span"), strong = document.createElement("strong");
  strong.textContent = numbers.format(value);
  span.append(strong, " " + label); elements.datasetSummary.append(span);
}
for (const value of [...new Set(dataset.runs.map(run => run.evaluation_percent))].sort((a,b) => a-b)) {
  const option = document.createElement("option");
  option.value = value; option.textContent = value + "% evaluation / " + (100-value) + "% training";
  elements.evaluation.append(option);
}
for (const value of [...new Set(dataset.runs.map(run => run.seed ?? 42))].sort((a,b) => a-b)) {
  const option = document.createElement("option"); option.value = value; option.textContent = value;
  elements.seed.append(option);
}
const workerUrl = URL.createObjectURL(new Blob([elements["worker-source"].textContent], {type:"text/javascript"}));
const worker = new Worker(workerUrl);
worker.postMessage({type:"init", runs:dataset.runs, count:dataset.scoreCount,
  payload:elements["score-data"].textContent.trim()});

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
  filters.includeMissing = elements.includeMissing.checked;
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
    worker.postMessage({type:"query",filters,priority:elements.priority.value,requestId:++requestId});
  } catch (error) {
    requestId++; showError(error.message); elements.download.disabled = true;
    elements.resultCount.textContent = "Fix the invalid filters to update results.";
  }
}
function queueQuery() { clearTimeout(debounce); debounce = setTimeout(query,150); }
worker.onmessage = ({data: message}) => {
  if (message.type === "ready") {
    ready = true; elements.download.disabled = false;
    URL.revokeObjectURL(workerUrl); query();
  } else if (message.type === "results" && message.requestId === requestId) {
    rows = message.rows.map(row => ({...row,
      alpha:row.hasMetrics ? dataset.alphas[row.alphaIndex] : null,
      scale:row.hasMetrics ? dataset.scales[row.scaleIndex] : null}));
    page = 0; orderRows(); render();
  } else if (message.type === "error") showError(message.message);
};
worker.onerror = event => showError("Unable to load the local score data. Use a current Chrome or Edge browser. " + event.message);

function orderRows() {
  rows.sort((left,right) => {
    const a = left[sortKey], b = right[sortKey];
    if (missing(a) && missing(b)) return left.fit_id-right.fit_id;
    if (missing(a)) return 1;
    if (missing(b)) return -1;
    return (a-b) * sortDirection || left.fit_id-right.fit_id;
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
  elements.resultCount.textContent = numbers.format(rows.length) + " matching runs / "
    + numbers.format(scored) + " scored / " + numbers.format(rows.length-scored) + " without saved metrics";
  elements.tableBody.replaceChildren();
  const maximumPage = Math.max(0,Math.ceil(rows.length/pageSize)-1);
  page = Math.min(page,maximumPage);
  const start = page * pageSize, fragment = document.createDocumentFragment();
  for (const row of rows.slice(start,start+pageSize)) {
    const tr = document.createElement("tr"); tr.dataset.run = row.fit_id;
    const id = cell(String(row.fit_id));
    if (row.status) {
      const note = document.createElement("span");
      note.className = "availability";
      note.textContent = row.status + " / " + row.name + " / seed " + row.seed; id.append(note);
    }
    if (!row.hasMetrics) {
      const note = document.createElement("span");
      note.className = "availability"; note.textContent = "No saved qualifying metrics"; id.append(note);
    }
    tr.append(id,cell(row.evaluation_percent + "%"));
    for (const name of ["mae","precision","recall","threshold"]) {
      const value = row[name];
      const display = missing(value) ? String.fromCharCode(8212) : (name === "precision" || name === "recall") ? percent(value) : decimal(value,name === "threshold" ? 3 : 4);
      const td = cell(display, missing(value) ? "missing" : "score" + (name === sortKey ? " metric-primary" : ""));
      if (!missing(value)) td.title = String(name === "precision" || name === "recall" ? value * 100 : value);
      tr.append(td);
    }
    tr.append(cell(row.hasMetrics ? numbers.format(row.tp) : String.fromCharCode(8212)),
      cell(row.hasMetrics ? numbers.format(row.fp) : String.fromCharCode(8212)),cell(row.oversampling_percent + "%"));
    const action = cell(""), button = document.createElement("button");
    button.className = "view-button"; button.dataset.openRun = row.fit_id; button.textContent = "View";
    button.setAttribute("aria-label","View settings for run " + row.fit_id);
    action.append(button); tr.append(action); fragment.append(tr);
  }
  if (!rows.length) {
    const tr = document.createElement("tr"), td = cell("No saved configurations match these filters. Clear or widen the filters.","empty");
    td.colSpan = 10; tr.append(td); fragment.append(tr);
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
  const fields = ["fit_id","status","name","severity_method","evaluation_percent","mae","precision","recall","threshold","alpha","scale","tp","fp",
    "max_iter","learning_rate","max_leaf_nodes","min_samples_leaf","l2_regularization","oversampling_percent","smoothing","hasMetrics"];
  const lines = [fields.join(",")];
  for (const row of rows) lines.push(fields.map(key => missing(row[key]) ? "" : String(row[key])).join(","));
  const url = URL.createObjectURL(new Blob([lines.join("\r\n")],{type:"text/csv;charset=utf-8"}));
  const link = document.createElement("a"); link.href = url; link.download = "filtered_model_runs.csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
}
for (const key of [...metricInputs,"runId"]) elements[key].addEventListener("input",queueQuery);
elements.evaluation.addEventListener("change",query); elements.seed.addEventListener("change",query);
elements.includeMissing.addEventListener("change",query);
elements.priority.addEventListener("change",() => {
  sortKey = elements.priority.value; sortDirection = sortKey === "mae" ? 1 : -1; query();
});
elements.reset.addEventListener("click",() => {
  for (const key of [...metricInputs,"runId"]) elements[key].value = "";
  elements.evaluation.value = ""; elements.seed.value = ""; elements.includeMissing.checked = true; query();
});
elements.previousLimits.addEventListener("click",() => {
  for (const key of metricInputs) elements[key].value = "";
  elements.maeMax.value = 96; elements.precisionMin.value = 45; elements.recallMin.value = 30; query();
});
for (const button of document.querySelectorAll("[data-sort]")) button.addEventListener("click",() => {
  const key = button.dataset.sort;
  sortDirection = key === sortKey ? -sortDirection : ["precision","recall","tp"].includes(key) ? -1 : 1;
  sortKey = key; page = 0; orderRows(); render();
});
elements.pageSize.addEventListener("change",() => {
  pageSize = elements.pageSize.value === "all" ? Math.max(1, rows.length) : Number(elements.pageSize.value);
  page = 0; render();
});
elements.previous.addEventListener("click",() => {page--;render();}); elements.next.addEventListener("click",() => {page++;render();});
elements.tableBody.addEventListener("click",event => {
  const button = event.target.closest("[data-open-run]"); if (button) openRun(Number(button.dataset.openRun));
});
elements.closeDialog.addEventListener("click",() => elements.runDialog.close());
setupReproductionActions();
elements.download.addEventListener("click",exportCsv);
