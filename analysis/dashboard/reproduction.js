/* Training configuration downloads and instructions for every dashboard run. */
const severityDescriptions = {
  weighted_profile:"Probability × profile median, with Ridge fallback",
  weighted_ridge:"Probability × Ridge severity",
  profile_median:"Unweighted profile median, with Ridge fallback",
  mixture_quantile:"Profile residual quantile, with minimum quantile 0.05",
};

function trainingConfiguration(row) {
  const classifier = row.settings ? {...row.settings} : {
    name:"Histogram Gradient Boosting",family:"hist",iterations:row.max_iter,rate:row.learning_rate,
    leaves:row.max_leaf_nodes,leaf:row.min_samples_leaf,l2:row.l2_regularization,
    oversampling:row.oversampling_percent,smoothing:row.smoothing,early_stopping:"auto",
  };
  delete classifier.amount_cap;
  return {schema_version:1,classifier,threshold:row.hasMetrics ? row.threshold : .1,
    evaluation_percent:row.evaluation_percent,random_seed:row.seed ?? 42,
    severity_method:row.severity_method || "weighted_profile",ridge_alpha:row.hasMetrics ? row.alpha : 1,
    severity_scale:row.hasMetrics ? row.scale : 1,amount_cap:row.amount_cap ?? null,n_splits:row.folds ?? 5};
}

function trainingPromptValues(row) {
  const config = trainingConfiguration(row), settings = config.classifier;
  const values = [["Claim probability threshold",config.threshold],["Evaluation percentage",config.evaluation_percent],
    ["Evaluation/model random seed",config.random_seed],["Severity rule",severityDescriptions[config.severity_method]],
    ["Ridge alpha",config.ridge_alpha],["Amount multiplier",config.severity_scale],
    ["Amount cap before multiplier",config.amount_cap ?? "None"],["CV folds",config.n_splits]];
  const labels = {family:"Classifier family",iterations:"Boosting iterations",rate:"Learning rate",leaves:"Maximum leaf nodes",
    leaf:"Minimum samples per leaf",depth:"Depth",l2:"L2 regularization",oversampling:"Oversampling target (%)",
    smoothing:"Profile-history smoothing",early_stopping:"Early stopping",amount_history:"Past-amount history"};
  for (const [key,value] of Object.entries(settings)) {
    if (key !== "name") values.push([labels[key] || key,String(value)]);
  }
  return values;
}

function trainingConfigInputs(row) {
  return JSON.stringify(trainingConfiguration(row),null,2) + "\n";
}

function reproductionPlan(row) {
  const filename = "run-" + row.fit_id + ".json";
  return {mode:"training",badge:"Training CLI",filename,prompts:trainingPromptValues(row),
    command:"python -m src.train --config " + filename,
    note:row.hasMetrics ? "Download the configuration, place it in the repository folder, then run the command below."
      : "Only classifier settings were saved for this run. The configuration uses defaults for the missing threshold, Ridge alpha and multiplier; its historical score cannot be reproduced exactly.",
    footnote:"Review the configuration summary and press Enter to train. The cutoff, amount rule, holdout and seed are saved in the file. After evaluation, the model is refitted on all labeled rows and saved for src.predict."};
}

function fillDefinitions(target, entries) {
  target.replaceChildren();
  for (const [label,value] of entries) {
    if (value === undefined || value === null) continue;
    const term = document.createElement("dt"), description = document.createElement("dd");
    term.textContent = label; description.textContent = value; target.append(term,description);
  }
}

function renderResultSummary(row) {
  elements.dialogTitle.textContent = "Run " + row.fit_id;
  elements.dialogSubtitle.textContent = row.name || "Histogram Gradient Boosting";
  elements.dialogMetrics.replaceChildren();
  for (const [label,key] of [["Holdout MAE","mae"],["Claim precision","precision"],["Claim recall","recall"]]) {
    const card = document.createElement("div"), name = document.createElement("span"), value = document.createElement("strong");
    name.textContent = label;
    value.textContent = row.hasMetrics ? (key === "mae" ? decimal(row[key]) : percent(row[key])) : "—";
    card.append(name,value); elements.dialogMetrics.append(card);
  }
  elements.dialogContext.replaceChildren();
  for (const label of [row.evaluation_percent + "% holdout", "Seed " + (row.seed ?? 42),
    row.hasMetrics ? "Threshold " + decimal(row.threshold,3) : "No saved metrics",
    ...(row.hasMetrics ? ["Amount × " + row.scale] : [])]) {
    const chip = document.createElement("span"); chip.textContent = label; elements.dialogContext.append(chip);
  }
}

function renderReproduction(row) {
  const plan = reproductionPlan(row);
  elements.workflowBadge.textContent = plan.badge;
  elements.workflowBadge.dataset.mode = plan.mode;
  elements.reproductionNote.textContent = plan.note;
  elements.reproductionSteps.hidden = plan.mode === "missing";
  elements.configSection.hidden = plan.mode !== "training";
  elements.reproCommand.textContent = plan.command || "";
  elements.reproductionFootnote.textContent = plan.footnote || "";
  elements.configValues.value = trainingConfigInputs(row) || "";
  elements.copyStatus.textContent = "";
  elements.promptValues.replaceChildren();
  for (const [label,value,description] of plan.prompts || []) {
    const item = document.createElement("li"), name = document.createElement("span"), input = document.createElement("code");
    name.textContent = label;
    input.textContent = value + (description ? " — " + description : "");
    item.append(name,input); elements.promptValues.append(item);
  }
  elements.promptValuesLabel.textContent = "Configuration values (" + (plan.prompts?.length || 0) + ")";
}

function renderModelDetails(row) {
  const settings = row.settings || {};
  fillDefinitions(elements.settingsList, [
    ["Status",row.status || "Historical fit"],["Classifier",row.classifier || "Histogram Gradient Boosting"],
    ["Training / evaluation rows",numbers.format(row.training_rows) + " / " + numbers.format(row.holdout_rows)],
    ["Saved metric configurations",numbers.format(row.qualifying)],
    ["Boosting iterations",settings.iterations ?? row.max_iter],["Learning rate",settings.rate ?? row.learning_rate],
    ["Maximum leaf nodes",settings.leaves ?? row.max_leaf_nodes],["Depth",settings.depth],
    ["Minimum samples per leaf",settings.leaf ?? row.min_samples_leaf],
    ["L2 regularization",settings.l2 ?? row.l2_regularization],
    ["Early stopping",settings.early_stopping ?? (row.status ? "Disabled" : "auto")],
    ["Oversampling target",row.oversampling_percent + "%"],["Oversampling sampling seed",42],
    ["Profile-history smoothing",row.smoothing],["Past claim amount feature",settings.amount_history ? "Enabled" : "Disabled"],
    ["Severity rule",severityDescriptions[row.severity_method || "weighted_profile"]],
    ["Ridge alpha",row.hasMetrics ? row.alpha : null],["Amount cap before multiplier",row.amount_cap],
  ]);
}

function showRunDetails(row) {
  renderResultSummary(row); renderReproduction(row); renderModelDetails(row);
  elements.modelDetails.open = false;
  elements.runDialog.showModal(); elements.runDialog.scrollTop = 0;
}

async function copyReproduction(target, message) {
  const value = target.value ?? target.textContent;
  try { await navigator.clipboard.writeText(value); }
  catch {
    const temporary = document.createElement("textarea");
    temporary.value = value; temporary.className = "clipboard-buffer"; elements.runDialog.append(temporary);
    temporary.focus(); temporary.select();
    const copied = document.execCommand("copy"); temporary.remove();
    if (!copied) {elements.copyStatus.textContent = "Copy is unavailable. Select and copy the displayed text.";return;}
  }
  elements.copyStatus.textContent = message;
}

function setupReproductionActions() {
  elements.copyCommand.addEventListener("click",() => copyReproduction(elements.reproCommand,"Command copied."));
  elements.copyConfig.addEventListener("click",() => copyReproduction(elements.configValues,"Configuration JSON copied."));
  elements.downloadConfig.addEventListener("click",() => {
    const blob = new Blob([elements.configValues.value], {type:"application/json"});
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = elements.reproCommand.textContent.split(" ").at(-1);
    link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
    elements.copyStatus.textContent = "Configuration downloaded. Place it in the repository folder.";
  });
}
