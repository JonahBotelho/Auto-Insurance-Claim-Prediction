/* Training configuration downloads and instructions for every dashboard run. */
const severityDescriptions = {
  weighted_profile:"Probability × profile median, with Ridge fallback",
  weighted_ridge:"Probability × Ridge severity",
  profile_median:"Unweighted profile median, with Ridge fallback",
  mixture_quantile:"Profile residual quantile, with minimum quantile 0.05",
  blended_weighted_profile:"Blended probability × primary model profile median",
};

function trainingConfiguration(row) {
  const classifier = row.settings ? {...row.settings} : {
    name:"Histogram Gradient Boosting",family:"hist",iterations:row.max_iter,rate:row.learning_rate,
    leaves:row.max_leaf_nodes,leaf:row.min_samples_leaf,l2:row.l2_regularization,
    oversampling:row.oversampling_percent,smoothing:row.smoothing,early_stopping:"auto",
  };
  delete classifier.amount_cap;
  return {schema_version:1,classifier,threshold:row.hasMetrics ? row.threshold : .1,
    evaluation_percent:row.evaluation_percent,random_seed:row.reproduction_seed ?? row.seed ?? 42,
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
  if (row.reproduction === "checkpoint") {
    const seed=row.reproduction_seed ?? row.seed ?? 42;
    return {mode:"analysis",badge:"Checkpoint 2",prompts:[],
      command:"python -m analysis.evaluate_checkpoint2 --submission "+row.settings.submission+" --seeds "+seed,
      note:"This is the exact checkpoint 2 submission rule. The command evaluates its fixed blend on one 25% holdout without replacing application models or submission files.",
      footnote:"To regenerate all ten submission CSVs after fitting on all labeled rows, run python -m src.generate_checkpoint2_candidates --seed 42. It writes data/2_2_1.csv through data/2_2_10.csv."};
  }
  const filename = "settings-" + row.fit_id + "-seed-" + (row.reproduction_seed ?? row.seed ?? 42) + ".json";
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
  elements.dialogTitle.textContent = row.labels.join(", ");
  elements.dialogSubtitle.textContent = row.settings.family==='checkpoint_blend' ? "Checkpoint 2 submission" : (row.name || row.settings.family);
  elements.dialogMetrics.replaceChildren();
  for (const [label,key] of [["Mean MAE","mae"],["All-zero mean MAE","baseline_mae"],
    ["MAE reduction","mae_difference"],["Improvement","relative_gain"],["Claim precision","precision"],["Claim recall","recall"],["Mean F1","f1"]]) {
    const card = document.createElement("div"), name = document.createElement("span"), value = document.createElement("strong");
    name.textContent = label;
    value.textContent = row.hasMetrics ? (["precision","recall","relative_gain"].includes(key) ? percent(row[key]) : decimal(row[key])) : "—";
    card.append(name,value); elements.dialogMetrics.append(card);
  }
  elements.dialogContext.replaceChildren();
  for (const label of [row.evaluation_percent + "% holdout", "Seeds " + ((row.seeds || []).join(", ") || "unavailable"),
    row.hasMetrics ? "Threshold " + decimal(row.threshold,3) : "No saved metrics",
    ...(row.hasMetrics ? ["Amount × " + row.scale] : [])]) {
    const chip = document.createElement("span"); chip.textContent = label; elements.dialogContext.append(chip);
  }
  elements.seedMetrics.replaceChildren();
  for (const score of row.per_seed || []) {
    const tr = document.createElement("tr");
    for (const value of [score.seed,decimal(score.mae),decimal(score.baseline_mae),decimal(score.mae_difference),
      percent(score.precision),percent(score.recall),decimal(score.f1)]) {
      const td = document.createElement("td"); td.textContent = String(value); tr.append(td);
    }
    elements.seedMetrics.append(tr);
  }
  elements.observationMetrics.replaceChildren();
  for (const score of row.observations || []) {
    const tr=document.createElement('tr');
    for (const value of [score.label,score.seed,decimal(score.mae),percent(score.precision),percent(score.recall),decimal(score.f1)]) {
      const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }
    elements.observationMetrics.append(tr);
  }
  elements.observationNote.textContent=(row.observations?.length || 0)+" source observations, "+row.seed_count+" unique seeds. "
    +(row.conflicts ? row.conflicts+" differing repeated-seed observations are retained; the first saved observation supplies that seed's average." : "Repeated observations do not add weight to the averages.");
  elements.downloadObservations.onclick=()=>downloadPayload('settings-'+row.fit_id+'-observations.json',row);
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
  elements.configValues.value = plan.mode === "training" ? trainingConfigInputs(row) : "";
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
    ["Sources",row.source_ids.join(', ')],
    ["Evaluation rows by seed",row.per_seed.map(score=>score.seed+': '+numbers.format(score.rows)).join(', ')],
    ["Aggregate",row.hasMetrics ? decimal(row.aggregate) : null],
    ["Boosting iterations",settings.iterations ?? row.max_iter],["Learning rate",settings.rate ?? row.learning_rate],
    ["Maximum leaf nodes",settings.leaves ?? row.max_leaf_nodes],["Depth",settings.depth],
    ["Minimum samples per leaf",settings.leaf ?? row.min_samples_leaf],
    ["L2 regularization",settings.l2 ?? row.l2_regularization],
    ["Early stopping",settings.early_stopping ?? (row.status ? "Disabled" : "auto")],
    ["Oversampling target",settings.oversampling!=null ? settings.oversampling + "%" : "See blend components"],["Oversampling sampling seed",42],
    ["Profile-history smoothing",settings.smoothing],["Past claim amount feature",settings.amount_history ? "Enabled" : "Disabled"],
    ["Severity rule",severityDescriptions[row.severity_method || "weighted_profile"]],
    ["Ridge alpha",row.hasMetrics ? row.alpha : null],["Amount cap before multiplier",row.amount_cap],
    ["Blend components",settings.family==='checkpoint_blend' ? JSON.stringify(settings) : null],
  ]);
}

function showRunDetails(row) {
  elements.reproductionSeed.replaceChildren();
  for (const seed of row.seeds?.length ? row.seeds : [row.seed ?? 42]) {
    const option = document.createElement("option"); option.value = seed; option.textContent = seed;
    elements.reproductionSeed.append(option);
  }
  elements.reproductionSeed.value = row.reproduction_seed ?? row.seed ?? 42;
  elements.reproductionSeed.onchange = () => {
    row.reproduction_seed = Number(elements.reproductionSeed.value); renderReproduction(row);
  };
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
