/* Training configuration downloads and instructions for every dashboard run. */
const severityDescriptions = {
  weighted_profile:"Probability × profile median, with Ridge fallback",
  weighted_ridge:"Probability × Ridge severity",
  profile_median:"Unweighted profile median, with Ridge fallback",
  mixture_quantile:"Profile residual quantile, with minimum quantile 0.05",
  blended_weighted_profile:"Blended probability × primary model profile median",
  median_payout:"Median payout with tune 264 pricing for profiles without positive history",
  conditional_payout:"Conditional median of zero, repeated and novel payout outcomes",
  aggregate_payout:"Conditional median using observed payouts and components of scaled totals",
  relative_payout:"Conditional median with novel payouts estimated relative to the profile history",
  component_payout:"Conditional median from repeatable payouts and inferred missing components",
  cold_pooled_tail:"Pooled fitting second-percentile payout with the configured multiplier",
  cold_marked_tail:"Fitting second-percentile payout weighted by the predicted payout source",
  ordinal_survival:"Payout quote inferred from the fitted survival curve",
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
    smoothing:"Profile-history smoothing",early_stopping:"Early stopping",amount_history:"Past-amount history",
    extended_amount_history:"Past log-amount variance and whole-cent share"};
  for (const [key,value] of Object.entries(settings)) {
    if (key !== "name") values.push([labels[key] || key,String(value)]);
  }
  return values;
}

function trainingConfigInputs(row) {
  return JSON.stringify(trainingConfiguration(row),null,2) + "\n";
}

function usesMatchedHoldouts(row) {
  return ["profile_reranking","fixed_flag_severity","no_positive_history","known_profile_coverage","claim_pruning",
    "batch_claim_detection","claim_detector_comparison","payout_signatures","claim_coverage_screen","profile_structure"].includes(row.reproduction);
}

function reproductionPlan(row) {
  if (row.reproduction === "profile_generalization") {
    return {mode:"analysis",badge:"Profile generalization",prompts:[],command:row.reproduction_command,
      note:row.reproduction_note || "Run the group-disjoint reference before the cold-start calibration.",
      footnote:"Each evaluation holds out at least 20% of the labeled rows. The split definition is in the run's command."};
  }
  if (row.reproduction === "profile_structure") {
    const command = row.native_training_command ||
      (row.opt_in_config ? "python -m src.train --config " + row.opt_in_config : row.reproduction_command);
    return {mode:"analysis",badge:"Profile structure study",prompts:[],command,
      note:row.reproduction_note || (row.opt_in_config ? "Train the opt-in configuration from the repository root."
        : "Generate the median-pricing study outputs and run the corresponding profile experiments."),
      footnote:row.native_prediction_command
        ? "Predict with " + row.native_prediction_command + ". Reproduce the holdouts with " + row.reproduction_command + "."
        : "The dashboard aggregates five matching holdouts. Reserved-split checks are in the source download."};
  }
  if (row.reproduction === "payout_signatures") {
    return {mode:"analysis",badge:"Median pricing and detection",prompts:[],
      command:"python -m analysis.evaluate_payout_signatures --output-dir analysis/output/payout_signatures_rerun",
      note:"Evaluate all configurations on the matched 25% holdouts for seeds 41–45.",
      footnote:"The fixed CatBoost median-pricing recipe was also evaluated on reserved seed 2601."};
  }
  if (row.reproduction === "claim_coverage_screen") {
    return {mode:"analysis",badge:"Claim coverage screen",prompts:[],
      command:"python -m analysis.screen_claim_coverage --results analysis/output/payout_signatures_rerun",
      note:"Run the median-pricing and detection study first. The screen adds the 100 highest-ranked rows with no positive history.",
      footnote:"The fixed addition budget measures candidate precision on seeds 41–45."};
  }
  if (row.reproduction === "claim_detector_comparison") {
    return {mode:"analysis",badge:"Experiment 7",prompts:[],
      command:"python -m analysis.evaluate_claim_detectors --output-dir analysis/output/claim_detector_comparison_rerun",
      note:"Replay the controls and evaluate both challengers on seeds 41–45.",
      footnote:"The dashboard scores are historical. Confirmation status and learned cutoffs are saved with the study."};
  }
  if (row.reproduction === "batch_claim_detection") {
    return {mode:"missing",badge:"Archived experiment",
      note:"Tune 264 was retained after development. Scores and predictions are in analysis/output/batch_claim_detection/."};
  }
  if (row.reproduction === "claim_pruning") {
    return {mode:"missing",badge:"Archived experiment",
      note:"Tune 264 was retained after development. Scores and predictions are in analysis/output/claim_pruning/."};
  }
  if (row.reproduction === "known_profile_coverage") {
    return {mode:"analysis",badge:"Historical-size comparison",prompts:[],
      command:"python -m analysis.evaluate_known_profile_coverage --output-dir analysis/output/known_profile_coverage_rerun",
      note:"Evaluate four coverage rules on tune 264's historical splits. The runner also compares 42,000 fitting rows on the same holdouts.",
      footnote:"Choose a fresh output directory for each run."};
  }
  if (row.reproduction === "no_positive_history") {
    return {mode:"missing",badge:"Archived experiment",
      note:"Tune 264 was retained after development. Saved scores and predictions are in analysis/output/no_positive_history/."};
  }
  if (row.reproduction === "fixed_flag_severity") {
    return {mode:"missing",badge:"Archived experiment",
      note:"Tune 264 was retained after development. Saved scores and predictions are in analysis/output/fixed_flag_severity/."};
  }
  if (row.reproduction === "profile_reranking") {
    return {mode:"missing",badge:"Archived experiment",
      note:"The experiment stopped after development and its runner was removed. Saved predictions and split manifests are in analysis/output/known_profile_reranking_41_45/."};
  }
  if (row.submission) {
    return {mode:"analysis",badge:"Checkpoint 2",prompts:[],
      command:"python -m src.generate_checkpoint2_candidates --submission "+row.submission+" --seed "+row.training_seed,
      note:"Generate this submission using its frozen configuration and the full training dataset.",
      footnote:"The command writes the submission CSV to data/checkpoint2/ and the training metrics to data/checkpoint2 metrics/."};
  }
  if (row.reproduction === "checkpoint") {
    const seed=row.reproduction_seed ?? row.seed ?? 42;
    return {mode:"analysis",badge:"Checkpoint 2",prompts:[],
      command:"python -m analysis.evaluate_checkpoint2 --submission "+row.settings.submission+" --seeds "+seed,
      note:"This evaluates the original checkpoint blend shown in this historical row on one 25% holdout without replacing application models or submission files.",
      footnote:"The current ten-file selection is documented in submission.md. Generate it with python -m src.generate_checkpoint2_candidates --seed 42; CSVs are written to data/checkpoint2/ and metrics to data/checkpoint2 metrics/."};
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
  elements.dialogSubtitle.textContent = row.submission || row.settings.family==='checkpoint_blend'
    ? "Checkpoint 2 submission" : (row.name || row.settings.family);
  elements.dialogMetrics.replaceChildren();
  for (const [label,key] of [["Mean MAE","mae"],["Lowest seed MAE","min_mae"],["All-zero mean MAE","baseline_mae"],
    ["MAE reduction","mae_difference"],["Improvement","relative_gain"],["Claim precision","precision"],["Claim recall","recall"],["Mean F1","f1"]]) {
    const card = document.createElement("div"), name = document.createElement("span"), value = document.createElement("strong");
    name.textContent = label;
    value.textContent = row.hasMetrics ? (["precision","recall","relative_gain"].includes(key) ? percent(row[key]) : decimal(row[key])) : "—";
    card.append(name,value); elements.dialogMetrics.append(card);
  }
  elements.dialogContext.replaceChildren();
  const thresholdLabel = usesMatchedHoldouts(row) ? "Incumbent threshold " : "Threshold ";
  const thresholdText = row.threshold_method === "experiment7_oof"
    ? "Cutoffs fitted per seed"
    : row.threshold_method === "experiment7_blend" ? "Average log-odds margin ≥ 0"
    : row.threshold_method === "coverage_top_100" ? "100 additions ranked per split"
    : row.threshold_method === "coverage_top_25" ? "25 additions ranked per split"
    : row.threshold_method === "coverage_top_10" ? "10 additions ranked per split"
    : row.threshold_method === "exposure_link" ? "Maximum difference in feature1 " + decimal(row.threshold,3)
    : row.threshold_method === "expected_f1" ? "Cutoff selected from expected F1"
    : row.threshold_method === "bootstrap_majority" ? "Flag when more than half of fitted-profile resamples agree"
    : row.threshold_method === "detector_agreement" ? "Apply an agreement rule to the fitted classifiers' flags"
    : row.threshold_method === "source_prune" ? "Reference flags with corrected probability of at least 0.22"
    : row.threshold_method === "reference_volume" ? "Reference flag count retained after reranking"
    : row.threshold_method === "fixed_flags" ? "Same claim flags as the reference detector"
    : row.threshold_method === "component_veto" ? "Veto supported flags below claim probability " + decimal(row.threshold,3)
    : row.threshold_method === "cold_hazard_gate" ? "Add no-history flags at claim probability " + decimal(row.threshold,3) + " or above"
    : row.threshold_method === "marginal_f1" ? "Add no-history flags when they improve expected portfolio F1"
    : row.threshold_method === "reduced_flag_volume" ? "Rerank at 95% of the reference flag count"
    : row.threshold_method === "reference_cutoff" ? "Apply the reference cutoff to adjusted claim risk"
    : row.threshold_method === "profile_presence_veto" ? "Reference flags with profile-presence risk of at least 0.25"
    : row.threshold_method === "replicated_risk" ? "Add no-history flags only when independently confirmed risk clears 25%"
    : row.threshold_method === "integrated_expected_f1" ? "Optimize expected F1 over the predictive claim-count distribution"
    : row.threshold_method === "majority_five" ? "Flag when at least three of five detectors agree"
    : row.threshold_method === "override_vote_five" ? "Change a profile-presence decision only on four-model agreement"
    : row.threshold_method === "prune_vote_five" ? "Prune a profile-presence flag rejected by four of five detectors"
    : ["claim_detector_comparison","payout_signatures","profile_structure"].includes(row.reproduction) && !row.threshold_method
    ? "Threshold " + decimal(row.threshold,3)
    : row.threshold_method === "nested_f1"
    ? "Mean selected threshold " + decimal(row.per_seed.reduce((sum, seed) => sum + seed.threshold, 0) / row.per_seed.length, 3)
    : thresholdLabel + decimal(row.threshold,3);
  for (const label of [row.evaluation_label || row.evaluation_percent + "% holdout", "Seeds " + ((row.seeds || []).join(", ") || "unavailable"),
    row.hasMetrics ? thresholdText : "No saved metrics",
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
  const unseenProfiles = row.reproduction === "profile_generalization";
  const reference = row.comparison_label || (unseenProfiles ? "payout-source ensemble" : "tune 264");
  const pairedDelta = key => (usesMatchedHoldouts(row) || unseenProfiles) && row.per_seed.every(score=>Number.isFinite(score[key]))
    ? decimal(row.per_seed.reduce((total, score) => total + score[key], 0) / row.per_seed.length, 6)
    : null;
  fillDefinitions(elements.settingsList, [
    ["Status",row.status || "Historical fit"],["Classifier",row.classifier || "Histogram Gradient Boosting"],
    ["Sources",row.source_ids.join(', ')],
    ["Evaluation rows by seed",row.per_seed.map(score=>score.seed+': '+numbers.format(score.rows)).join(', ')],
    ["Model fitting seed",row.model_seed],["MAE change vs " + reference,pairedDelta("delta_mae")],
    ["F1 change vs " + reference,pairedDelta("delta_f1")],
    ["MAE points",row.mae_points?.toFixed(2)],["F1 points",row.f1_points?.toFixed(2)],["Aggregate points",row.aggregate?.toFixed(2)],
    ["Boosting iterations",settings.iterations ?? row.max_iter],["Learning rate",settings.rate ?? row.learning_rate],
    ["Maximum leaf nodes",settings.leaves ?? row.max_leaf_nodes],["Depth",settings.depth],
    ["Minimum samples per leaf",settings.leaf ?? row.min_samples_leaf],
    ["L2 regularization",settings.l2 ?? row.l2_regularization],
    ["Early stopping",settings.early_stopping ?? (row.status ? "Disabled" : "auto")],
    ["Oversampling target",settings.oversampling!=null ? settings.oversampling + "%" : "See blend components"],["Oversampling sampling seed",42],
    ["Profile-history smoothing",settings.smoothing],["Past claim amount feature",settings.amount_history ? "Enabled" : "Disabled"],
    ["Extended amount history",settings.extended_amount_history ? "Past log-amount variance and whole-cent share" : null],
    ["Profile amount relationships",settings.profile_amount_relationships],
    ["Experiment amount rule",row.severity_experiment_rule],
    ["Experiment cap on adjusted amounts",row.severity_experiment_cap],
    ["Severity rule",severityDescriptions[row.severity_method || "weighted_profile"]],
    ["Ridge alpha",row.hasMetrics ? row.alpha : null],["Amount cap before multiplier",row.amount_cap],
    ["Blend components",settings.family==='checkpoint_blend' ? JSON.stringify(settings) : null],
  ]);
}

function showRunDetails(row) {
  elements.reproductionSeed.closest("label").hidden = Boolean(row.submission) || usesMatchedHoldouts(row);
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
