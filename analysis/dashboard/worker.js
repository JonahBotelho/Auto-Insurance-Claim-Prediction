let data, runs, maximumId;
self.onmessage = async ({data: message}) => {
  try {
    if (message.type === "init") {
      runs = message.runs;
      maximumId = Math.max(...runs.map(run => run.fit_id));
      const bytes = Uint8Array.from(atob(message.payload), character => character.charCodeAt(0));
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      const buffer = await new Response(stream).arrayBuffer();
      if (buffer.byteLength !== message.count * 18) throw new Error("Embedded score data is incomplete.");
      data = new DataView(buffer);
      self.postMessage({type:"ready"});
      return;
    }
    if (message.type !== "query") return;
    const {filters, priority, requestId} = message;
    const matchingRun = new Uint8Array(maximumId + 1);
    const metadata = new Array(maximumId + 1);
    for (const run of runs) {
      metadata[run.fit_id] = run;
      matchingRun[run.fit_id] = (!filters.evaluation || run.evaluation_percent === filters.evaluation)
        && (!filters.runId || run.fit_id === filters.runId) ? 1 : 0;
    }
    const indices = new Int32Array(maximumId + 1).fill(-1);
    const bestMae = new Float64Array(maximumId + 1).fill(Infinity);
    const bestPrecision = new Float64Array(maximumId + 1);
    const bestRecall = new Float64Array(maximumId + 1);
    for (let offset = 0; offset < data.byteLength; offset += 18) {
      const id = data.getUint16(offset, true);
      if (!matchingRun[id]) continue;
      const tp = data.getUint16(offset + 6, true);
      const fp = data.getUint16(offset + 8, true);
      const precision = tp / (tp + fp);
      const recall = tp / metadata[id].claims;
      const mae = data.getFloat64(offset + 10, true);
      if ((filters.maeMin !== null && mae < filters.maeMin)
          || (filters.maeMax !== null && mae > filters.maeMax)
          || (filters.precisionMin !== null && precision * 100 < filters.precisionMin)
          || (filters.precisionMax !== null && precision * 100 > filters.precisionMax)
          || (filters.recallMin !== null && recall * 100 < filters.recallMin)
          || (filters.recallMax !== null && recall * 100 > filters.recallMax)) continue;
      let better = indices[id] === -1;
      if (!better && priority === "mae")
        better = mae < bestMae[id] || (mae === bestMae[id] && (precision > bestPrecision[id]
          || (precision === bestPrecision[id] && recall > bestRecall[id])));
      if (!better && priority === "precision")
        better = precision > bestPrecision[id] || (precision === bestPrecision[id]
          && (recall > bestRecall[id] || (recall === bestRecall[id] && mae < bestMae[id])));
      if (!better && priority === "recall")
        better = recall > bestRecall[id] || (recall === bestRecall[id]
          && (precision > bestPrecision[id] || (precision === bestPrecision[id] && mae < bestMae[id])));
      if (better) {
        indices[id] = offset;
        bestMae[id] = mae;
        bestPrecision[id] = precision;
        bestRecall[id] = recall;
      }
    }
    const hasBounds = ["maeMin","maeMax","precisionMin","precisionMax","recallMin","recallMax"]
      .some(key => filters[key] !== null);
    const rows = [];
    for (const run of runs) {
      const id = run.fit_id, offset = indices[id];
      if (!matchingRun[id]) continue;
      if (offset < 0) {
        if (run.qualifying === 0 && filters.includeMissing && !hasBounds)
          rows.push({...run, hasMetrics:false});
        continue;
      }
      rows.push({...run, hasMetrics:true, mae:bestMae[id], precision:bestPrecision[id],
        recall:bestRecall[id], threshold:data.getUint16(offset + 2,true)/1000,
        alphaIndex:data.getUint8(offset+4), scaleIndex:data.getUint8(offset+5),
        tp:data.getUint16(offset+6,true), fp:data.getUint16(offset+8,true)});
    }
    self.postMessage({type:"results", requestId, rows});
  } catch (error) {
    self.postMessage({type:"error", message:String(error.message || error)});
  }
};
