/* Load the dashboard's static data assets from the local server. */
async function fetchDashboardData(directory, name) {
  const response = await fetch(new URL(name, directory));
  if (!response.ok) throw new Error(`Could not load ${name} (HTTP ${response.status}).`);
  return response;
}

async function loadHistoricalArchive(directory) {
  const response = await fetchDashboardData(directory, "historical-scores.json");
  const manifest = await response.json();
  if (!Array.isArray(manifest?.parts) || !manifest.parts.length) {
    throw new Error("The historical score manifest is incomplete.");
  }
  // The chunks are consecutive bytes of one gzip stream. Their order also preserves the original download.
  const chunks = await Promise.all(manifest.parts.map(async ({name, bytes}) => {
    const response = await fetchDashboardData(directory, name);
    const chunk = await response.blob();
    if (chunk.size !== bytes) throw new Error(`Historical score file ${name} is incomplete.`);
    return chunk;
  }));
  return new Blob(chunks, {type: "application/gzip"});
}
