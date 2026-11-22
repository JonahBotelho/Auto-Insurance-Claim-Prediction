"""Publish the historical-size coverage comparison to the local dashboard."""

import argparse
import base64
import csv
import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

from analysis.evaluate_known_profile_coverage import ARMS, OUTPUT, ROOT, SPLIT_SEEDS

SOURCE = "known-profile-coverage"
SOURCE_NAME = "Historical-size comparison · seeds 41–45"
FIRST_ID = 830001
DATA_TAG = '<script id="unified-data" type="application/json">'
REPRODUCTION_MARKER = "/* Training configuration downloads and instructions for every dashboard run. */"


def observation(row):
    counts = ("seed", "rows", "claims", "tp", "fp", "fn", "flags", "added_flags", "added_claims", "added_false_alarms")
    metrics = ("mae", "f1", "precision", "recall", "baseline_mae", "delta_mae", "delta_f1", "break_even_precision")
    return {
        **{key: int(row[key]) for key in counts},
        **{key: float(row[key]) for key in metrics},
        "addition_precision": float(row["addition_precision"]) if row["addition_precision"] else None,
    }


def study_source(results):
    manifest = json.loads((results / "split_manifest.json").read_text(encoding="utf-8"))
    verdict = json.loads((results / "verdict.json").read_text(encoding="utf-8"))
    with (results / "scores.csv").open(encoding="utf-8", newline="") as stream:
        historical = [row for row in csv.DictReader(stream) if row["cohort"] == "historical"]
    config = manifest["baseline_config"]
    policies = {policy["rule"]: policy for policy in verdict["policies"]}
    statuses = {"A": "Historical incumbent"}
    for arm in ("B", "C", "D"):
        rule = policies["agreement" if arm == "D" else "all_known"]
        statuses[arm] = "Coverage held up" if rule["holds_up"] else "Coverage fell short"
    if verdict["specialist"]:
        statuses[verdict["specialist"]] = "F1 specialist"
    if verdict["benchmark"]:
        statuses[verdict["benchmark"]] = "Retained as benchmark"

    records = []
    for index, (arm, name) in enumerate(ARMS.items()):
        observations = [observation(row) for row in historical if row["arm"] == arm]
        if tuple(row["seed"] for row in observations) != SPLIT_SEEDS:
            raise ValueError(f"Missing or unordered scores for arm {arm}.")
        label = f"Coverage {arm} · {name}"
        records.append({
            "uid": f"{SOURCE}:{arm}", "source": SOURCE, "candidate": FIRST_ID + index, "label": label, "name": label,
            "status": statuses[arm], "classifier": "Histogram Gradient Boosting", "settings": config["classifier"],
            "model_seed": config["random_seed"], "evaluation_cohort": f"{SOURCE}:{arm}",
            "evaluation_percent": 25, "evaluation_label": "52,500 fitting / 17,500 holdout rows",
            "reproduction": "known_profile_coverage", "threshold": config["threshold"], "ridge_alpha": config["ridge_alpha"],
            "scale": config["severity_scale"], "severity_method": config["severity_method"],
            "amount_cap": config["amount_cap"], "folds": config["n_splits"], "per_seed": observations,
        })
    return {"id": SOURCE, "name": SOURCE_NAME, "data": {
        "records": records, "metadata": {"baseline_uid": manifest["baseline_uid"], **verdict},
    }}


def update_studies(unified, source):
    source_id = source["id"]
    records = source["data"]["records"]
    kept = [row for row in unified["records"] if row["source"] != source_id]
    if {row.get("candidate") for row in kept} & {row["candidate"] for row in records}:
        raise ValueError("One or more study result candidate IDs are already in use.")
    unified["records"] = kept + records
    unified["sources"] = [entry for entry in unified["sources"] if entry["id"] != source_id] + [source]
    return json.dumps(unified, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", r"\u003c")


def update_legacy_dashboard(original, source):
    start = original.index(DATA_TAG) + len(DATA_TAG)
    stop = original.index("</script>", start)
    payload = original[start:stop]
    unified = json.loads(payload)
    if isinstance(unified, str):
        original_json = gzip.decompress(base64.b64decode(unified))
        updated_json = update_studies(json.loads(original_json), source).encode("utf-8")
        if updated_json != original_json:
            payload = json.dumps(base64.b64encode(gzip.compress(updated_json, mtime=0)).decode("ascii"))
    else:
        payload = update_studies(unified, source)
    updated = original[:start] + payload + original[stop:]

    start = updated.index(REPRODUCTION_MARKER)
    stop = updated.index("</script>", start)
    script = (ROOT / "dashboard/js/reproduction.js").read_text(encoding="utf-8").strip()
    if "\r\n" in original:
        script = script.replace("\n", "\r\n")
    return updated[:start] + script + updated[stop:]


def replace_file(target, content):
    staging = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}-", delete=False) as stream:
            staging = Path(stream.name)
            stream.write(content)
        # Windows can briefly lock a large dashboard during scanning or preview.
        for attempt in range(4):
            try:
                staging.replace(target)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 3:
                    raise
                sleep(.25 * 2 ** attempt)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def publish(dashboard, source):
    original = dashboard.read_bytes()
    if DATA_TAG.encode("ascii") in original:
        updated = update_legacy_dashboard(original.decode("utf-8"), source).encode("utf-8")
        if updated != original:
            replace_file(dashboard, updated)
        return

    studies = dashboard.parent / "data" / "studies.json.gz"
    original_json = gzip.decompress(studies.read_bytes())
    updated_json = update_studies(json.loads(original_json), source).encode("utf-8")
    if updated_json != original_json:
        replace_file(studies, gzip.compress(updated_json, mtime=0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUTPUT)
    parser.add_argument("--dashboard", type=Path, default=ROOT / "dashboard/index.html")
    args = parser.parse_args(argv)
    source = study_source(args.results)
    publish(args.dashboard, source)
    print(f"Published {len(source['data']['records'])} historical-size coverage results to {args.dashboard.resolve()}.")


if __name__ == "__main__":
    main()
