"""Publish the near-zero pricing runs to the local dashboard (dashboard/index.html).

Adds one dashboard source with four matched-holdout records on seeds 41-45. For 20% and
25% holdouts it publishes the count-retrieval timeline model as rerun on this computer,
and the same fits with $1 written for flagged rows whose claim risk is below 0.45.
Run analysis/cache_final_model_outputs.py for matched_20 and matched_25 first.

Example: python -m analysis.publish_final_candidates
"""

import argparse
import base64
import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from analysis.cache_final_model_outputs import OUTPUT_DIRECTORY, ROOT, cache_path
from analysis.evaluate_final_candidates import NEAR_ZERO_PRICE, variant_predictions
from src.reproduce_timeline_models import SEEDS

SOURCE = "near-zero-pricing"
SOURCE_NAME = "Near-zero pricing for low-confidence flags (seeds 41–45)"
FIRST_ID = 1170001
LOW_CUT = .45
FRESH_SEEDS = (46, 47, 48, 49, 50)
DATA_TAG = '<script id="unified-data" type="application/json">'
REPRODUCTION_MARKER = "/* Training configuration downloads and instructions for every dashboard run. */"
# The same model as published by the Bonus-Malus timeline study; used for labels and rerun deltas.
PUBLISHED_UIDS = {"matched_20": "bonus-malus-timeline:matched_20:count_retrieval_bm_timeline",
                  "matched_25": "bonus-malus-timeline:matched_25:count_retrieval_bm_timeline"}


def seed_scores(frame, predictions):
    """Return one dashboard per-seed entry from held-out amounts and predictions."""
    actual = frame.actual.to_numpy()
    claimed, flagged = actual > 0, predictions > 0
    tp, fp, claims = int((claimed & flagged).sum()), int((~claimed & flagged).sum()), int(claimed.sum())
    return {"seed": int(frame.seed.iloc[0]), "threshold": float(frame.cutoff.iloc[0]), "rows": len(frame),
            "claims": claims, "baseline_mae": float(np.abs(actual).mean()),
            "mae": float(np.abs(actual - predictions).mean()), "f1": 2 * tp / (tp + fp + claims),
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / claims,
            "tp": tp, "fp": fp, "fn": claims - tp, "flags": tp + fp}


def with_deltas(scores, reference):
    """Attach per-seed F1 and MAE changes against the matching reference seeds."""
    by_seed = {row["seed"]: row for row in reference}
    return [{**row, "delta_f1": row["f1"] - by_seed[row["seed"]]["f1"],
             "delta_mae": row["mae"] - by_seed[row["seed"]]["mae"]} for row in scores]


def cohort_scores(cohort, seeds, low_cut):
    frames = [pd.read_csv(cache_path(OUTPUT_DIRECTORY, cohort, seed)) for seed in seeds]
    return [seed_scores(frame, variant_predictions(frame, "cr", low_cut, 1., 1.)) for frame in frames]


def cohort_records(cohort, published, first_id):
    rerun, rule = cohort_scores(cohort, SEEDS, None), cohort_scores(cohort, SEEDS, LOW_CUT)
    family = "Count-retrieval + Bonus-Malus timeline"
    shared = {key: published[key] for key in ("evaluation_percent", "evaluation_label", "threshold", "ridge_alpha",
                                               "scale", "amount_cap", "severity_method", "folds")}
    shared.update(source=SOURCE, classifier=family, model_seed=42, threshold_method="expected_f1",
                  reproduction="profile_structure",
                  reproduction_command=f"python -m analysis.cache_final_model_outputs --cohort {cohort}; "
                                       f"python -m analysis.evaluate_final_candidates --cohort {cohort}")
    variants = [
        ("rerun", "Count-retrieval + French Bonus-Malus timeline, rerun on this computer",
         "Rerun control for the $1 rule", {"family": family}, with_deltas(rerun, published["per_seed"]),
         f"published run {published['candidate']}",
         "Fit the count-retrieval timeline model on each holdout. Package versions can shift a few flags."),
        ("rule", f"Count-retrieval + French Bonus-Malus timeline with $1 below {LOW_CUT:.0%} claim risk",
         "Lowest mean MAE on matched holdouts",
         {"family": family, "low_confidence_below": LOW_CUT, "low_confidence_price": NEAR_ZERO_PRICE},
         with_deltas(rule, rerun), "the same rerun without the $1 rule",
         "Fit the count-retrieval timeline model, then write $1 for flagged rows below 0.45 claim risk."),
    ]
    records = []
    for offset, (key, label, status, settings, per_seed, reference, note) in enumerate(variants):
        uid = f"{SOURCE}:{cohort}:{key}"
        # A unique evaluation cohort keeps the dashboard from pooling these seeds with the published run.
        records.append({**shared, "uid": uid, "candidate": first_id + offset, "label": label, "name": label,
                        "status": status, "settings": settings, "evaluation_cohort": uid,
                        "comparison_label": reference, "reproduction_note": note, "per_seed": per_seed})
    return records


def read_payload(html):
    start = html.index(DATA_TAG) + len(DATA_TAG)
    stop = html.index("</script>", start)
    stored = json.loads(html[start:stop])
    if not isinstance(stored, str):
        raise ValueError("Expected the compressed dashboard payload.")
    return start, stop, json.loads(gzip.decompress(base64.b64decode(stored)))


def encode_payload(unified):
    text = json.dumps(unified, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", r"<")
    return json.dumps(base64.b64encode(gzip.compress(text.encode("utf-8"), mtime=0)).decode("ascii"))


def build_source(unified):
    published = {record["uid"]: record for record in unified["records"] if record["uid"] in PUBLISHED_UIDS.values()}
    records = []
    for index, (cohort, uid) in enumerate(PUBLISHED_UIDS.items()):
        records += cohort_records(cohort, published[uid], FIRST_ID + 2 * index)
    rerun, rule = cohort_scores("matched_20", FRESH_SEEDS, None), cohort_scores("matched_20", FRESH_SEEDS, LOW_CUT)
    fresh = {name: {"mean_f1": float(np.mean([row["f1"] for row in rows])),
                    "mean_mae": float(np.mean([row["mae"] for row in rows]))}
             for name, rows in (("rerun", rerun), ("rule", rule))}
    metadata = {"low_confidence_below": LOW_CUT, "low_confidence_price": NEAR_ZERO_PRICE,
                "reserved_split_check": {"cohort": "matched_20", "seeds": list(FRESH_SEEDS), **fresh}}
    return {"id": SOURCE, "name": SOURCE_NAME, "data": {"records": records, "metadata": metadata}}


def publish(dashboard):
    html = dashboard.read_text(encoding="utf-8")
    start, stop, unified = read_payload(html)
    source = build_source(unified)
    kept = [record for record in unified["records"] if record["source"] != SOURCE]
    if {record.get("candidate") for record in kept} & {record["candidate"] for record in source["data"]["records"]}:
        raise ValueError("One or more new candidate IDs are already in use.")
    unified["records"] = kept + source["data"]["records"]
    unified["sources"] = [entry for entry in unified["sources"] if entry["id"] != SOURCE] + [source]
    updated = html[:start] + encode_payload(unified) + html[stop:]
    # Keep the page's inline copy of reproduction.js in step with the source file.
    start = updated.index(REPRODUCTION_MARKER)
    stop = updated.index("</script>", start)
    script = (ROOT / "analysis/dashboard/reproduction.js").read_text(encoding="utf-8").strip()
    updated = updated[:start] + script + updated[stop:]
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=dashboard.parent, delete=False,
                            prefix=f".{SOURCE}-") as stream:
        stream.write(updated)
    Path(stream.name).replace(dashboard)
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=ROOT / "dashboard/index.html")
    args = parser.parse_args(argv)
    source = publish(args.dashboard)
    for record in source["data"]["records"]:
        f1, mae = (np.mean([row[key] for row in record["per_seed"]]) for key in ("f1", "mae"))
        print(f"{record['candidate']}  {record['evaluation_percent']}%  F1 {f1:.4f}  MAE {mae:.4f}  {record['label']}")


if __name__ == "__main__":
    main()
