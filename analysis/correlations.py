"""Review all training-variable pairs and export scatterplots and a local gallery."""

import argparse
import html
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


def load_data(source):
    data = pd.read_csv(source).drop(columns="rowIndex", errors="ignore")
    if len(data.columns) < 2 or not all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise ValueError("Expected at least two numeric columns after excluding rowIndex.")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Missing or infinite values require review before correlation analysis.")
    if (data.nunique() < 2).any():
        raise ValueError("Constant columns require review before correlation analysis.")
    return data


def summarize_pairs(data):
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    features = data.drop(columns="ClaimAmount", errors="ignore")
    unique_profiles = features.drop_duplicates()
    unique_pearson = unique_profiles.corr()
    records = []
    for first, second in combinations(data.columns, 2):
        coefficient = pearson.loc[first, second]
        ranked = spearman.loc[first, second]
        deduplicated = (unique_pearson.loc[first, second]
                        if first in features and second in features else np.nan)
        records.append({"first": first, "second": second, "rows": len(data),
                        "pearson": coefficient, "spearman": ranked,
                        "max_abs_correlation": max(abs(coefficient), abs(ranked)),
                        "pearson_unique_feature_profiles": deduplicated})
    pairs = pd.DataFrame(records).sort_values("max_abs_correlation", ascending=False)
    return pearson, spearman, pairs


def plot_pair(axis, data, first, second, log_target=False):
    # Collapse coincident coordinates without sampling or inventing jitter.
    points = data.groupby([first, second], sort=False).size().reset_index(name="rows")
    horizontal = points[first]
    vertical = points[second]
    if log_target:
        if first == "ClaimAmount":
            horizontal = np.log1p(horizontal)
        if second == "ClaimAmount":
            vertical = np.log1p(vertical)
    scatter = axis.scatter(horizontal, vertical, c=points.rows, s=9, alpha=0.75,
                           norm=LogNorm(vmin=1, vmax=max(2, int(points.rows.max()))),
                           cmap="viridis", linewidths=0, rasterized=True)
    axis.set_xlabel(f"log1p({first})" if log_target and first == "ClaimAmount" else first)
    axis.set_ylabel(f"log1p({second})" if log_target and second == "ClaimAmount" else second)
    axis.grid(alpha=0.16)
    axis.set_axisbelow(True)
    return scatter


def export_pair(data, pair, destination):
    first, second = pair.first, pair.second
    has_target = "ClaimAmount" in (first, second) and data["ClaimAmount"].min() >= 0
    figure, axes = plt.subplots(1, 2 if has_target else 1,
                                figsize=(12, 4.8) if has_target else (7.2, 5.3),
                                squeeze=False, layout="constrained")
    for index, axis in enumerate(axes.flat):
        scatter = plot_pair(axis, data, first, second, log_target=index == 1)
        figure.colorbar(scatter, ax=axis, label="Rows at this coordinate (log color scale)")
        axis.set_title("Log view of target" if index else "Original values")
    figure.suptitle(f"{first} vs {second} | n={len(data):,}\n"
                   f"Pearson r={pair.pearson:.3f} | Spearman rho={pair.spearman:.3f}",
                   fontsize=12)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def export_overviews(data, pearson, spearman, pairs, output):
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    for axis, matrix, title in zip(axes, [pearson, spearman], ["Pearson (linear)", "Spearman (rank)"]):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=8)
        axis.set_yticks(range(len(matrix)), matrix.columns, fontsize=8)
        axis.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.7, label="Correlation")
    figure.suptitle(f"Training data: {len(data):,} rows; rowIndex excluded")
    figure.savefig(output / "correlation_heatmaps.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for axis, pair in zip(axes.flat, pairs.head(6).itertuples()):
        plot_pair(axis, data, pair.first, pair.second)
        axis.set_title(f"r={pair.pearson:.3f}, rho={pair.spearman:.3f}")
    figure.suptitle("Six strongest pairs by absolute Pearson or Spearman correlation\n"
                   "All rows included; color shows frequency at each coordinate")
    figure.savefig(output / "strongest_pairs.png", dpi=140)
    plt.close(figure)


def export_gallery(data, pairs, output):
    cards = []
    for pair in pairs.itertuples():
        filename = f"pairs/{pair.first}__{pair.second}.png"
        label = html.escape(f"{pair.first} vs {pair.second}")
        cards.append(f'<article data-features="{html.escape(pair.first)} {html.escape(pair.second)}" '
                     f'data-strength="{pair.max_abs_correlation}"><h2>{label}</h2>'
                     f'<p>Pearson {pair.pearson:.3f} · Spearman {pair.spearman:.3f}</p>'
                     f'<a href="{filename}"><img loading="lazy" src="{filename}" '
                     f'alt="Scatterplot: {label}"></a></article>')
    options = ''.join(f'<option>{html.escape(column)}</option>' for column in data.columns)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training data correlations</title><style>
body{font:16px system-ui;margin:32px;background:#f6f7fa;color:#18212d}
main{max-width:1500px;margin:auto}h1{font-size:28px}h2{font-size:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}
article{background:white;padding:18px;border:1px solid #dbe0e8;border-radius:8px}
img{width:100%;height:auto}select{padding:8px;margin:8px 20px 20px 8px}
article[hidden]{display:none}a{color:#204e91}</style><main>
<h1>Training data correlations</h1>
<p>COUNT pair plots from ROWS training rows, sorted by maximum absolute Pearson or Spearman
correlation. rowIndex is excluded. All rows contribute to correlations and plots.</p>
<p>Coincident points are collapsed and colored by row count. Integer codes may represent
categories: their correlation depends on the coding. Target pairs include an additional
log1p view; displayed correlations always use the original values.</p>
<p><a href="correlation_heatmaps.png">Correlation heatmaps</a> ·
<a href="strongest_pairs.png">Six strongest pairs</a> ·
<a href="pair_correlations.csv">All coefficients (CSV)</a></p>
<label>Variable<select id="variable"><option value="">All</option>OPTIONS</select></label>
<label>Minimum |correlation|<select id="strength"><option value="0">All</option>
<option value="0.5">0.5</option><option value="0.7">0.7</option>
<option value="0.9">0.9</option></select></label><span id="count"></span>
<div class="grid">CARDS</div></main><script>
const variable=document.querySelector('#variable'),strength=document.querySelector('#strength');
function filter(){let count=0;document.querySelectorAll('article').forEach(card=>{
const match=(!variable.value||card.dataset.features.split(' ').includes(variable.value))
&&Number(card.dataset.strength)>=Number(strength.value);card.hidden=!match;if(match)count++;
});document.querySelector('#count').textContent=count+' pairs shown';}
variable.onchange=filter;strength.onchange=filter;filter();</script></html>'''
    page = page.replace("COUNT", str(len(pairs))).replace("ROWS", f"{len(data):,}")
    page = page.replace("OPTIONS", options).replace("CARDS", '\n'.join(cards))
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=repository / "trainingset.csv")
    parser.add_argument("--output", type=Path, default=repository / "analysis" / "output")
    arguments = parser.parse_args()
    data = load_data(arguments.input)
    pearson, spearman, pairs = summarize_pairs(data)
    (arguments.output / "pairs").mkdir(parents=True, exist_ok=True)
    pairs.to_csv(arguments.output / "pair_correlations.csv", index=False)
    pearson.to_csv(arguments.output / "pearson_matrix.csv")
    spearman.to_csv(arguments.output / "spearman_matrix.csv")
    export_overviews(data, pearson, spearman, pairs, arguments.output)
    for index, pair in enumerate(pairs.itertuples(), start=1):
        export_pair(data, pair, arguments.output / "pairs" / f"{pair.first}__{pair.second}.png")
        if index % 25 == 0:
            print(f"Generated {index}/{len(pairs)} pair plots", flush=True)
    export_gallery(data, pairs, arguments.output)
    print(pairs.head(10).to_string(index=False))
    print(f"Gallery: {(arguments.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
