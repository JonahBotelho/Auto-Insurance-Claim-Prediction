# Claim prediction experiments

These experiments compare 24 detection and payout rules around Group 2's run 1190003 at project commit `a3ea3b5`. Each model fit used an 80/20 stratified split. Seeds 41–45 were used for development and seeds 151–155 for a further check. The split sets reuse the same underlying training records.

| Rule | Development MAE | Development F1 | Additional MAE | Additional F1 |
| --- | ---: | ---: | ---: | ---: |
| Original 1190003 | 92.5004 | 0.52190 | 89.5581 | 0.51467 |
| Stricter cold-profile detection and nearest supported payout within 10% | 92.4916 | 0.52231 | 89.5672 | 0.51367 |

The development gain reversed on the additional splits. `competition/test_predictions.csv` uses the original 1190003 rule fitted on all 70,000 labeled records and preserves the order of the 30,000 rows in `data/testset.csv`. `competition/summary.csv` contains the mean scores and `competition/per_seed.csv` contains each split. The simpler comparison model is recorded in `alternative_scores.csv`.

## Reproduce from the repository root

Use the project's optional training dependencies. These commands write results under this experiment directory. The per-row component files are generated during fitting.

```powershell
python analysis/claim_runs/compare_competition.py . --seeds 41 42 43 44 45 151 152 153 154 155
python analysis/claim_runs/summarize_competition.py . analysis/claim_runs/competition
python analysis/claim_runs/compare_competition.py . --test-components
python analysis/claim_runs/summarize_competition.py . analysis/claim_runs/competition --export-rule original original
python analysis/claim_runs/alternative.py --repo . --out analysis/claim_runs
```
