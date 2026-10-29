# Reproducibility

This page states which parts of the archived experiment can be reproduced exactly and how the two lost values are handled. It corresponds to Table I and Section III-G of the paper.

## Exactly reproducible

Every number in the paper's Results section is computed from archived artifacts released in `data/`:

| Paper result | Inputs | Command |
|---|---|---|
| Table II (F1-CheXbert, F1-RadGraph) and paired bootstrap CIs | `data/generated/*.csv` | `python code/run_independent_eval.py --dir data/generated` |
| Table III (per-label CheXbert) | `data/generated/*.csv` | same, with RadEval `detailed=True` |
| Table IV and operating-point sweep | `data/detector/iu_test_probs.npy`, `data/labels/iu_test_labels.npy` | `python code/run_sensitivity.py threshold ...` |
| Retrieval-quality grid | `data/retrieval/*` | `python code/run_sensitivity.py retrieval ...` |
| Composite-score ranking statement | archived diagnostics → `templates/composite_components.csv` | `python code/run_composite_sensitivity.py ...` |
| Internal consistency of all printed numbers | none | `python code/verify_reported_numbers.py` |

Expected values for every row are in `reported_results/`. If a re-run differs, the paper is wrong, not the output. Please open an issue.

## Not exactly reproducible

| Value | Why | Handling |
|---|---|---|
| Retrieval weight α of the archived Combined RAG run | Not written to any retained configuration; back-inference from saved retrievals resolved only two cases | Sensitivity grid over α ∈ {0, 0.25, 0.5, 0.75, 1} × K ∈ {1, 3, 5, 10} |
| Confidence cut-off between the "Possible … cannot be excluded" and "… noted" templates | Not present in retained source | Reference configuration uses one cautious template for all labels |

Consequently, **regenerating** the Combined RAG drafts or the Gated RAG++ rewrites from scratch will not match the archived outputs exactly. The archived outputs themselves are released, so every evaluation remains exact.

## Reference configuration

`configs/reference_config.yaml` contains two blocks:

- `archived_run`: every recovered setting. Lost values are `null`; they are never guessed.
- `reference_config`: a fully specified setting for new experiments (α = 0.5, K = 3, single cautious template). Results from it are **not** expected to equal the paper's numbers.

## Recommended before reusing the rewrite

The detector is miscalibrated on IU at its NIH thresholds: seven-label micro-F1 is 0.202, and two findings have zero F1. Select thresholds on the IU validation split before any further use:

1. Run `code/recalibrate_thresholds.py --strategy nih`. It must reproduce micro-F1 0.202. If it does not, the inputs or label convention differ from the archived run.
2. Run with `--strategy precision --min-precision 0.5`, or `--strategy f1`. Thresholds are chosen on validation and frozen; the test split is used only for reporting.

## Software versions used for independent evaluation

- RadGraph 0.1.18, model `radgraph-xl`, `reward_level="all"`
- CheXbert: standard pretrained checkpoint via RadEval
- Paired bootstrap: 10,000 resamples, seed 42
