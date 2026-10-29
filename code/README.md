# OpenSafeCXR-RAG++ — evaluation, recalibration and reproducibility code

Companion code for *OpenSafeCXR-RAG++: Detector-Constrained Retrieval-Augmented Structured
Chest X-ray Report Generation with Independent Factuality Evaluation* (ICTACS 2026).

## What is exactly reproducible

| Paper result | Reproducible from released files? | Script |
|---|---|---|
| Table II (F1-CheXbert, F1-RadGraph) and paired bootstrap CIs | Yes, from archived generated reports | `run_independent_eval.py` |
| Table III (per-label CheXbert) | Yes (RadEval `detailed=True`) | `run_independent_eval.py` |
| Table IV, operating-point sweep (Sec. IV-C) | Yes, from archived detector probabilities | `run_sensitivity.py threshold` |
| Retrieval-quality grid (Sec. IV-D) | Yes, from archived embeddings and probabilities | `run_sensitivity.py retrieval` |
| Composite ranking statement (Sec. IV-E) | Yes, given the archived diagnostics CSV | `run_composite_sensitivity.py` |
| *Regenerating* Combined RAG drafts | **No** — the archived α was not stored | use `reference_config` |
| *Regenerating* Gated RAG++ rewrites | **No** — template confidence cut-off not stored | use `reference_config` |

`reference_config.yaml` records every recovered value of the archived run, marks the two
lost values as `null`, and defines a fully specified reference configuration for new work.

## Scripts

| Script | Purpose |
|---|---|
| `common.py` | Label order, NIH→CheXbert name mapping, archived thresholds |
| `label_reports_chexbert.py` | CheXbert labels (7 shared findings) for IU reference reports |
| `recalibrate_thresholds.py` | Select detector thresholds on IU **validation**, freeze, evaluate on test |
| `run_independent_eval.py` | F1-CheXbert (RadEval ≥ 2.2) and official F1-RadGraph 0.1.18 + bootstrap |
| `run_sensitivity.py` | Retrieval α/K grid; global-threshold operating-point sweep |
| `run_composite_sensitivity.py` | Legacy composite, equal weights, ±25% perturbation ranking |
| `verify_reported_numbers.py` | Internal-consistency audit of the numbers printed in the paper |

## Target-domain recalibration (recommended before any further use of the rewrite)

```bash
pip install "radeval>=2.2" numpy pandas scikit-learn
python label_reports_chexbert.py --csv iu_val.csv  --out iu_val_labels.npy
python label_reports_chexbert.py --csv iu_test.csv --out iu_test_labels.npy

# 1) sanity check: must reproduce Table IV (micro-F1 0.202 at NIH thresholds)
python recalibrate_thresholds.py --val-probs iu_val_probs.npy --val-labels iu_val_labels.npy \
    --test-probs iu_test_probs.npy --test-labels iu_test_labels.npy --strategy nih

# 2) recalibrate: precision-constrained is the safer choice for a report-editing rule
python recalibrate_thresholds.py ... --strategy precision --min-precision 0.5 --out-dir recal_p50
python recalibrate_thresholds.py ... --strategy f1 --calibrate platt --out-dir recal_f1
```

Each run writes `recalibration_snippet.tex` with a table and paragraph filled from that run.
If step 1 does not reproduce 0.202, the label convention or inputs differ from the archived
run (try `--uncertain-negative` in the labelling step) — resolve that before reporting.

Note: recalibration changes only the detector decisions. To measure its effect on generated
reports, rerun the rewrite with the new thresholds under `reference_config` and score the
output with `run_independent_eval.py`; those numbers are not comparable one-to-one with the
archived Gated RAG++ row because the template cut-off differs.

## Data
IU X-ray/OpenI and NIH ChestX-ray14 must be obtained from their providers. This repository
releases only derived artifacts (split ID lists, detector weights/outputs, embeddings,
generated reports, per-case scores).
