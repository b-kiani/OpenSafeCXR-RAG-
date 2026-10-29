# Reported results

Every value printed in the paper, as CSV, for checking re-runs against. These files are the *expected* outputs. Re-running the scripts in `code/` on `data/` should reproduce them to the printed precision.

| File | Paper location |
|---|---|
| `table2_independent_evaluation.csv` | Table II (F1-CheXbert, F1-RadGraph; six conditions, n = 1,492) |
| `radgraph_paired_bootstrap.csv` | Sec. IV-A, paired bootstrap of Gated RAG++ minus Combined RAG |
| `table3_per_label_chexbert.csv` | Table III |
| `detector_transfer_iu.csv` | Table IV |
| `operating_points_iu.csv` | Sec. IV-C |
| `retrieval_quality.csv` | Sec. IV-D (Hamming agreement was reported only for the best setting) |
| `table5_internal_diagnostics.csv` | Table V. Rows marked `shares_rewrite_ontology = yes` are not evidence of factuality |
| `composite_score_legacy.csv` | Sec. IV-E, legacy composite (supplementary; not used for any conclusion) |
| `nih_detector_test.csv` | Detector on the NIH test set (n = 25,596) and archived thresholds |

Notes:
- `Retrieval-only` is a non-generative comparator: it returns the nearest training report.
- The NIH mean row (AUROC 0.800, AUPRC 0.267) is the mean of the unrounded per-label values. Averaging the 3-decimal values shown gives 0.8006 for AUROC, which is within rounding.
- IU "proxy" values use CheXbert labels of IU reference reports, not radiologist image labels.
