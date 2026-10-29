# Model card: DenseNet121 abnormality detector and Gated RAG++ rewrite

> **Research use only. Not a medical device. Not for clinical decision-making.**

## Model details

| Item | Value |
|---|---|
| Architecture | DenseNet121, ImageNet-pretrained, 14 sigmoid outputs |
| Training data | NIH ChestX-ray14 (77,988 train / 8,536 val images; patient-disjoint) |
| Loss / optimiser | Per-label binary cross-entropy; AdamW, lr 1e-4, weight decay 1e-4 |
| Epochs | 12, early stopping on validation mean AUROC |
| Report generator | T5-base fine-tuned on 5,188 IU X-ray training reports |
| Rewrite | Deterministic rule driven by detector thresholds (see paper Sec. III-E) |

## Performance on the NIH test set (n = 25,596)

Mean AUROC 0.800, mean AUPRC 0.267. Per-label values are in `reported_results/nih_detector_test.csv`. The weakest AUPRC values are pneumonia (0.050) and fibrosis (0.098).

## Performance after transfer to IU X-ray

Against CheXbert labels derived from IU reference reports, at the archived NIH thresholds:

- Seven-label micro precision 0.292, recall 0.155, F1 0.202
- **Zero F1 for pneumothorax and consolidation**
- About 0.141 unsupported-positive and 0.318 missed labels per case

Pneumothorax AUPRC falls from 0.375 (NIH) to 0.017 (IU proxy). Per-label values are in `reported_results/detector_transfer_iu.csv`.

## Effect of the rewrite on reports

Compared with ungated Combined RAG, the rewrite raised five-label F1-CheXbert (micro 0.028 → 0.219). It lowered all-label F1-CheXbert (0.293 → 0.167) and all three RadGraph rewards to the lowest values among the six evaluated conditions. Consolidation is inserted at recall 0.769 with precision 0.013, about 76 incorrect mentions for each correct one.

## Intended use

- Studying how detector-constrained post-editing changes report errors
- Benchmarking factuality metrics and detector transfer
- Baseline for recalibration and verification methods

## Out-of-scope use

- Any clinical, diagnostic or triage use
- Deployment on a new dataset without recalibrating thresholds on that dataset
- Treating detector outputs as findings; they are hypotheses

## Known limitations

- Detector trained on weak NLP-mined NIH labels; transfer to IU is uneven
- Target-domain labels are report-derived proxies, not image annotations
- Detector false positives are inserted into reports, and false negatives can delete correct statements
- Detector-defined omission rate after rewriting remains 0.735
- No radiologist evaluation; no external report-generation dataset
- Two archived settings (retrieval α, template cut-off) are lost; see `docs/REPRODUCIBILITY.md`
