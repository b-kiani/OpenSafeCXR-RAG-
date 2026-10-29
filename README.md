# OpenSafeCXR-RAG++

**Detector-Constrained Retrieval-Augmented Structured Chest X-ray Report Generation with Independent Factuality Evaluation**

Behnam Kiani Kalejahi, Sajid Khan, Adnan Manzoor, Eugene Castro, Kamila Atakhodjaeva
Department of Computer Science, School of Engineering, Central Asian University, Tashkent, Uzbekistan
*ICTACS 2026* · [Paper (accepted version)](paper/OpenSafeCXR-RAG_ICTACS2026.pdf)

---

## Overview

OpenSafeCXR-RAG++ is an interpretable pipeline for structured chest X-ray reporting built only on open data:

1. A **DenseNet121 detector** (14 labels, trained on NIH ChestX-ray14) produces an abnormality probability vector.
2. **Top-3 retrieval** over an IU X-ray training-only bank uses image embeddings and detector probabilities, excluding the query's own UID.
3. **T5-base** writes a structured *Findings* + *Impression* draft.
4. A **deterministic rewrite** removes abnormality statements the detector does not support and adds cautious statements for detector-positive labels the draft omits.

The repository supports the paper's evaluation: independent F1-CheXbert and official F1-RadGraph (RadGraph-XL) over six controlled conditions on one 1,492-record held-out IU test set.

## Main finding

The rewrite is a **trade-off, not an improvement**. Compared with ungated Combined RAG:

| Metric | Combined RAG | Gated RAG++ |
|---|---|---|
| F1-CheXbert, 5-label micro / macro | 0.028 / 0.013 | **0.219 / 0.252** |
| F1-CheXbert, all-label micro | **0.293** | 0.167 |
| F1-RadGraph entity / entity-relation / complete | **0.273 / 0.261 / 0.191** | 0.183 / 0.167 / 0.133 |

Paired bootstrap (10,000 resamples) 95% CIs for all three RadGraph differences exclude zero. At its NIH thresholds, the detector reaches a seven-label micro-F1 of only 0.202 against IU report-derived labels, with zero F1 for pneumothorax and consolidation.

**The rewrite does not establish clinical factuality or safety.** No radiologist evaluation has been performed. Nothing in this repository is intended for clinical use.

All reported numbers are in [`reported_results/`](reported_results/).

## Repository layout

```
.
├── README.md                  this file
├── LICENSE                    MIT (code and documentation)
├── CITATION.cff               citation metadata
├── requirements.txt           Python dependencies
├── configs/
│   └── reference_config.yaml  archived-run record + reference configuration
├── code/                      evaluation, recalibration and audit scripts
├── data/                      derived artifacts (see data/README.md); no images
│   ├── splits/                UID lists for train / val / test
│   ├── detector/              detector weights, thresholds, IU probabilities
│   ├── retrieval/             image embeddings, retrieval label sets
│   ├── labels/                CheXbert labels of IU reference reports
│   └── generated/             generated reports for the six conditions
├── reported_results/          every table and figure value in the paper, as CSV
├── templates/                 input templates (composite score components)
├── docs/
│   ├── DATA.md                dataset access and file formats
│   ├── REPRODUCIBILITY.md     what is and is not exactly reproducible
│   └── MODEL_CARD.md          detector card, intended use, limitations
└── paper/                     accepted manuscript (PDF)
```

## Quick start

```bash
git clone <this-repository>
cd OpenSafeCXR-RAG-plus-plus
pip install -r requirements.txt

# 1. Audit the numbers printed in the paper (no data needed)
python code/verify_reported_numbers.py

# 2. Recompute Table II and the bootstrap CIs from the archived generated reports
python code/run_independent_eval.py --dir data/generated --out results

# 3. Recompute Table IV and the operating-point sweep
python code/run_sensitivity.py threshold \
    --probs data/detector/iu_test_probs.npy --labels data/labels/iu_test_labels.npy

# 4. Recalibrate detector thresholds on the IU validation split (recommended)
python code/recalibrate_thresholds.py \
    --val-probs  data/detector/iu_val_probs.npy  --val-labels  data/labels/iu_val_labels.npy \
    --test-probs data/detector/iu_test_probs.npy --test-labels data/labels/iu_test_labels.npy \
    --strategy precision --min-precision 0.5 --out-dir results/recal
```

The images and original reports must be obtained from the dataset providers; see [`docs/DATA.md`](docs/DATA.md).

## Reproducibility status

| Element | Status |
|---|---|
| Data splits, detector weights and thresholds, embeddings, detector probabilities | Exact (released) |
| Retrieval depth K = 3, T5 training and decoding settings | Exact |
| Generated reports for all six conditions; all metrics in the paper | Exact (re-runnable) |
| Retrieval weight α used for Combined RAG | **Lost**: covered by a sensitivity grid |
| Confidence cut-off between the two rewrite templates | **Lost**: reference configuration provided |

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for details.

## Citation

```bibtex
@inproceedings{kiani2026opensafecxr,
  author    = {Behnam Kiani Kalejahi and Sajid Khan and Adnan Manzoor and Eugene Castro and Kamila Atakhodjaeva},
  title     = {{OpenSafeCXR-RAG++}: Detector-Constrained Retrieval-Augmented Structured Chest {X-ray} Report Generation with Independent Factuality Evaluation},
  booktitle = {Proc. ICTACS 2026},
  year      = {2026}
}
```

## License and data terms

Code and documentation in this repository are released under the [MIT License](LICENSE). The NIH ChestX-ray14 and IU X-ray/OpenI datasets are **not** redistributed here; they remain under their providers' terms. Derived artifacts in `data/` are provided for research use only and must be used in accordance with those terms.

## Contact

Behnam Kiani Kalejahi: b.kiani@centralasian.uz
