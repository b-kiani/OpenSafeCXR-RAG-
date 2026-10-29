# Data access and file formats

## Source datasets (not redistributed)

| Dataset | Used for | Obtain from |
|---|---|---|
| NIH ChestX-ray14 | Training and testing the 14-label detector | NIH Clinical Center (ChestX-ray14 release) |
| IU X-ray / OpenI | Retrieval bank, report generation and evaluation | Open-i, U.S. National Library of Medicine |

No images and no source-dataset files are stored in this repository. Download both datasets from their providers and follow their terms of use.

**Splits.** IU records were split at UID level (70/10/20, seed 42) into 5,188 train, 746 validation and 1,492 test records. NIH used the official test split, with a patient-disjoint validation split taken from the training portion (77,988 / 8,536 / 25,596 images). The UID lists in `data/splits/` reproduce the IU split exactly.

## Derived artifacts in `data/`

All arrays are NumPy `.npy`. Rows are aligned with the matching UID file in the same folder. The *seven shared findings* are always stored in this column order:

`Atelectasis, Cardiomegaly, Effusion, Pneumonia, Pneumothorax, Consolidation, Edema`

The 14 detector labels use NIH order:

`Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia`

| Path | Shape / columns | Content |
|---|---|---|
| `splits/iu_{train,val,test}_uids.txt` | one UID per line | IU split membership |
| `splits/nih_{train,val,test}_images.txt` | one image ID per line | NIH split membership |
| `detector/densenet121_nih14.pt` | PyTorch state dict | detector weights |
| `detector/thresholds.yaml` | 14 values | archived positivity thresholds |
| `detector/iu_{train,val,test}_probs14.npy` | (n, 14) float | detector probabilities, NIH label order |
| `detector/iu_{val,test}_probs.npy` | (n, 7) float | the seven shared findings only |
| `retrieval/{train,test}_img.npy` | (n, 1024) float | DenseNet121 image embeddings |
| `retrieval/{train,test}_abn.npy` | (n, 14) float | abnormality vectors used for retrieval |
| `retrieval/{train,test}_chexbert.npy` | (n, 14) 0/1 | CheXbert label sets of the IU reports, for retrieval quality |
| `retrieval/{train,test}_uid.npy` | (n,) str | UIDs for same-UID exclusion |
| `labels/iu_{val,test}_labels.npy` | (n, 7) 0/1 | CheXbert labels of IU reference reports (proxy target labels) |
| `generated/<condition>.csv` | `uid,generated,reference` | outputs of the six conditions on the test split |

Condition file names: `abnormality_only_rag.csv`, `image_only_rag.csv`, `combined_rag.csv`, `no_retrieval.csv`, `retrieval_only.csv`, `gated_ragpp.csv`.

## Report text

`generated/*.csv` contains model outputs in `generated` and the IU reference report in `reference`. If the IU terms of use do not permit redistributing report text, release `uid,generated` only. Users then recreate the `reference` column from their own copy of IU X-ray by joining on `uid`. The evaluation script refuses to run if any condition's references differ, so a mismatched join is detected automatically.

## Label convention

The proxy labels in `labels/` come from CheXbert applied to the IU *reference* reports. In the RadEval "rrg" convention, positive and uncertain map to 1, and negative and blank map to 0. They are text-derived and are **not** radiologist image annotations.
