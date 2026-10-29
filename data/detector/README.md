# Detector

Expected files:

- `densenet121_nih14.pt`: DenseNet121 state dict, 14 outputs in NIH label order
- `thresholds.yaml`: archived per-label positivity thresholds (included)
- `iu_train_probs14.npy`, `iu_val_probs14.npy`, `iu_test_probs14.npy`: (n, 14) sigmoid probabilities on IU images
- `iu_val_probs.npy`, `iu_test_probs.npy`: (n, 7) subset in the shared-finding order of `docs/DATA.md`

Rows follow `../splits/iu_{split}_uids.txt`. Detector outputs are hypotheses, not findings; see `../../docs/MODEL_CARD.md`.
