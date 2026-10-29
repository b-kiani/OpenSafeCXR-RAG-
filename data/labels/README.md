# Proxy target-domain labels

Expected files:

- `iu_val_labels.npy`: (746, 7) 0/1
- `iu_test_labels.npy`: (1492, 7) 0/1

These are CheXbert labels of the IU **reference reports**, in the shared-finding order of `docs/DATA.md` (RadEval "rrg" convention: positive or uncertain = 1). They are text-derived proxies, not radiologist image annotations.
