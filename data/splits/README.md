# Splits

Expected files (one ID per line, sorted):

- `iu_train_uids.txt`: 5,188 IU UIDs
- `iu_val_uids.txt`: 746 IU UIDs
- `iu_test_uids.txt`: 1,492 IU UIDs
- `nih_train_images.txt`, `nih_val_images.txt`, `nih_test_images.txt`: 77,988 / 8,536 / 25,596 NIH image IDs

IU: UID-level split, ratios 70/10/20, seed 42. NIH: official test list; validation patients disjoint from training patients.
