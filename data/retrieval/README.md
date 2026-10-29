# Retrieval artifacts

Expected files (train = 5,188 bank records, test = 1,492 queries):

- `train_img.npy`, `test_img.npy`: (n, 1024) DenseNet121 image embeddings
- `train_abn.npy`, `test_abn.npy`: (n, 14) detector probability vectors used in the retrieval score
- `train_chexbert.npy`, `test_chexbert.npy`: (n, 14) 0/1 CheXbert label sets of the IU reports (retrieval-quality metric)
- `train_uid.npy`, `test_uid.npy`: (n,) UIDs; candidates sharing the query UID are excluded

Score: `s = α·cos(v_q, v_j) + (1 − α)·cos(e_q, e_j)`, top K = 3. The α of the archived Combined RAG run was not stored; see `../../docs/REPRODUCIBILITY.md`.
