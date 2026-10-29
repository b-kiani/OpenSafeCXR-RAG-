# Generated reports (IU test split, n = 1,492)

Expected files, each with columns `uid,generated,reference`:

| File | Condition |
|---|---|
| `abnormality_only_rag.csv` | retrieval on detector vector only (α = 1) |
| `image_only_rag.csv` | retrieval on image embedding only (α = 0) |
| `combined_rag.csv` | combined retrieval (archived α, not recoverable) |
| `no_retrieval.csv` | detector description prompt, no retrieved reports |
| `retrieval_only.csv` | nearest training report returned unchanged (non-generative) |
| `gated_ragpp.csv` | Combined RAG followed by the deterministic rewrite |

All six files must share identical `uid` and `reference` columns; the evaluation script checks this. If report text cannot be redistributed, publish `uid,generated` only (see `docs/DATA.md`).
