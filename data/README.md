# Derived data

This folder holds derived artifacts only. **No images and no copies of the source datasets.** File formats, label orders and naming are specified in [`../docs/DATA.md`](../docs/DATA.md).

| Folder | Contents |
|---|---|
| `splits/` | UID / image-ID lists for every split |
| `detector/` | detector weights, thresholds, detector probabilities on IU |
| `retrieval/` | image embeddings, abnormality vectors, CheXbert label sets, UIDs |
| `labels/` | CheXbert labels of IU reference reports (7 shared findings) |
| `generated/` | test-split outputs of the six conditions |

Large binary files (`.npy`, `.pt`) are tracked with Git LFS (see `../.gitattributes`). If a file exceeds GitHub LFS limits, attach it to a GitHub Release or Zenodo record and link it in that subfolder's README.
