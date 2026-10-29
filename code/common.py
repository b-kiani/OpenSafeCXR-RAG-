"""Shared constants for the OpenSafeCXR-RAG++ release scripts."""

# The seven findings shared by the NIH ChestX-ray14 detector and CheXbert.
# Order here defines the column order of every (n, 7) array these scripts read or write.
SEVEN_NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia",
             "Pneumothorax", "Consolidation", "Edema"]
NIH_TO_CHEXBERT = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
}

# Per-label positivity thresholds of the archived run (paper, Sec. III-B).
NIH_THRESHOLDS_14 = {
    "Atelectasis": 0.70, "Cardiomegaly": 0.90, "Effusion": 0.70,
    "Infiltration": 0.55, "Mass": 0.85, "Nodule": 0.75, "Pneumonia": 0.80,
    "Pneumothorax": 0.85, "Consolidation": 0.75, "Edema": 0.90,
    "Emphysema": 0.90, "Fibrosis": 0.85, "Pleural_Thickening": 0.85,
    "Hernia": 0.95,
}
NIH_THRESHOLDS_7 = [NIH_THRESHOLDS_14[l] for l in SEVEN_NIH]
