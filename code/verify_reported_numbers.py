#!/usr/bin/env python3
"""
verify_reported_numbers.py -- internal-consistency audit of the numbers and comparative
claims printed in the camera-ready paper. Needs no data. Exit code 1 on any failure.
"""
import sys

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def close(a, b, tol=0.0015):
    return abs(a - b) <= tol


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ---------------- Table II
T2 = {  # cond: (5mic, 5mac, allmic, ent, er, comp)
    "Abnormality-only RAG": (0.000, 0.000, 0.324, 0.347, 0.328, 0.200),
    "Image-only RAG":       (0.004, 0.002, 0.321, 0.346, 0.314, 0.230),
    "Combined RAG":         (0.028, 0.013, 0.293, 0.273, 0.261, 0.191),
    "No-retrieval":         (0.051, 0.023, 0.334, 0.236, 0.230, 0.188),
    "Retrieval-only":       (0.270, 0.209, 0.297, 0.262, 0.237, 0.169),
    "Gated RAG++":          (0.219, 0.252, 0.167, 0.183, 0.167, 0.133),
}
W5 = {"Combined RAG": 0.027, "Gated RAG++": 0.325, "Retrieval-only": 0.270,
      "Abnormality-only RAG": 0.000, "Image-only RAG": 0.004, "No-retrieval": 0.049}
BOLD = {0: "Retrieval-only", 1: "Gated RAG++", 2: "No-retrieval",
        3: "Abnormality-only RAG", 4: "Abnormality-only RAG", 5: "Image-only RAG"}

print("\n=== Table II: bold marks and comparative claims")
for col, who in BOLD.items():
    best = max(T2, key=lambda c: T2[c][col])
    check(f"bold in column {col} is the maximum", best == who, f"max={best}")
for col, nm in [(2, "all-label micro"), (3, "RG entity"), (4, "RG E+R"), (5, "RG complete")]:
    check(f"Gated RAG++ lowest on {nm}", min(T2, key=lambda c: T2[c][col]) == "Gated RAG++")
check("Retrieval-only > Gated on 5-label micro (0.270 vs 0.219)", T2["Retrieval-only"][0] > T2["Gated RAG++"][0])
check("Gated leads 5-label macro", max(T2, key=lambda c: T2[c][1]) == "Gated RAG++")
check("Gated leads 5-label weighted", max(W5, key=W5.get) == "Gated RAG++")

print("\n=== Table III: F1 = harmonic mean of P, R; macro-F1 consistency with Table II")
T3 = {"Cardiomegaly": (0.636, 0.033, 0.063, 0.424, 0.290, 0.345),
      "Edema": (0, 0, 0, 0.115, 0.600, 0.194),
      "Consolidation": (0, 0, 0, 0.013, 0.769, 0.026),
      "Atelectasis": (0, 0, 0, 0.259, 0.556, 0.354),
      "Pleural effusion": (0, 0, 0, 0.215, 0.838, 0.342)}
for k, v in T3.items():
    check(f"{k} combined F1", close(f1(v[0], v[1]), v[2]))
    check(f"{k} gated F1", close(f1(v[3], v[4]), v[5]))
check("combined 5-label macro = mean per-label F1", close(sum(v[2] for v in T3.values()) / 5, 0.013))
check("gated 5-label macro = mean per-label F1", close(sum(v[5] for v in T3.values()) / 5, 0.252))
r = 1 / 0.013 - 1
check("'about 76 incorrect mentions per correct one' (precision 0.013)", 74 <= r <= 78, f"{r:.1f}")

print("\n=== Table IV and Sec. IV-C claims")
T4 = {"Atelectasis": (0.756, 0.767, 0.320, 0.300), "Cardiomegaly": (0.894, 0.707, 0.349, 0.373),
      "Effusion": (0.815, 0.908, 0.488, 0.454), "Pneumonia": (0.695, 0.691, 0.050, 0.100),
      "Pneumothorax": (0.833, 0.615, 0.375, 0.017), "Consolidation": (0.732, 0.760, 0.147, 0.030),
      "Edema": (0.835, 0.811, 0.155, 0.198)}
both_worse = [k for k, (a, b, c, d) in T4.items() if b < a and d < c]
check("Pneumothorax is the only finding worse on both AUROC and AUPRC", both_worse == ["Pneumothorax"], str(both_worse))
check("Cardiomegaly: AUROC down, AUPRC up", T4["Cardiomegaly"][1] < T4["Cardiomegaly"][0] and T4["Cardiomegaly"][3] > T4["Cardiomegaly"][2])
check("Consolidation: AUROC up, AUPRC down", T4["Consolidation"][1] > T4["Consolidation"][0] and T4["Consolidation"][3] < T4["Consolidation"][2])
check("micro-F1 at NIH thresholds 0.202 from P=0.292, R=0.155", close(f1(0.292, 0.155), 0.202))
check("micro-F1 at global 0.65 is 0.236 from P=0.204, R=0.279", close(f1(0.204, 0.279), 0.236))
TH = {"Atelectasis": 0.70, "Cardiomegaly": 0.90, "Effusion": 0.70, "Pneumonia": 0.80,
      "Pneumothorax": 0.85, "Consolidation": 0.75, "Edema": 0.90}
TEXT_TH = dict(atelectasis=0.70, cardiomegaly=0.90, effusion=0.70, pneumonia=0.80,
               pneumothorax=0.85, consolidation=0.75, edema=0.90)
check("Table IV thresholds match Sec. III-B list", all(TH[k] == TEXT_TH[k.lower()] for k in TH))

print("\n=== Sec. IV-D retrieval")
J = [0.3046, 0.3019, 0.2909, 0.2883, 0.2669]
check("Jaccard falls steadily with alpha", all(J[i] > J[i + 1] for i in range(4)))
check("best grid value 0.3063 >= every K=3 value", 0.3063 >= max(J))

print("\n=== Sec. IV-E internal diagnostics")
check("748 identical top outputs consistent with repetition 0.501", close(748 / 1492, 0.501, 0.001), f"{748/1492:.4f}")
check("'ROUGE-L nearly halves' (0.307 -> 0.163)", 0.45 <= 0.163 / 0.307 <= 0.56, f"ratio {0.163/0.307:.2f}")

print("\n=== Bootstrap CIs contain their point estimates and exclude zero")
for nm, d, lo, hi in [("entity", -0.0891, -0.0983, -0.0801), ("E+R", -0.0939, -0.1027, -0.0851),
                      ("complete", -0.0581, -0.0670, -0.0497)]:
    check(f"{nm}: lo < delta < hi < 0", lo < d < hi < 0)
for nm, base, gated, d in [("entity", 0.273, 0.183, -0.0891), ("E+R", 0.261, 0.167, -0.0939),
                           ("complete", 0.191, 0.133, -0.0581)]:
    check(f"{nm}: table difference ~ bootstrap delta", abs((gated - base) - d) < 0.0015,
          f"{gated - base:+.4f} vs {d:+.4f}")

print("\n=== Data split arithmetic")
check("5188 + 746 + 1492 = 7426", 5188 + 746 + 1492 == 7426)
check("70/10/20 split", abs(5188 / 7426 - 0.70) < 0.005 and abs(1492 / 7426 - 0.20) < 0.005)

print("\n" + "=" * 60)
if FAIL:
    print(f"AUDIT FAILED ({len(FAIL)}):"); [print("  -", f) for f in FAIL]; sys.exit(1)
print("AUDIT PASSED: every checkable number and comparative claim is consistent.")
