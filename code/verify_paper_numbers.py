#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify every number in the TANGO paper against the released evaluation records.

    python code/verify_paper_numbers.py

Reads only ../results/ (no GPU, no datasets, no model loading) and re-derives each
table cell from the per-seed JSONs, then compares against the values printed in
the paper.  Exits non-zero if anything disagrees.

Conventions, stated once:
  * 'auroc'           - fixed orientation: the detector's published sign, never flipped.
  * 'auroc_oriented'  - empirically correct orientation per (method, score), the
                        convention used by the main paper's Table 1/2 and Fig. 3.
  * far4              - mean over MNIST / SVHN / DTD / Places365, Entropy-scored.
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, "results")
DS = os.path.join(RES, "detector_suite")
GM = os.path.join(RES, "gram_mdsens")
FAR4 = ["mnist", "svhn", "dtd", "places365"]
TOL = 0.0015

fails, checks = [], 0


def _load(idset, method, arch, seed, src):
    p = os.path.join(src, f"{idset}__{method}__{arch}__seed{seed}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def cell(idset, method, split, det, conv="auroc", arch="resnet8x4", seeds=(0, 1, 2)):
    src = GM if det in ("GRAM", "MDSEns") else DS
    v = []
    for sd in seeds:
        r = _load(idset, method, arch, sd, src)
        if r and split in r["ood"] and det in r["ood"][split]:
            v.append(r["ood"][split][det][conv])
    return st.mean(v) if v else None


def far4(idset, method, det="Entropy", conv="auroc", arch="resnet8x4"):
    per = [cell(idset, method, s, det, conv, arch) for s in FAR4]
    return st.mean(per) if all(x is not None for x in per) else None


def idacc(idset, method, arch="resnet8x4", seeds=(0, 1, 2)):
    v = [r["id_acc"] for sd in seeds
         if (r := _load(idset, method, arch, sd, DS)) is not None]
    return st.mean(v) if v else None


def ck(label, printed, got, tol=TOL):
    global checks
    checks += 1
    ok = got is not None and abs(printed - got) <= tol
    if not ok:
        fails.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52} paper {printed:.3f}  "
          f"got {'n/a' if got is None else format(got, '.4f')}")


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- Table 1
head("Table 1  CIFAR-100 ID   (recipe score on SVHN/TIN; Entropy on C-10 and far4)")
T1 = {  # method: (id_acc, svhn, tin, c10, far4)
    "VanillaKD":   (0.733, 0.906, 0.796, 0.749, 0.727),
    "DKD":         (0.728, 0.883, 0.798, 0.754, 0.725),
    "Feat-MSE":    (0.748, 0.902, 0.794, 0.756, 0.752),
    "Cosine-Feat": (0.730, 0.867, 0.782, 0.754, 0.748),
    "LogitNormKD": (0.771, 0.970, 0.776, 0.689, 0.834),
}
for m, (a, sv, tin, c10, f4) in T1.items():
    det = "Entropy" if m == "LogitNormKD" else "Energy"
    conv = "auroc" if m == "LogitNormKD" else "auroc_oriented"
    ck(f"{m} id_acc", a, idacc("cifar100", m))
    ck(f"{m} SVHN ({det})", sv, cell("cifar100", m, "svhn", det, conv))
    ck(f"{m} TIN ({det})", tin, cell("cifar100", m, "tinyimagenet", det, conv))
    ck(f"{m} C-10 (Entropy)", c10, cell("cifar100", m, "cifar10", "Entropy"))
    ck(f"{m} far4 (Entropy)", f4, far4("cifar100", m))

# ---------------------------------------------------------------- Table 2
head("Table 2  CIFAR-10 ID")
T2 = {
    "VanillaKD":   (0.919, 0.966, 0.880, 0.881, 0.887),
    "DKD":         (0.925, 0.970, 0.901, 0.901, 0.918),
    "Feat-MSE":    (0.939, 0.959, 0.904, 0.896, 0.915),
    "Cosine-Feat": (0.931, 0.961, 0.894, 0.884, 0.907),
    "LogitNormKD": (0.934, 0.983, 0.900, 0.885, 0.947),
}
for m, (a, sv, tin, c100, f4) in T2.items():
    det = "Entropy" if m == "LogitNormKD" else "Energy"
    conv = "auroc" if m == "LogitNormKD" else "auroc_oriented"
    ck(f"{m} id_acc", a, idacc("cifar10", m))
    ck(f"{m} SVHN ({det})", sv, cell("cifar10", m, "svhn", det, conv))
    ck(f"{m} TIN ({det})", tin, cell("cifar10", m, "tinyimagenet", det, conv))
    ck(f"{m} C-100 ({det})", c100, cell("cifar10", m, "cifac100" if False else "cifar100", det, conv))
    ck(f"{m} far4 (Entropy)", f4, far4("cifar10", m))

# ---------------------------------------------------------------- Table 3
head("Table 3  per-split far-OOD, Entropy, both ID setups")
T3 = {
    "VanillaKD":   (0.654, 0.662, 0.757, 0.925, 0.846, 0.846),
    "DKD":         (0.629, 0.683, 0.771, 0.964, 0.884, 0.881),
    "Feat-MSE":    (0.728, 0.683, 0.755, 0.957, 0.877, 0.893),
    "Cosine-Feat": (0.749, 0.697, 0.749, 0.957, 0.857, 0.879),
    "LogitNormKD": (0.870, 0.746, 0.749, 0.982, 0.909, 0.915),
}
for m, vals in T3.items():
    for (idset, split), pr in zip([("cifar100", "mnist"), ("cifar100", "dtd"),
                                   ("cifar100", "places365"), ("cifar10", "mnist"),
                                   ("cifar10", "dtd"), ("cifar10", "places365")], vals):
        ck(f"{m} {idset}/{split}", pr, cell(idset, m, split, "Entropy"))

# ---------------------------------------------------------------- Table 4
head("Table 4  decomposition (CIFAR-100, all rows Entropy)")
T4 = {
    "CE":            (0.726, 0.769, 0.775, 0.729),
    "LogitNorm-CE":  (0.736, 0.907, 0.720, 0.767),
    "ELogitNorm-CE": (0.734, 0.942, 0.723, 0.792),
    "VanillaKD":     (0.733, 0.834, 0.776, 0.727),
    "LogitNormKD":   (0.771, 0.970, 0.776, 0.834),
}
for m, (a, sv, tin, f4) in T4.items():
    ck(f"{m} id_acc", a, idacc("cifar100", m))
    ck(f"{m} SVHN", sv, cell("cifar100", m, "svhn", "Entropy"))
    ck(f"{m} TIN", tin, cell("cifar100", m, "tinyimagenet", "Entropy"))
    ck(f"{m} far4", f4, far4("cifar100", m))

# ---------------------------------------------------------------- teacher
head("Teacher (frozen DINOv2-S + linear probe)")
T = json.load(open(os.path.join(RES, "teacher_ood_all.json"), encoding="utf-8"))
for idset, pr in [("cifar100", 0.806), ("cifar10", 0.961)]:
    ck(f"teacher far4 {idset}", pr, st.mean([T[idset][s]["entropy"] for s in FAR4]))
ck("teacher SVHN Entropy (C100)", 0.955, T["cifar100"]["svhn"]["entropy"])
ck("teacher SVHN Energy  (C100)", 0.978, T["cifar100"]["svhn"]["energy_T1"])
ck("teacher MNIST Entropy (C100)", 0.577, T["cifar100"]["mnist"]["entropy"])

# ---------------------------------------------------------------- supp S9.1
head("Supp S9.1  detector families, CIFAR-100 far4, fixed orientation")
S9 = {
    "CE":            (0.709, 0.729, 0.761, 0.759, 0.876, 0.589, 0.822, 0.838, 0.723),
    "LogitNorm-CE":  (0.809, 0.767, 0.771, 0.809, 0.879, 0.393, 0.370, 0.860, 0.586),
    "ELogitNorm-CE": (0.815, 0.792, 0.787, 0.814, 0.891, 0.338, 0.312, 0.856, 0.547),
    "VanillaKD":     (0.702, 0.727, 0.788, 0.782, 0.835, 0.677, 0.813, 0.839, 0.758),
    "DKD":           (0.695, 0.725, 0.787, 0.778, 0.865, 0.694, 0.854, 0.851, 0.757),
    "Feat-MSE":      (0.726, 0.752, 0.798, 0.794, 0.856, 0.654, 0.862, 0.836, 0.735),
    "Cosine-Feat":   (0.723, 0.748, 0.793, 0.789, 0.876, 0.605, 0.858, 0.835, 0.729),
    "LogitNormKD":   (0.826, 0.834, 0.617, 0.827, 0.791, 0.411, 0.361, 0.840, 0.598),
}
DETS = ["MSP", "Entropy", "Energy", "MaxLogit", "KNN", "MDS", "ViM", "GRAM", "MDSEns"]
for m, vals in S9.items():
    for det, pr in zip(DETS, vals):
        ck(f"{m} {det}", pr, far4("cifar100", m, det), 0.002)

# ---------------------------------------------------------------- RMDS
head("Supp S9.1  RMDS (relative Mahalanobis / MDS++), CIFAR-100 far4")
import csv
_r = {}
with open(os.path.join(RES, "posthoc_rmds_cifar100.csv"), encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["ood"] in FAR4:
            _r.setdefault(row["method"], {})[row["ood"]] = float(row["rmds_auroc_mean"])
RM = {"CE": 0.763, "LogitNorm-CE": 0.780, "ELogitNorm-CE": 0.769,
      "VanillaKD": 0.763, "DKD": 0.766, "Feat-MSE": 0.788,
      "Cosine-Feat": 0.765, "LogitNormKD": 0.781}
for m, pr in RM.items():
    v = _r.get(m, {})
    ck(f"{m} RMDS", pr, st.mean(v.values()) if len(v) == 4 else None, 0.002)
a_, b_ = _r.get("VanillaKD"), _r.get("LogitNormKD")
if a_ and b_:
    ck("lift RMDS", 0.018, st.mean(b_.values()) - st.mean(a_.values()), 0.002)

# ---------------------------------------------------------------- coupling
head("Coupling claim  Vanilla KD -> LOGITNORMKD, far4 lift per detector")
EXP = {"MSP": +0.124, "Entropy": +0.107, "MaxLogit": +0.045, "GRAM": +0.002,
       "KNN": -0.044, "MDSEns": -0.160, "Energy": -0.171, "MDS": -0.266, "ViM": -0.452}
for det, pr in EXP.items():
    a, b = far4("cifar100", "VanillaKD", det), far4("cifar100", "LogitNormKD", det)
    ck(f"lift {det}", pr, (b - a) if (a is not None and b is not None) else None, 0.002)

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print(f"{checks - len(fails)}/{checks} checks passed")
if fails:
    print("FAILED:")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("All paper numbers reproduce from the released records.")
