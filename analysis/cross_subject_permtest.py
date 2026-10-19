"""Per-subject permutation test on frozen cross-subject predictions (BH-FDR), all models."""

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parents[1]
M1_RESULTS = BASE / "modeling" / "01_csp_lda" / "results"
M2_RESULTS = BASE / "modeling" / "02_riemannian" / "results"
M3_RESULTS = BASE / "modeling" / "03_eegnet" / "results"
OUT = Path(__file__).resolve().parent / "cross_subject_persubject_significance.csv"

N_PERM = 1000
SEED = 42


def per_subject_perm_test(y_true, score, subject, threshold, n_perm, rng):
    y_pred = (score >= threshold).astype(int)
    rows = []
    for subj in np.unique(subject):
        mask = subject == subj
        yt = y_true[mask]
        yp = y_pred[mask]
        n = len(yt)
        real_acc = (yt == yp).mean()
        perm_accs = np.empty(n_perm)
        for i in range(n_perm):
            shuffled = rng.permutation(yt)
            perm_accs[i] = (shuffled == yp).mean()
        pval = (np.sum(perm_accs >= real_acc) + 1) / (n_perm + 1)
        rows.append((subj, n, real_acc, pval))
    return pd.DataFrame(rows, columns=["subject", "n_test", "acc", "pval"])


def fdr(df):
    df = df.copy()
    df["fdr_sig"] = multipletests(df["pval"], alpha=0.05, method="fdr_bh")[0]
    return df


TASKS = [
    ("T1", "model1_task1_LR_cross_subject_rocdata.npz", "model2_task1_LR_cross_subject_rocdata.npz", "model3_eegnet_cross_task1_LR_rocdata.npz"),
    ("T2", "model1_task2_FF_cross_subject_rocdata.npz", "model2_task2_FF_cross_subject_rocdata.npz", "model3_eegnet_cross_task2_FF_rocdata.npz"),
]

all_results = []
rng = np.random.default_rng(SEED)

for task, m1_file, m2_file, m3_file in TASKS:
    # ── M1 CSP+LDA ──
    d = np.load(M1_RESULTS / m1_file)
    res = per_subject_perm_test(d["y_true"], d["y_score"], d["subject"], threshold=0, n_perm=N_PERM, rng=rng)
    res = fdr(res)
    res["model"], res["task"], res["variant"] = "M1 CSP+LDA", task, "M1-LDA"
    all_results.append(res)

    # ── M2 unaligned (2A) and aligned (2B) ──
    d = np.load(M2_RESULTS / m2_file)
    for scorekey, variant, label in [("m2a_score", "M2A-Riemann", "M2 unaligned (2A)"), ("m2b_score", "M2B-Riemann-aligned", "M2 aligned (2B)")]:
        res = per_subject_perm_test(d["y_true"], d[scorekey], d["subject"], threshold=0, n_perm=N_PERM, rng=rng)
        res = fdr(res)
        res["model"], res["task"], res["variant"] = label, task, variant
        all_results.append(res)

    # ── M3 EEGNet ──
    d = np.load(M3_RESULTS / m3_file)
    res = per_subject_perm_test(d["y_true"], d["eegnet_score"], d["subject"], threshold=0.5, n_perm=N_PERM, rng=rng)
    res = fdr(res)
    res["model"], res["task"], res["variant"] = "M3 EEGNet", task, "M3-EEGNet"
    all_results.append(res)

final = pd.concat(all_results, ignore_index=True)
final = final[["model", "variant", "task", "subject", "n_test", "acc", "pval", "fdr_sig"]]
final.to_csv(OUT, index=False)

print(f"Saved {len(final)} rows to {OUT}")
print("\nFDR-significant fraction per model x task:")
print(final.groupby(["model", "task"])["fdr_sig"].mean().round(3))
