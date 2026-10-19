#!/usr/bin/env python
"""Model 1 (CSP+LDA): executed -> imagined transfer, Task 2."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, SUBJECTS, TRANSFER, FMIN, FMAX, TMIN, TMAX,
                             N_CSP, N_PERM_TRANSFER)

RESULTS_DIR = M1_DIR
TRAIN_RUNS  = TRANSFER["task2_FF"]["train"]
TEST_RUNS   = TRANSFER["task2_FF"]["test"]
N_PERM      = N_PERM_TRANSFER
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var

# ── helpers ────────────────────────────────────────────────────────────────────
def load_runs(subject, runs):
    Xs, ys, gs = [], [], []
    for run in runs:
        path = EPOCHS_DIR / f"S{subject:03d}_R{run:02d}-epo.fif"
        if not path.exists():
            continue
        ep = mne.read_epochs(path, preload=True, verbose=False)
        ep.filter(FMIN, FMAX, picks="eeg", verbose=False)
        ep.crop(tmin=TMIN, tmax=TMAX)
        eeg = mne.pick_types(ep.info, eeg=True)
        Xs.append(ep.get_data(picks=eeg))
        ys.append((ep.events[:, -1] == ep.event_id["T2"]).astype(int))
        gs.append(np.full(len(ep), run, dtype=int))
    if not Xs:
        return None, None, None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)

def make_clf():
    return Pipeline([
        ("csp", CSP(n_components=N_CSP, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

# ── per-subject worker ─────────────────────────────────────────────────────────
def process_subject(subj):
    print(f"S{subj:03d}  starting...", flush=True)
    try:
        Xtr, ytr, _   = load_runs(subj, TRAIN_RUNS)
        Xte, yte, gte = load_runs(subj, TEST_RUNS)
        if Xtr is None or Xte is None:
            print(f"S{subj:03d}  no-data", flush=True)
            return None

        # ── train on executed, predict imagined ───────────────────────────────
        clf   = make_clf().fit(Xtr, ytr)
        yp    = clf.predict(Xte)
        ys    = clf.decision_function(Xte)

        # ── per imagined run ──────────────────────────────────────────────────
        per_run = {}
        for run in TEST_RUNS:
            m = gte == run
            per_run[f"acc_r{run}"] = round(float((clf.predict(Xte[m]) == yte[m]).mean()), 4) if m.any() else float("nan")

        # ── metrics ──────────────────────────────────────────────────────────
        acc     = accuracy_score(yte, yp)
        bal_acc = balanced_accuracy_score(yte, yp)
        auc     = roc_auc_score(yte, ys)
        kappa   = cohen_kappa_score(yte, yp)
        tn, fp, fn, tp_v = confusion_matrix(yte, yp).ravel()
        sens    = tp_v / (tp_v + fn)
        spec    = tn  / (tn  + fp)

        # ── permutation test: permute TEST labels only (model stays fixed) ───
        rng = np.random.default_rng(42 + subj)
        count = 0
        for _ in range(N_PERM):
            yte_perm = rng.permutation(yte)
            if accuracy_score(yte_perm, yp) >= acc:
                count += 1
        pval = (count + 1) / (N_PERM + 1)

        print(f"S{subj:03d}  acc={acc:.3f}  bal={bal_acc:.3f}  auc={auc:.3f}  "
              f"kappa={kappa:.3f}  p={pval:.3f}", flush=True)

        return dict(
            subject=subj, n_train=len(ytr), n_test=len(yte),
            transfer_acc=round(float(acc), 4),
            bal_acc=round(float(bal_acc), 4),
            auc=round(float(auc), 4),
            kappa=round(float(kappa), 4),
            sensitivity=round(float(sens), 4),
            specificity=round(float(spec), 4),
            perm_pvalue=round(float(pval), 4),
            **per_run
        )
    except Exception as e:
        print(f"S{subj:03d}  FAIL  {e!r}", flush=True)
        return None

# ── Benjamini-Hochberg FDR correction ──────────────────────────────────────────
def bh_fdr(pvalues, alpha=0.05):
    n   = len(pvalues)
    idx = np.argsort(pvalues)
    below = pvalues[idx] <= (np.arange(1, n + 1) / n) * alpha
    if not below.any():
        return np.zeros(n, dtype=bool)
    sig = np.zeros(n, dtype=bool)
    sig[idx[:np.where(below)[0].max() + 1]] = True
    return sig

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:       {EPOCHS_DIR}")
    print(f"Results:      {RESULTS_DIR}")
    print(f"Subjects:     {len(SUBJECTS)}  Workers: {N_WORKERS}  Permutations: {N_PERM}\n")

    t0 = time.time()
    results = []
    with mp.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(process_subject, SUBJECTS), 1):
            results.append(r)
            print(f"  [{i}/{len(SUBJECTS)} subjects done, {time.time()-t0:.0f}s elapsed]", flush=True)

    rows = [r for r in results if r is not None]
    df   = pd.DataFrame(rows)
    out  = RESULTS_DIR / "model1_transfer_task2_FF.csv"
    df.to_csv(out, index=False)

    n_ok = len(rows); n_fail = len(SUBJECTS) - n_ok
    print(f"\nDone. {n_ok} ok | {n_fail} failed -> {out}")
    if len(df):
        print("\n== Transfer Task 2 (executed -> imagined, fists vs feet), Model 1 ==")
        for col in ["transfer_acc", "bal_acc", "auc", "kappa"]:
            s = df[col]
            print(f"{col:>14}: mean={s.mean():.3f}  median={s.median():.3f}  std={s.std():.3f}  "
                  f">0.6={(s > 0.6).mean():.2f}  >0.7={(s > 0.7).mean():.2f}")
        print(f"{'sensitivity':>14}: mean={df['sensitivity'].mean():.3f}")
        print(f"{'specificity':>14}: mean={df['specificity'].mean():.3f}")
        pvals   = df["perm_pvalue"].values
        n_raw   = (pvals < 0.05).sum()
        n_fdr   = bh_fdr(pvals).sum()
        print(f"\nPermutation (N={N_PERM}):")
        print(f"  p<0.05 uncorrected : {n_raw}/{len(df)} ({100*n_raw/len(df):.0f}%)")
        print(f"  p<0.05 FDR-corrected: {n_fdr}/{len(df)} ({100*n_fdr/len(df):.0f}%)")

if __name__ == "__main__":
    main()
