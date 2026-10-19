#!/usr/bin/env python
"""Model 1 (CSP+LDA): Task 1 within-subject LORO evaluation."""

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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, SUBJECTS, TASKS, FMIN, FMAX, TMIN, TMAX,
                             N_CSP, N_PERM_WITHIN)

RESULTS_DIR = M1_DIR
TASK_RUNS   = TASKS["task1_LR"]
N_PERM      = N_PERM_WITHIN
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var

# ── helpers ────────────────────────────────────────────────────────────────────
def load_subject(subject):
    Xs, ys, gs = [], [], []
    for run in TASK_RUNS:
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
        X, y, g = load_subject(subj)
        if X is None:
            print(f"S{subj:03d}  no-data", flush=True)
            return None

        # ── LORO: collect predictions + decision values across all folds ──────
        loro = LeaveOneGroupOut()
        y_true_all, y_pred_all, y_score_all = [], [], []
        fold_accs, per_run = [], {}

        for tr, te in loro.split(X, y, g):
            clf = make_clf().fit(X[tr], y[tr])
            yp  = clf.predict(X[te])
            ys  = clf.decision_function(X[te])
            fa  = float((yp == y[te]).mean())
            fold_accs.append(fa)
            y_true_all.append(y[te]);  y_pred_all.append(yp);  y_score_all.append(ys)
            per_run[f"acc_r{g[te[0]]}"] = round(fa, 4)

        yt = np.concatenate(y_true_all)
        yp = np.concatenate(y_pred_all)
        ys = np.concatenate(y_score_all)

        # ── metrics ──────────────────────────────────────────────────────────
        # acc_loro = mean of fold accuracies (permutation-test statistic)
        acc     = float(np.mean(fold_accs))
        bal_acc = balanced_accuracy_score(yt, yp)
        auc     = roc_auc_score(yt, ys)
        kappa   = cohen_kappa_score(yt, yp)
        tn, fp, fn, tp_v = confusion_matrix(yt, yp).ravel()
        sens    = tp_v / (tp_v + fn)
        spec    = tn  / (tn  + fp)

        # ── permutation test: shuffle all labels, re-run LORO ────────────────
        rng = np.random.default_rng(42 + subj)
        count = 0
        for _ in range(N_PERM):
            y_perm = rng.permutation(y)
            perm_accs = []
            for tr, te in loro.split(X, y_perm, g):
                clf_p = make_clf().fit(X[tr], y_perm[tr])
                perm_accs.append(clf_p.score(X[te], y_perm[te]))
            if np.mean(perm_accs) >= acc:
                count += 1
        pval = (count + 1) / (N_PERM + 1)

        print(f"S{subj:03d}  acc={acc:.3f}  bal={bal_acc:.3f}  auc={auc:.3f}  "
              f"kappa={kappa:.3f}  p={pval:.3f}", flush=True)

        return dict(
            subject=subj, n_trials=len(y),
            n0=int((y == 0).sum()), n1=int((y == 1).sum()),
            acc_loro=round(float(acc), 4),
            acc_loro_std=round(float(np.std(fold_accs)), 4),
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
    out  = RESULTS_DIR / "model1_task1_LR_within_subject.csv"
    df.to_csv(out, index=False)

    n_ok = len(rows); n_fail = len(SUBJECTS) - n_ok
    print(f"\nDone. {n_ok} ok | {n_fail} failed -> {out}")
    if len(df):
        print("\n== Task 1 within-subject (LORO), Model 1 ==")
        for col in ["acc_loro", "bal_acc", "auc", "kappa"]:
            s = df[col]
            print(f"{col:>12}: mean={s.mean():.3f}  median={s.median():.3f}  std={s.std():.3f}  "
                  f">0.6={(s > 0.6).mean():.2f}  >0.7={(s > 0.7).mean():.2f}")
        print(f"{'sensitivity':>12}: mean={df['sensitivity'].mean():.3f}")
        print(f"{'specificity':>12}: mean={df['specificity'].mean():.3f}")
        pvals   = df["perm_pvalue"].values
        n_raw   = (pvals < 0.05).sum()
        n_fdr   = bh_fdr(pvals).sum()
        print(f"\nPermutation (N={N_PERM}):")
        print(f"  p<0.05 uncorrected : {n_raw}/{len(df)} ({100*n_raw/len(df):.0f}%)")
        print(f"  p<0.05 FDR-corrected: {n_fdr}/{len(df)} ({100*n_fdr/len(df):.0f}%)")

if __name__ == "__main__":
    main()
