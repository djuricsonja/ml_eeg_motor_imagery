#!/usr/bin/env python
"""Model 2A (Riemannian tangent space + LDA): within-subject LORO evaluation, both tasks."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M2_DIR, SUBJECTS, TASKS, FMIN, FMAX, TMIN, TMAX,
                             N_PERM_WITHIN)

RESULTS_DIR = M2_DIR
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var

# ── helpers ────────────────────────────────────────────────────────────────────
def load_within(subject, runs):
    """Load epochs for LORO, tracking which run each epoch came from (0/1/2)."""
    Xs, ys, gs = [], [], []
    for run_idx, run in enumerate(runs):
        path = EPOCHS_DIR / f"S{subject:03d}_R{run:02d}-epo.fif"
        if not path.exists():
            return None, None, None   # any missing run → skip subject
        ep = mne.read_epochs(path, preload=True, verbose=False)
        ep.filter(FMIN, FMAX, picks="eeg", verbose=False)
        ep.crop(tmin=TMIN, tmax=TMAX)
        eeg = mne.pick_types(ep.info, eeg=True)
        Xs.append(ep.get_data(picks=eeg))
        ys.append((ep.events[:, -1] == ep.event_id["T2"]).astype(int))
        gs.append(np.full(len(ep), run_idx, dtype=int))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def compute_metrics(y_true, y_pred, y_score):
    """acc, bal_acc, auc, kappa, sensitivity (T2 recall), specificity (T1 recall)."""
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc     = roc_auc_score(y_true, y_score)
    kappa   = cohen_kappa_score(y_true, y_pred)
    tn, fp, fn, tp_v = confusion_matrix(y_true, y_pred).ravel()
    sens    = tp_v / (tp_v + fn)
    spec    = tn  / (tn  + fp)
    return acc, bal_acc, auc, kappa, sens, spec


def bh_fdr(pvalues, alpha=0.05):
    n   = len(pvalues)
    idx = np.argsort(pvalues)
    below = pvalues[idx] <= (np.arange(1, n + 1) / n) * alpha
    if not below.any():
        return np.zeros(n, dtype=bool)
    sig = np.zeros(n, dtype=bool)
    sig[idx[:np.where(below)[0].max() + 1]] = True
    return sig


# ── within-subject worker (LORO + fast permutation) ───────────────────────────
def _within_one(args):
    subj, runs = args
    mne.set_log_level("WARNING")
    try:
        print(f"S{subj:03d}  starting...", flush=True)
        X, y, run_groups = load_within(subj, runs)
        if X is None:
            print(f"S{subj:03d}  no-data", flush=True)
            return None

        cov_est = Covariances(estimator="lwf")
        n_runs  = len(runs)

        # Phase 1: precompute LORO tangent features once (label-free).
        fold_data = []
        for fold in range(n_runs):
            tr = np.where(run_groups != fold)[0]
            te = np.where(run_groups == fold)[0]
            covs_tr = cov_est.fit_transform(X[tr])
            covs_te = cov_est.fit_transform(X[te])
            ts      = TangentSpace(metric="riemann").fit(covs_tr)
            fold_data.append((tr, te, ts.transform(covs_tr), ts.transform(covs_te)))
        print(f"S{subj:03d}  phase1_done", flush=True)

        # Phase 2: observed predictions — pool across all 3 test folds
        y_true_all, y_pred_all, y_score_all = [], [], []
        obs_fold_accs = []
        for tr, te, Xtr, Xte in fold_data:
            clf = make_lda().fit(Xtr, y[tr])
            yp  = clf.predict(Xte)
            ys  = clf.decision_function(Xte)
            obs_fold_accs.append((yp == y[te]).mean())
            y_true_all.append(y[te])
            y_pred_all.append(yp)
            y_score_all.append(ys)

        yt  = np.concatenate(y_true_all)
        yp  = np.concatenate(y_pred_all)
        ys  = np.concatenate(y_score_all)
        _, bal_acc, auc, kappa, sens, spec = compute_metrics(yt, yp, ys)
        # acc = mean of fold accuracies
        acc = float(np.mean(obs_fold_accs))
        print(f"S{subj:03d}  phase2_done  acc={acc:.3f}", flush=True)

        # Phase 3: fast permutation — refit LDA only (tangent features fixed)
        rng   = np.random.default_rng(42 + subj)
        count = 0
        for i in range(N_PERM_WITHIN):
            y_perm     = rng.permutation(y)
            fold_accs  = []
            for tr, te, Xtr, Xte in fold_data:
                clf_p = make_lda().fit(Xtr, y_perm[tr])
                fold_accs.append((clf_p.predict(Xte) == y_perm[te]).mean())
            if np.mean(fold_accs) >= acc:
                count += 1
            if (i + 1) % 10 == 0:
                print(f"S{subj:03d}  perm_{i+1}_done", flush=True)
        pval = (count + 1) / (N_PERM_WITHIN + 1)

        print(f"S{subj:03d}  acc={acc:.3f}  bal={bal_acc:.3f}  auc={auc:.3f}  "
              f"kappa={kappa:.3f}  p={pval:.3f}", flush=True)

        return dict(
            subject=subj, n_trials=len(y),
            acc=round(float(acc), 4),         bal_acc=round(float(bal_acc), 4),
            auc=round(float(auc), 4),         kappa=round(float(kappa), 4),
            sensitivity=round(float(sens), 4), specificity=round(float(spec), 4),
            perm_pvalue=round(float(pval), 4),
        )
    except Exception as e:
        print(f"S{subj:03d}  FAIL  {e!r}", flush=True)
        return None


def run_within(task_name, runs):
    print(f"\n{'='*60}")
    print(f"Within-subject  {task_name}  runs={runs}  LORO  2A only")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    results = []
    with mp.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_within_one, [(s, runs) for s in SUBJECTS]), 1):
            results.append(r)
            print(f"  [{i}/{len(SUBJECTS)} subjects done, {time.time()-t0:.0f}s elapsed]", flush=True)

    rows = [r for r in results if r is not None]
    df   = pd.DataFrame(rows)
    sig  = bh_fdr(df["perm_pvalue"].values)
    df["perm_pvalue_fdr_sig"] = sig.astype(int)

    out = RESULTS_DIR / f"model2_{task_name}_within_subject.csv"
    df.to_csv(out, index=False)

    n_raw = (df["perm_pvalue"] < 0.05).sum()
    n_fdr = int(sig.sum())
    print(f"\n== {task_name} within-subject (LORO, 2A), Model 2 ==")
    for col in ["acc", "bal_acc", "auc", "kappa"]:
        sc = df[col]
        print(f"{col:>10}: mean={sc.mean():.3f}  median={sc.median():.3f}  "
              f"std={sc.std():.3f}  >0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
    print(f"{'sensitivity':>10}: mean={df['sensitivity'].mean():.3f}")
    print(f"{'specificity':>10}: mean={df['specificity'].mean():.3f}")
    print(f"\nPermutation (N={N_PERM_WITHIN}):")
    print(f"  p<0.05 uncorrected:   {n_raw}/{len(df)} ({100*n_raw/len(df):.0f}%)")
    print(f"  p<0.05 FDR-corrected: {n_fdr}/{len(df)} ({100*n_fdr/len(df):.0f}%)")
    print(f"Saved -> {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:   {EPOCHS_DIR}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Subjects: {len(SUBJECTS)}  N_PERM_WITHIN: {N_PERM_WITHIN}  N_WORKERS: {N_WORKERS}\n")

    for task_name, runs in TASKS.items():
        run_within(task_name, runs)

    print("\nAll done.")


if __name__ == "__main__":
    main()
