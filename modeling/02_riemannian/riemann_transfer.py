#!/usr/bin/env python
"""Model 2A/2B (Riemannian, unaligned/aligned): executed -> imagined transfer evaluation, both tasks."""

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
from scipy.stats import wilcoxon

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, M2_DIR, SUBJECTS, TRANSFER, FMIN, FMAX, TMIN, TMAX,
                             N_PERM_TRANSFER)
from recentering import recenter_to_identity

RESULTS_DIR = M2_DIR
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var

# ── helpers ────────────────────────────────────────────────────────────────────
def load_runs(subject, runs):
    """Load and concatenate epochs across runs (no run-group tracking)."""
    Xs, ys = [], []
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
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)


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


# ── transfer worker (executed -> imagined, 2A + 2B) ───────────────────────────
def _transfer_one(args):
    subj, train_runs, test_runs = args
    print(f"S{subj:03d}  starting...", flush=True)
    mne.set_log_level("WARNING")
    try:
        Xe, ye = load_runs(subj, train_runs)
        Xi, yi = load_runs(subj, test_runs)
        if Xe is None or Xi is None:
            print(f"S{subj:03d}  no-data", flush=True)
            return None

        cov_est = Covariances(estimator="lwf")
        ce = cov_est.fit_transform(Xe)
        ci = cov_est.fit_transform(Xi)

        # 2A: tangent reference = executed Riemannian mean, no alignment
        ts_a  = TangentSpace(metric="riemann").fit(ce)
        clf_a = make_lda().fit(ts_a.transform(ce), ye)
        pred_2a  = clf_a.predict(ts_a.transform(ci))
        score_2a = clf_a.decision_function(ts_a.transform(ci))

        # 2B: recenter executed and imagined domains independently to identity
        cer  = recenter_to_identity(ce)
        cir  = recenter_to_identity(ci)
        ts_b  = TangentSpace(metric="riemann").fit(cer)
        clf_b = make_lda().fit(ts_b.transform(cer), ye)
        pred_2b  = clf_b.predict(ts_b.transform(cir))
        score_2b = clf_b.decision_function(ts_b.transform(cir))

        a_a, b_a, u_a, k_a, sn_a, sp_a = compute_metrics(yi, pred_2a, score_2a)
        a_b, b_b, u_b, k_b, sn_b, sp_b = compute_metrics(yi, pred_2b, score_2b)

        # permutation test: permute test labels, predictions are fixed
        rng   = np.random.default_rng(42 + subj)
        c_a, c_b = 0, 0
        for _ in range(N_PERM_TRANSFER):
            yi_p = rng.permutation(yi)
            if accuracy_score(yi_p, pred_2a) >= a_a: c_a += 1
            if accuracy_score(yi_p, pred_2b) >= a_b: c_b += 1
        pval_2a = (c_a + 1) / (N_PERM_TRANSFER + 1)
        pval_2b = (c_b + 1) / (N_PERM_TRANSFER + 1)

        print(f"S{subj:03d}  2A acc={a_a:.3f} auc={u_a:.3f} p={pval_2a:.3f}  |  "
              f"2B acc={a_b:.3f} auc={u_b:.3f} p={pval_2b:.3f}", flush=True)

        return dict(
            subject=subj, n_train=len(ye), n_test=len(yi),
            m2a_acc=round(a_a,4), m2a_bal_acc=round(b_a,4), m2a_auc=round(u_a,4),
            m2a_kappa=round(k_a,4), m2a_sensitivity=round(sn_a,4), m2a_specificity=round(sp_a,4),
            m2b_acc=round(a_b,4), m2b_bal_acc=round(b_b,4), m2b_auc=round(u_b,4),
            m2b_kappa=round(k_b,4), m2b_sensitivity=round(sn_b,4), m2b_specificity=round(sp_b,4),
            m2a_perm_pvalue=round(pval_2a,4),
            m2b_perm_pvalue=round(pval_2b,4),
        )
    except Exception as e:
        print(f"S{subj:03d}  FAIL  {e!r}", flush=True)
        return None


def run_transfer(task_name, train_runs, test_runs):
    print(f"\n{'='*60}")
    print(f"Transfer  {task_name}  train={train_runs} -> test={test_runs}  2A+2B")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    results = []
    with mp.Pool(N_WORKERS) as pool:
        args = [(s, train_runs, test_runs) for s in SUBJECTS]
        for i, r in enumerate(pool.imap_unordered(_transfer_one, args), 1):
            results.append(r)
            print(f"  [{i}/{len(SUBJECTS)} subjects done, {time.time()-t0:.0f}s elapsed]", flush=True)

    rows = [r for r in results if r is not None]
    df   = pd.DataFrame(rows)

    for tag in ("m2a", "m2b"):
        sig = bh_fdr(df[f"{tag}_perm_pvalue"].values)
        df[f"{tag}_perm_fdr_sig"] = sig.astype(int)

    out = RESULTS_DIR / f"model2_transfer_{task_name}.csv"
    df.to_csv(out, index=False)

    print(f"\n== Transfer {task_name}, Model 2 ==")
    for tag in ("2A", "2B"):
        t     = "m" + tag.lower()
        fdr_n = df[f"{t}_perm_fdr_sig"].sum()
        print(f"  {tag}: acc={df[f'{t}_acc'].mean():.3f}  "
              f"auc={df[f'{t}_auc'].mean():.3f}  "
              f"kappa={df[f'{t}_kappa'].mean():.3f}  "
              f"sens={df[f'{t}_sensitivity'].mean():.3f}  "
              f"spec={df[f'{t}_specificity'].mean():.3f}  "
              f"FDR={fdr_n}/{len(df)}")
    d = df["m2b_acc"] - df["m2a_acc"]
    print(f"  2B-2A: {d.mean():+.3f}  p={wilcoxon(df.m2b_acc, df.m2a_acc).pvalue:.4f}  "
          f"({(d > 0).sum()}/{len(df)} up)")

    # compare against saved M1 transfer baseline
    m1_path = M1_DIR / f"model1_transfer_{task_name}.csv"
    if m1_path.exists():
        m1 = pd.read_csv(m1_path)[["subject", "transfer_acc", "auc"]].rename(
            columns={"transfer_acc": "m1_acc", "auc": "m1_auc"})
        mg = df.merge(m1, on="subject")
        print(f"\n  Vs M1 (CSP+LDA, loaded from saved CSV):")
        print(f"    M1  acc={mg.m1_acc.mean():.3f}  auc={mg.m1_auc.mean():.3f}")
        print(f"    2A-M1: {(mg.m2a_acc-mg.m1_acc).mean():+.3f}  "
              f"p={wilcoxon(mg.m2a_acc, mg.m1_acc).pvalue:.4f}")
        print(f"    2B-M1: {(mg.m2b_acc-mg.m1_acc).mean():+.3f}  "
              f"p={wilcoxon(mg.m2b_acc, mg.m1_acc).pvalue:.4f}")
    else:
        print(f"  [warn] M1 baseline not found: {m1_path}")

    print(f"Saved -> {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:   {EPOCHS_DIR}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"M1 CSVs:  {M1_DIR}")
    print(f"Subjects: {len(SUBJECTS)}  N_PERM_TRANSFER: {N_PERM_TRANSFER}  N_WORKERS: {N_WORKERS}\n")

    for task_name, tr in TRANSFER.items():
        run_transfer(task_name, tr["train"], tr["test"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
