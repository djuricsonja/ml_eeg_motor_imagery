#!/usr/bin/env python
"""Model 2A/2B (Riemannian, unaligned/aligned): cross-subject GroupKFold-5 evaluation, both tasks."""

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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)
from scipy.stats import wilcoxon

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, M2_DIR, SUBJECTS, TASKS, FMIN, FMAX, TMIN, TMAX,
                             N_PERM_CROSS)
from recentering import recenter_per_domain

RESULTS_DIR = M2_DIR
N_WORKERS   = int(os.environ.get("N_WORKERS", 30))  # override via env var


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


# ── permutation worker ─────────────────────────────────────────────────────────
_PERM_CTX = {}   # shared with forked workers


def _perm_one(perm_idx):
    y_all     = _PERM_CTX["y_all"]
    subj_idx  = _PERM_CTX["subj_idx"]
    fold_data = _PERM_CTX["fold_data"]

    rng    = np.random.default_rng(1_000_000 + perm_idx)
    y_perm = y_all.copy()
    for idx in subj_idx.values():
        y_perm[idx] = rng.permutation(y_all[idx])

    perm_fold_accs = []
    for tr, te, Xtr, Xte in fold_data:
        clf_p = make_lda().fit(Xtr, y_perm[tr])
        perm_fold_accs.append((clf_p.predict(Xte) == y_perm[te]).mean())
    return float(np.mean(perm_fold_accs))


# ── cross-subject (GroupKFold-5, 2A + 2B) ─────────────────────────────────────
def run_cross(task_name, runs):
    print(f"\n{'='*60}")
    print(f"Cross-subject  {task_name}  runs={runs}  GroupKFold-5  2A+2B")
    print(f"{'='*60}", flush=True)

    # load all subjects
    X_list, y_list, g_list = [], [], []
    for s in SUBJECTS:
        X, y = load_runs(s, runs)
        if X is None:
            print(f"S{s:03d}  skipped (missing runs)", flush=True)
            continue
        X_list.append(X); y_list.append(y)
        g_list.append(np.full(len(y), s, dtype=int))

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)
    subjs = np.unique(g_all)
    print(f"Loaded: {len(X_all)} epochs, {len(subjs)} subjects", flush=True)

    # compute covariances once
    covs_all = Covariances(estimator="lwf").fit_transform(X_all)

    # 2B: per-subject recentering (label-free)
    covs_rc = recenter_per_domain(covs_all, g_all)
    print("Riemannian recentering (2B) done.", flush=True)

    gkf = GroupKFold(n_splits=5)

    # pre-allocate prediction arrays
    pred_2a  = np.empty(len(y_all), dtype=int)
    score_2a = np.empty(len(y_all))
    pred_2b  = np.empty(len(y_all), dtype=int)
    score_2b = np.empty(len(y_all))
    fold_id_all = np.empty(len(y_all), dtype=int)
    # cached per-fold tangent features, reused for permutation refits
    fold_data_2a = []  # (tr, te, Xtr_a, Xte_a)
    fold_data_2b = []  # (tr, te, Xtr_b, Xte_b)

    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, g_all), 1):
        # 2A: no alignment
        ts_a  = TangentSpace(metric="riemann").fit(covs_all[tr])
        Xtr_a = ts_a.transform(covs_all[tr])
        Xte_a = ts_a.transform(covs_all[te])
        clf_a = make_lda().fit(Xtr_a, y_all[tr])
        pred_2a[te]  = clf_a.predict(Xte_a)
        score_2a[te] = clf_a.decision_function(Xte_a)

        # 2B: recentered covariances (precomputed)
        ts_b  = TangentSpace(metric="riemann").fit(covs_rc[tr])
        Xtr_b = ts_b.transform(covs_rc[tr])
        Xte_b = ts_b.transform(covs_rc[te])
        clf_b = make_lda().fit(Xtr_b, y_all[tr])
        pred_2b[te]  = clf_b.predict(Xte_b)
        score_2b[te] = clf_b.decision_function(Xte_b)

        fold_id_all[te] = fold
        fold_data_2a.append((tr, te, Xtr_a, Xte_a))
        fold_data_2b.append((tr, te, Xtr_b, Xte_b))
        print(f"  fold {fold}/5  train={len(np.unique(g_all[tr]))} subj  "
              f"test={len(np.unique(g_all[te]))} subj  [{time.time()-t0:.0f}s elapsed]", flush=True)

    # per-fold metrics
    fold_rows = []
    for fold in range(1, 6):
        fm = fold_id_all == fold
        yt_f = y_all[fm]
        a_a, b_a, u_a, k_a, sn_a, sp_a = compute_metrics(yt_f, pred_2a[fm], score_2a[fm])
        a_b, b_b, u_b, k_b, sn_b, sp_b = compute_metrics(yt_f, pred_2b[fm], score_2b[fm])
        fold_rows.append(dict(
            fold=fold, n_test=int(fm.sum()),
            m2a_acc=round(a_a,4), m2a_bal_acc=round(b_a,4), m2a_auc=round(u_a,4),
            m2a_kappa=round(k_a,4), m2a_sensitivity=round(sn_a,4), m2a_specificity=round(sp_a,4),
            m2b_acc=round(a_b,4), m2b_bal_acc=round(b_b,4), m2b_auc=round(u_b,4),
            m2b_kappa=round(k_b,4), m2b_sensitivity=round(sn_b,4), m2b_specificity=round(sp_b,4),
        ))
    fold_df  = pd.DataFrame(fold_rows)
    fold_out = RESULTS_DIR / f"model2_{task_name}_cross_subject_perfold.csv"
    fold_df.to_csv(fold_out, index=False)
    print(f"\n== {task_name} cross-subject PER-FOLD (GroupKFold-5), Model 2 ==")
    for tag, acol, aucol in [("2A", "m2a_acc", "m2a_auc"), ("2B", "m2b_acc", "m2b_auc")]:
        print(f"  {tag}: acc mean={fold_df[acol].mean():.4f} std={fold_df[acol].std():.4f}  "
              f"auc mean={fold_df[aucol].mean():.4f} std={fold_df[aucol].std():.4f}")
    print(f"Saved -> {fold_out}", flush=True)

    # permutation test (labels shuffled within subject), parallel
    subj_idx = {int(s): np.where(g_all == s)[0] for s in subjs}
    variants = {"m2a": (pred_2a, score_2a, fold_data_2a), "m2b": (pred_2b, score_2b, fold_data_2b)}

    pooled_rows = []
    for tag, (pred, score, fold_data) in variants.items():
        _, p_bal, p_auc, p_kap, p_sn, p_sp = compute_metrics(y_all, pred, score)
        obs_mean_acc = float(fold_df[f"{tag}_acc"].mean())

        _PERM_CTX["y_all"]     = y_all
        _PERM_CTX["subj_idx"]  = subj_idx
        _PERM_CTX["fold_data"] = fold_data

        count  = 0
        t_perm = time.time()
        with mp.Pool(N_WORKERS) as pool:
            for i, perm_acc in enumerate(pool.imap_unordered(_perm_one, range(N_PERM_CROSS)), 1):
                if perm_acc >= obs_mean_acc:
                    count += 1
                if i % 50 == 0:
                    print(f"  [{tag}] perm_{i}/{N_PERM_CROSS}_done  [{time.time()-t_perm:.0f}s elapsed]", flush=True)
        pval = (count + 1) / (N_PERM_CROSS + 1)

        pooled_rows.append(dict(
            variant=tag, acc=round(obs_mean_acc,4), bal_acc=round(p_bal,4), auc=round(p_auc,4),
            kappa=round(p_kap,4), sensitivity=round(p_sn,4), specificity=round(p_sp,4),
            perm_pvalue=round(float(pval),4), n_total=len(y_all),
        ))
    pooled_df  = pd.DataFrame(pooled_rows)
    pooled_out = RESULTS_DIR / f"model2_{task_name}_cross_subject_pooled.csv"
    pooled_df.to_csv(pooled_out, index=False)
    print(f"\n== {task_name} cross-subject POOLED (GroupKFold-5), Model 2 ==")
    for _, r in pooled_df.iterrows():
        print(f"  {r['variant'].upper()}: acc={r['acc']:.4f}  auc={r['auc']:.4f}  "
              f"kappa={r['kappa']:.4f}  perm_pvalue={r['perm_pvalue']:.4f}")
    print(f"Saved -> {pooled_out}", flush=True)

    # ROC data
    roc_out = RESULTS_DIR / f"model2_{task_name}_cross_subject_rocdata.npz"
    np.savez(roc_out, y_true=y_all, m2a_score=score_2a, m2b_score=score_2b,
             fold_id=fold_id_all, subject=g_all)
    print(f"Saved -> {roc_out}", flush=True)

    # per-subject metrics
    rows = []
    for s in subjs:
        m = g_all == s
        a_a, b_a, u_a, k_a, sn_a, sp_a = compute_metrics(y_all[m], pred_2a[m], score_2a[m])
        a_b, b_b, u_b, k_b, sn_b, sp_b = compute_metrics(y_all[m], pred_2b[m], score_2b[m])
        rows.append(dict(
            subject=int(s), n_test=int(m.sum()),
            m2a_acc=round(a_a,4), m2a_bal_acc=round(b_a,4), m2a_auc=round(u_a,4),
            m2a_kappa=round(k_a,4), m2a_sensitivity=round(sn_a,4), m2a_specificity=round(sp_a,4),
            m2b_acc=round(a_b,4), m2b_bal_acc=round(b_b,4), m2b_auc=round(u_b,4),
            m2b_kappa=round(k_b,4), m2b_sensitivity=round(sn_b,4), m2b_specificity=round(sp_b,4),
        ))
        print(f"S{s:03d}  2A acc={a_a:.3f} auc={u_a:.3f}  |  "
              f"2B acc={a_b:.3f} auc={u_b:.3f}", flush=True)

    df  = pd.DataFrame(rows)
    out = RESULTS_DIR / f"model2_{task_name}_cross_subject.csv"
    df.to_csv(out, index=False)

    # summary
    print(f"\n== {task_name} cross-subject (GroupKFold-5), Model 2 ==")
    for tag, acol, aucol in [("2A", "m2a_acc", "m2a_auc"), ("2B", "m2b_acc", "m2b_auc")]:
        sc = df[acol]
        print(f"  {tag}: acc={sc.mean():.3f}  auc={df[aucol].mean():.3f}  "
              f">0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
    d = df["m2b_acc"] - df["m2a_acc"]
    print(f"  2B-2A: {d.mean():+.3f}  p={wilcoxon(df.m2b_acc, df.m2a_acc).pvalue:.4f}  "
          f"({(d > 0).sum()}/{len(df)} up)")

    # compare against saved M1 baseline
    m1_path = M1_DIR / f"model1_{task_name}_cross_subject.csv"
    if m1_path.exists():
        m1 = pd.read_csv(m1_path)[["subject", "acc", "auc"]].rename(
            columns={"acc": "m1_acc", "auc": "m1_auc"})
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
    print(f"Subjects: {len(SUBJECTS)}  N_PERM_CROSS: {N_PERM_CROSS}  N_WORKERS: {N_WORKERS}\n")

    for task_name, runs in TASKS.items():
        run_cross(task_name, runs)

    print("\nAll done.")


if __name__ == "__main__":
    main()
