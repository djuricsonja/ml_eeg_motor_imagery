#!/usr/bin/env python
"""Model 1 (CSP+LDA): cross-subject GroupKFold-5 evaluation, both tasks."""

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
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)

mne.set_log_level("WARNING")

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, SUBJECTS, TASKS, FMIN, FMAX, TMIN, TMAX,
                             N_CSP, N_PERM_CROSS)

RESULTS_DIR = M1_DIR
N_PERM      = N_PERM_CROSS
N_WORKERS   = int(os.environ.get("N_WORKERS", 30))  # override via env var

# ── helpers ────────────────────────────────────────────────────────────────────
def load_subject(subject, runs):
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


def make_clf():
    return Pipeline([
        ("csp", CSP(n_components=N_CSP, reg="ledoit_wolf", log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


# ── permutation worker ─────────────────────────────────────────────────────────
# shared with forked workers
_PERM_CTX = {}


def _perm_one(perm_idx):
    X_all       = _PERM_CTX["X_all"]
    y_all       = _PERM_CTX["y_all"]
    subj_idx    = _PERM_CTX["subj_idx"]
    fold_splits = _PERM_CTX["fold_splits"]

    rng    = np.random.default_rng(1_000_000 + perm_idx)  # seed per permutation
    y_perm = y_all.copy()
    for idx in subj_idx.values():
        y_perm[idx] = rng.permutation(y_all[idx])

    perm_fold_accs = []
    for tr, te in fold_splits:
        clf_p = make_clf().fit(X_all[tr], y_perm[tr])
        perm_fold_accs.append((clf_p.predict(X_all[te]) == y_perm[te]).mean())
    return float(np.mean(perm_fold_accs))


# ── cross-subject evaluation for one task ─────────────────────────────────────
def run_task(task_name, runs):
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  runs: {runs}")
    print(f"{'='*60}", flush=True)

    # load all subjects
    X_list, y_list, g_list = [], [], []
    for s in SUBJECTS:
        X, y = load_subject(s, runs)
        if X is None:
            print(f"S{s:03d}  skipped (missing runs)", flush=True)
            continue
        X_list.append(X)
        y_list.append(y)
        g_list.append(np.full(len(y), s, dtype=int))

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)
    subjects_ok = np.unique(g_all)
    print(f"Loaded: {len(X_all)} epochs, {len(subjects_ok)} subjects\n", flush=True)

    # GroupKFold-5: collect per-epoch predictions
    gkf = GroupKFold(n_splits=5)
    y_pred_all  = np.empty(len(y_all), dtype=int)
    y_score_all = np.empty(len(y_all))   # decision_function for AUC
    fold_id_all = np.empty(len(y_all), dtype=int)  # which of the 5 folds each trial was tested in
    fold_splits = []  # fold indices, reused for permutations

    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, g_all), 1):
        clf = make_clf().fit(X_all[tr], y_all[tr])
        y_pred_all[te]  = clf.predict(X_all[te])
        y_score_all[te] = clf.decision_function(X_all[te])
        fold_id_all[te] = fold
        fold_splits.append((tr, te))
        n_tr_s = len(np.unique(g_all[tr]))
        n_te_s = len(np.unique(g_all[te]))
        print(f"  fold {fold}/5  train={len(tr)} epochs ({n_tr_s} subj)  "
              f"test={len(te)} epochs ({n_te_s} subj)  [{time.time()-t0:.0f}s elapsed]", flush=True)

    # per-fold metrics
    fold_rows = []
    for fold in range(1, 6):
        fm = fold_id_all == fold
        yt_f, yp_f, ys_f = y_all[fm], y_pred_all[fm], y_score_all[fm]
        f_acc     = accuracy_score(yt_f, yp_f)
        f_bal_acc = balanced_accuracy_score(yt_f, yp_f)
        f_auc     = roc_auc_score(yt_f, ys_f)
        f_kappa   = cohen_kappa_score(yt_f, yp_f)
        tn, fp, fn, tp_v = confusion_matrix(yt_f, yp_f).ravel()
        fold_rows.append(dict(
            fold=fold, n_test=int(fm.sum()),
            acc=round(float(f_acc), 4), bal_acc=round(float(f_bal_acc), 4),
            auc=round(float(f_auc), 4), kappa=round(float(f_kappa), 4),
            sensitivity=round(float(tp_v / (tp_v + fn)), 4),
            specificity=round(float(tn / (tn + fp)), 4),
        ))
    fold_df  = pd.DataFrame(fold_rows)
    fold_out = RESULTS_DIR / f"model1_{task_name}_cross_subject_perfold.csv"
    fold_df.to_csv(fold_out, index=False)
    print(f"\n== {task_name} cross-subject PER-FOLD (GroupKFold-5), Model 1 ==")
    for col in ["acc", "auc", "kappa"]:
        print(f"  {col:>10}: mean={fold_df[col].mean():.4f}  std={fold_df[col].std():.4f}")
    print(f"Saved -> {fold_out}", flush=True)

    # pooled metrics
    p_bal_acc = balanced_accuracy_score(y_all, y_pred_all)
    p_auc     = roc_auc_score(y_all, y_score_all)
    p_kappa   = cohen_kappa_score(y_all, y_pred_all)
    tn, fp, fn, tp_v = confusion_matrix(y_all, y_pred_all).ravel()
    p_sens = tp_v / (tp_v + fn)
    p_spec = tn  / (tn  + fp)

    obs_mean_acc = float(fold_df["acc"].mean())

    subj_idx = {int(s): np.where(g_all == s)[0] for s in subjects_ok}

    # permutation test: full CSP+LDA refit, labels shuffled within subject (parallel)
    _PERM_CTX["X_all"]       = X_all
    _PERM_CTX["y_all"]       = y_all
    _PERM_CTX["subj_idx"]    = subj_idx
    _PERM_CTX["fold_splits"] = fold_splits

    count  = 0
    t_perm = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for i, perm_acc in enumerate(pool.imap_unordered(_perm_one, range(N_PERM)), 1):
            if perm_acc >= obs_mean_acc:
                count += 1
            if i % 50 == 0:
                print(f"  perm_{i}/{N_PERM}_done  [{time.time()-t_perm:.0f}s elapsed]", flush=True)
    pooled_pvalue = (count + 1) / (N_PERM + 1)

    pooled_out = RESULTS_DIR / f"model1_{task_name}_cross_subject_pooled.csv"
    pd.DataFrame([dict(
        acc=round(obs_mean_acc, 4), bal_acc=round(float(p_bal_acc), 4),
        auc=round(float(p_auc), 4), kappa=round(float(p_kappa), 4),
        sensitivity=round(float(p_sens), 4), specificity=round(float(p_spec), 4),
        perm_pvalue=round(float(pooled_pvalue), 4), n_total=len(y_all),
    )]).to_csv(pooled_out, index=False)
    print(f"\n== {task_name} cross-subject POOLED (GroupKFold-5), Model 1 ==")
    print(f"  acc(fold-mean)={obs_mean_acc:.4f}  bal_acc={p_bal_acc:.4f}  auc={p_auc:.4f}  kappa={p_kappa:.4f}")
    print(f"  sensitivity={p_sens:.4f}  specificity={p_spec:.4f}")
    print(f"  permutation p-value (N={N_PERM}, full refit per permutation): {pooled_pvalue:.4f}")
    print(f"Saved -> {pooled_out}", flush=True)

    # ROC data
    roc_out = RESULTS_DIR / f"model1_{task_name}_cross_subject_rocdata.npz"
    np.savez(roc_out, y_true=y_all, y_score=y_score_all, fold_id=fold_id_all, subject=g_all)
    print(f"Saved -> {roc_out}", flush=True)

    # per-subject metrics (no significance test)
    rows = []
    for s in subjects_ok:
        m  = g_all == s
        yt = y_all[m]
        yp = y_pred_all[m]
        ys = y_score_all[m]

        acc     = accuracy_score(yt, yp)
        bal_acc = balanced_accuracy_score(yt, yp)
        auc     = roc_auc_score(yt, ys)
        kappa   = cohen_kappa_score(yt, yp)
        tn, fp, fn, tp_v = confusion_matrix(yt, yp).ravel()
        sens    = tp_v / (tp_v + fn)
        spec    = tn  / (tn  + fp)

        rows.append(dict(
            subject=int(s), n_test=int(m.sum()),
            acc=round(float(acc), 4),
            bal_acc=round(float(bal_acc), 4),
            auc=round(float(auc), 4),
            kappa=round(float(kappa), 4),
            sensitivity=round(float(sens), 4),
            specificity=round(float(spec), 4),
        ))
        print(f"S{s:03d}  acc={acc:.3f}  bal={bal_acc:.3f}  auc={auc:.3f}  "
              f"kappa={kappa:.3f}  sens={sens:.3f}  spec={spec:.3f}", flush=True)

    df  = pd.DataFrame(rows)
    out = RESULTS_DIR / f"model1_{task_name}_cross_subject.csv"
    df.to_csv(out, index=False)

    print(f"\n== {task_name} cross-subject PER-SUBJECT (GroupKFold-5), Model 1 ==")
    for col in ["acc", "bal_acc", "auc", "kappa"]:
        sc = df[col]
        print(f"{col:>10}: mean={sc.mean():.3f}  median={sc.median():.3f}  "
              f"std={sc.std():.3f}  >0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
    print(f"{'sensitivity':>10}: mean={df['sensitivity'].mean():.3f}")
    print(f"{'specificity':>10}: mean={df['specificity'].mean():.3f}")
    print(f"Saved -> {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:   {EPOCHS_DIR}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Subjects: {len(SUBJECTS)}  N_PERM: {N_PERM}  N_WORKERS: {N_WORKERS}")

    for task_name, runs in TASKS.items():
        run_task(task_name, runs)

    print("\nAll done.")


if __name__ == "__main__":
    main()
