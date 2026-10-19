#!/usr/bin/env python
"""Model 3 (EEGNet): within-subject LORO evaluation, both tasks."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping
from braindecode import EEGClassifier
from braindecode.models import EEGNet

mne.set_log_level("ERROR")
torch.manual_seed(42)

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, M2_DIR, M3_DIR, SUBJECTS, TASKS,
                             FMIN, FMAX, TMIN, TMAX, MAX_EPOCHS, N_PERM_WITHIN)

RESULTS_DIR = M3_DIR
N_PERM      = N_PERM_WITHIN
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var
DEVICE      = "cpu"
SMOKE       = False  # True -> 6 subjects, 3 epochs for a fast sanity check

MODELS = {
    "EEGNet":  (EEGNet, {}),
}


# ── helpers ────────────────────────────────────────────────────────────────────
def load_within(subject, runs):
    """Load epochs with run index for LORO; None if any run is missing."""
    Xs, ys, gs = [], [], []
    for run_idx, run in enumerate(runs):
        path = EPOCHS_DIR / f"S{subject:03d}_R{run:02d}-epo.fif"
        if not path.exists():
            return None, None, None
        ep = mne.read_epochs(path, preload=True, verbose=False)
        ep.filter(FMIN, FMAX, picks="eeg", verbose=False)
        ep.crop(tmin=TMIN, tmax=TMAX)
        Xs.append(ep.get_data(picks=mne.pick_types(ep.info, eeg=True)))
        ys.append((ep.events[:, -1] == ep.event_id["T2"]).astype(np.int64))
        gs.append(np.full(len(ep), run_idx, dtype=int))
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(ys),
            np.concatenate(gs))


def standardize(train, test):
    """Per-channel z-score; stats from training fold only (no leakage)."""
    m = train.mean(axis=(0, 2), keepdims=True)
    s = train.std(axis=(0, 2), keepdims=True) + 1e-7
    return ((train - m) / s).astype(np.float32), ((test - m) / s).astype(np.float32)


def make_clf(model_cls, extra, n_chans, n_times, max_epochs):
    return EEGClassifier(
        model_cls,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        **extra,
        max_epochs=max_epochs,
        batch_size=16,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        train_split=ValidSplit(0.2, stratified=True),
        callbacks=[("early", EarlyStopping(patience=15, monitor="valid_loss"))],
        device=DEVICE,
        verbose=0,
    )


def compute_metrics(y_true, y_pred, y_prob):
    """acc, bal_acc, auc, kappa, sensitivity (T2 recall), specificity (T1 recall)."""
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc     = roc_auc_score(y_true, y_prob)
    kappa   = cohen_kappa_score(y_true, y_pred)
    tn, fp, fn, tp_v = confusion_matrix(y_true, y_pred).ravel()
    sens = tp_v / (tp_v + fn)
    spec = tn  / (tn  + fp)
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


# ── per-subject worker ─────────────────────────────────────────────────────────
def _within_one(subject, runs, max_epochs):
    print(f"S{subject:03d}  starting...", flush=True)
    torch.set_num_threads(1)
    mne.set_log_level("ERROR")

    X, y, run_groups = load_within(subject, runs)
    if X is None:
        print(f"S{subject:03d}  skipped (missing run file)", flush=True)
        return None

    n_chans, n_times = X.shape[1], X.shape[2]
    n_runs = len(runs)
    row = {"subject": subject, "n_trials": len(y)}

    # label-free standardization, reused for permutation refits
    fold_data = []
    for fold in range(n_runs):
        tr = np.where(run_groups != fold)[0]
        te = np.where(run_groups == fold)[0]
        Xtr, Xte = standardize(X[tr], X[te])
        fold_data.append((tr, te, Xtr, Xte))

    for name, (cls, extra) in MODELS.items():
        y_pred = np.empty_like(y)
        y_prob = np.empty(len(y), dtype=np.float32)
        fold_accs = []

        for tr, te, Xtr, Xte in fold_data:
            clf = make_clf(cls, extra, n_chans, n_times, max_epochs)
            clf.fit(Xtr, y[tr])
            y_pred[te] = clf.predict(Xte)
            y_prob[te] = clf.predict_proba(Xte)[:, 1]
            fold_accs.append((y_pred[te] == y[te]).mean())

        # acc = mean of fold accuracies (permutation-test statistic)
        _, bal_acc, auc, kappa, sens, spec = compute_metrics(y, y_pred, y_prob)
        acc = float(np.mean(fold_accs))

        # permutation test: full EEGNet refit per permutation
        rng   = np.random.default_rng(42 + subject)
        count = 0
        for _ in range(N_PERM):
            y_perm    = rng.permutation(y)
            perm_accs = []
            for tr, te, Xtr, Xte in fold_data:
                clf_p = make_clf(cls, extra, n_chans, n_times, max_epochs)
                clf_p.fit(Xtr, y_perm[tr])
                perm_accs.append((clf_p.predict(Xte) == y_perm[te]).mean())
            if np.mean(perm_accs) >= acc:
                count += 1
        pval = (count + 1) / (N_PERM + 1)

        tag = name.lower()
        row.update({
            f"{tag}_acc":         round(float(acc),     4),
            f"{tag}_bal_acc":     round(float(bal_acc), 4),
            f"{tag}_auc":         round(float(auc),     4),
            f"{tag}_kappa":       round(float(kappa),   4),
            f"{tag}_sensitivity": round(float(sens),    4),
            f"{tag}_specificity": round(float(spec),    4),
            f"{tag}_perm_pvalue": round(float(pval),    4),
        })

    print(f"S{subject:03d}  "
          f"EEGNet acc={row['eegnet_acc']:.3f} auc={row['eegnet_auc']:.3f} "
          f"p={row['eegnet_perm_pvalue']:.3f}",
          flush=True)
    return row


# ── run one task ───────────────────────────────────────────────────────────────
def run_within(task_name, runs, subjects, max_epochs):
    print(f"\n{'='*60}")
    print(f"Within-subject  {task_name}  runs={runs}  LORO ({len(runs)} folds)  EEGNet")
    print(f"{'='*60}", flush=True)

    results = Parallel(n_jobs=N_WORKERS, verbose=10)(
        delayed(_within_one)(s, runs, max_epochs) for s in subjects
    )

    df = pd.DataFrame([r for r in results if r is not None])
    for name in MODELS:
        tag = name.lower()
        sig = bh_fdr(df[f"{tag}_perm_pvalue"].values)
        df[f"{tag}_perm_pvalue_fdr_sig"] = sig.astype(int)
    out = RESULTS_DIR / f"model3_eegnet_within_{task_name}.csv"
    df.to_csv(out, index=False)

    print(f"\n== {task_name} within-subject (LORO), Model 3 ==")
    for name in MODELS:
        tag = name.lower()
        print(f"\n  {name}:")
        for metric in ["acc", "bal_acc", "auc", "kappa"]:
            sc = df[f"{tag}_{metric}"]
            print(f"    {metric:>8s}: mean={sc.mean():.3f}  median={sc.median():.3f}  "
                  f"std={sc.std():.3f}  >0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
        n_raw = (df[f"{tag}_perm_pvalue"] < 0.05).sum()
        n_fdr = int(df[f"{tag}_perm_pvalue_fdr_sig"].sum())
        print(f"    Permutation (N={N_PERM}):  p<0.05 raw={n_raw}/{len(df)}  FDR={n_fdr}/{len(df)}")
        print(f"    {'sens':>8s}: mean={df[f'{tag}_sensitivity'].mean():.3f}  "
              f"spec: mean={df[f'{tag}_specificity'].mean():.3f}")

    # compare vs M1 and M2
    for label, p, acc_col in [
        ("M1 (CSP+LDA)", M1_DIR / f"model1_{task_name}_within_subject.csv", "acc_loro"),
        ("M2 (Riemann)", M2_DIR / f"model2_{task_name}_within_subject.csv", "acc"),
    ]:
        if p.exists():
            bdf = pd.read_csv(p)[["subject", acc_col]].rename(columns={acc_col: "b_acc"})
            mg  = df[["subject", "eegnet_acc"]].merge(bdf, on="subject")
            print(f"\n  Vs {label} (same LORO scheme — direct comparison):")
            print(f"    {label}:  {mg.b_acc.mean():.3f}")
            print(f"    EEGNet:   {mg.eegnet_acc.mean():.3f}  "
                  f"diff={(mg.eegnet_acc - mg.b_acc).mean():+.3f}")
        else:
            print(f"  [warn] {label} CSV not found: {p}")

    print(f"\nSaved -> {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:   {EPOCHS_DIR}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Subjects: {len(SUBJECTS)}  MAX_EPOCHS: {MAX_EPOCHS}  N_PERM: {N_PERM}  "
          f"N_WORKERS: {N_WORKERS}  SMOKE: {SMOKE}\n")

    if SMOKE:
        subs = SUBJECTS[:6]
        print(f"[SMOKE] {len(subs)} subjects, 3 epochs — sanity check only")
        run_within("task1_LR", TASKS["task1_LR"], subs, max_epochs=3)
    else:
        for task_name, runs in TASKS.items():
            run_within(task_name, runs, SUBJECTS, MAX_EPOCHS)

    print("\nAll done.")


if __name__ == "__main__":
    main()
