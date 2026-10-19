#!/usr/bin/env python
"""Model 3 (EEGNet): executed -> imagined transfer evaluation, both tasks."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import sys
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              roc_auc_score, cohen_kappa_score, confusion_matrix)
from scipy.stats import wilcoxon
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping
from braindecode import EEGClassifier
from braindecode.models import EEGNet

mne.set_log_level("ERROR")
torch.manual_seed(42)

# ── config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modeling.config import (EPOCHS_DIR, M1_DIR, M2_DIR, M3_DIR, SUBJECTS, TRANSFER,
                             FMIN, FMAX, TMIN, TMAX, MAX_EPOCHS, N_PERM_TRANSFER)

RESULTS_DIR = M3_DIR
N_PERM      = N_PERM_TRANSFER
N_WORKERS   = int(os.environ.get("N_WORKERS", 40))  # override via env var
DEVICE      = "cpu"
SMOKE       = False  # True -> 6 subjects, 3 epochs for a fast sanity check

MODELS = {
    "EEGNet":  (EEGNet, {}),
}


# ── helpers ────────────────────────────────────────────────────────────────────
def load_runs(subject, runs):
    Xs, ys = [], []
    for run in runs:
        path = EPOCHS_DIR / f"S{subject:03d}_R{run:02d}-epo.fif"
        if not path.exists():
            continue
        ep = mne.read_epochs(path, preload=True, verbose=False)
        ep.filter(FMIN, FMAX, picks="eeg", verbose=False)
        ep.crop(tmin=TMIN, tmax=TMAX)
        Xs.append(ep.get_data(picks=mne.pick_types(ep.info, eeg=True)))
        ys.append((ep.events[:, -1] == ep.event_id["T2"]).astype(np.int64))
    if not Xs:
        return None, None
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def standardize(train, test):
    """Per-channel z-score; stats from train (executed) only — no leakage."""
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
def _transfer_one(args):
    subject, train_runs, test_runs, max_epochs, n_perm = args
    print(f"S{subject:03d}  starting...", flush=True)
    torch.set_num_threads(1)
    mne.set_log_level("ERROR")

    try:
        Xtr, ytr = load_runs(subject, train_runs)
        Xte, yte = load_runs(subject, test_runs)
        if Xtr is None or Xte is None:
            print(f"S{subject:03d}  skipped (missing runs)", flush=True)
            return None

        Xtr_z, Xte_z = standardize(Xtr, Xte)
        n_chans, n_times = Xtr_z.shape[1], Xtr_z.shape[2]

        row = {"subject": subject, "n_train": len(ytr), "n_test": len(yte)}

        for name, (cls, extra) in MODELS.items():
            clf    = make_clf(cls, extra, n_chans, n_times, max_epochs)
            clf.fit(Xtr_z, ytr)
            y_pred = clf.predict(Xte_z)
            y_prob = clf.predict_proba(Xte_z)[:, 1]

            acc, bal_acc, auc, kappa, sens, spec = compute_metrics(yte, y_pred, y_prob)

            # permutation: permute test labels, predictions fixed (no refit)
            rng   = np.random.default_rng(42 + subject)
            count = sum(
                1 for _ in range(n_perm)
                if accuracy_score(rng.permutation(yte), y_pred) >= acc
            )
            pval = (count + 1) / (n_perm + 1)

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
              f"EEGNet acc={row['eegnet_acc']:.3f} auc={row['eegnet_auc']:.3f} p={row['eegnet_perm_pvalue']:.3f}",
              flush=True)
        return row

    except Exception as e:
        print(f"S{subject:03d}  FAIL  {e!r}", flush=True)
        return None


# ── run transfer for one task ──────────────────────────────────────────────────
def run_transfer(task_name, train_runs, test_runs, subjects, max_epochs, n_perm):
    print(f"\n{'='*60}")
    print(f"Transfer  {task_name}  train={train_runs} -> test={test_runs}  EEGNet")
    print(f"{'='*60}", flush=True)

    args = [(s, train_runs, test_runs, max_epochs, n_perm) for s in subjects]
    t0 = time.time()
    results = []
    with mp.Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_transfer_one, args), 1):
            results.append(r)
            print(f"  [{i}/{len(subjects)} subjects done, {time.time()-t0:.0f}s elapsed]", flush=True)

    rows = [r for r in results if r is not None]
    df   = pd.DataFrame(rows)

    for name in MODELS:
        tag = name.lower()
        sig = bh_fdr(df[f"{tag}_perm_pvalue"].values)
        df[f"{tag}_perm_fdr_sig"] = sig.astype(int)

    out = RESULTS_DIR / f"model3_eegnet_transfer_{task_name}.csv"
    df.to_csv(out, index=False)

    print(f"\n== Transfer {task_name}, Model 3 ==")
    for name in MODELS:
        tag   = name.lower()
        n_fdr = int(df[f"{tag}_perm_fdr_sig"].sum())
        n_raw = int((df[f"{tag}_perm_pvalue"] < 0.05).sum())
        print(f"\n  {name}:")
        for metric in ["acc", "bal_acc", "auc", "kappa"]:
            sc = df[f"{tag}_{metric}"]
            print(f"    {metric:>8s}: mean={sc.mean():.3f}  median={sc.median():.3f}  "
                  f"std={sc.std():.3f}  >0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
        print(f"    {'sens':>8s}: mean={df[f'{tag}_sensitivity'].mean():.3f}  "
              f"spec: mean={df[f'{tag}_specificity'].mean():.3f}")
        print(f"    Permutation (N={n_perm}):  "
              f"p<0.05 raw={n_raw}/{len(df)}  FDR={n_fdr}/{len(df)}")

    # compare vs M1 and M2
    m1 = M1_DIR / f"model1_transfer_{task_name}.csv"
    m2 = M2_DIR / f"model2_transfer_{task_name}.csv"
    for label, p, b_col in [
        ("M1 CSP+LDA",        m1, "transfer_acc"),
        ("M2A Riemann",       m2, "m2a_acc"),
        ("M2B Riemann+align", m2, "m2b_acc"),
    ]:
        if p.exists():
            bdf = pd.read_csv(p)[["subject", b_col]].rename(columns={b_col: "b_acc"})
            mg  = df[["subject", "eegnet_acc"]].merge(bdf, on="subject")
            print(f"\n  Vs {label}:  {mg.b_acc.mean():.3f}")
            diff = (mg.eegnet_acc - mg.b_acc).mean()
            pval = wilcoxon(mg.eegnet_acc, mg.b_acc).pvalue
            print(f"    EEGNet: {mg.eegnet_acc.mean():.3f}  diff={diff:+.3f}  p={pval:.4f}")
        else:
            print(f"  [warn] {label} CSV not found: {p}")

    print(f"\nSaved -> {out}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Epochs:   {EPOCHS_DIR}")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Subjects: {len(SUBJECTS)}  MAX_EPOCHS: {MAX_EPOCHS}  "
          f"N_PERM: {N_PERM}  N_WORKERS: {N_WORKERS}  SMOKE: {SMOKE}\n")

    if SMOKE:
        subs = SUBJECTS[:6]
        print(f"[SMOKE] {len(subs)} subjects, 3 epochs — sanity check only")
        task_name = "task1_LR"
        cfg = TRANSFER[task_name]
        run_transfer(task_name, cfg["train"], cfg["test"],
                     subs, max_epochs=3, n_perm=0)
    else:
        for task_name, cfg in TRANSFER.items():
            run_transfer(task_name, cfg["train"], cfg["test"],
                         SUBJECTS, MAX_EPOCHS, N_PERM)

    print("\nAll done.")


if __name__ == "__main__":
    main()
