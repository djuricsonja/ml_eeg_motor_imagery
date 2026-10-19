#!/usr/bin/env python
"""Model 3 (EEGNet): cross-subject GroupKFold-5 evaluation, both tasks."""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from sklearn.model_selection import GroupKFold
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
from modeling.config import (EPOCHS_DIR, M1_DIR, M2_DIR, M3_DIR, SUBJECTS, TASKS,
                             FMIN, FMAX, TMIN, TMAX, MAX_EPOCHS, N_PERM_CROSS)

RESULTS_DIR = M3_DIR
N_PERM      = N_PERM_CROSS
DEVICE      = "cpu"
SMOKE       = False  # True -> 6 subjects, 2 folds, 5 epochs for a fast sanity check

MODELS = {
    "EEGNet":  (EEGNet, {}),
}


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
        Xs.append(ep.get_data(picks=mne.pick_types(ep.info, eeg=True)))
        ys.append((ep.events[:, -1] == ep.event_id["T2"]).astype(np.int64))
    if not Xs:
        return None, None
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


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
        batch_size=64,
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


# ── GroupKFold predictions for one model architecture ─────────────────────────
def _collect_predictions(name, cls, extra, X_all, y_all, g_all, n_splits, max_epochs, fold_id_all=None):
    """Run GroupKFold and return full-dataset y_pred and y_prob arrays."""
    n_chans, n_times = X_all.shape[1], X_all.shape[2]
    gkf    = GroupKFold(n_splits=n_splits)
    y_pred = np.empty(len(y_all), dtype=np.int64)
    y_prob = np.empty(len(y_all), dtype=np.float32)

    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, g_all), 1):
        Xtr, Xte = standardize(X_all[tr], X_all[te])
        clf = make_clf(cls, extra, n_chans, n_times, max_epochs)
        clf.fit(Xtr, y_all[tr])
        y_pred[te] = clf.predict(Xte)
        y_prob[te] = clf.predict_proba(Xte)[:, 1]
        if fold_id_all is not None:
            fold_id_all[te] = fold
        print(f"  [{name}] fold {fold}/{n_splits}  "
              f"train={len(np.unique(g_all[tr]))} subj  "
              f"test={len(np.unique(g_all[te]))} subj  [{time.time()-t0:.0f}s elapsed]", flush=True)

    return y_pred, y_prob


# ── run cross-subject for one task ─────────────────────────────────────────────
def run_cross(task_name, runs, subjects, n_splits, max_epochs, n_perm):
    print(f"\n{'='*60}")
    print(f"Cross-subject  {task_name}  runs={runs}  GroupKFold-{n_splits}  EEGNet")
    print(f"{'='*60}", flush=True)

    # load all subjects
    X_list, y_list, g_list = [], [], []
    for s in subjects:
        X, y = load_subject(s, runs)
        if X is None:
            print(f"S{s:03d}  skipped (missing runs)", flush=True)
            continue
        X_list.append(X); y_list.append(y)
        g_list.append(np.full(len(y), s, dtype=np.int64))

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    g_all = np.concatenate(g_list)
    subjs = np.unique(g_all)
    print(f"Loaded: {len(X_all)} epochs, {len(subjs)} subjects\n", flush=True)

    # collect predictions for each model
    # fold ids are deterministic; capture once
    model_preds = {}
    fold_id_all = np.empty(len(y_all), dtype=int)
    first = True
    for name, (cls, extra) in MODELS.items():
        print(f"\n-- {name} --", flush=True)
        y_pred, y_prob = _collect_predictions(
            name, cls, extra, X_all, y_all, g_all, n_splits, max_epochs,
            fold_id_all=(fold_id_all if first else None)
        )
        model_preds[name] = (y_pred, y_prob)
        first = False

    # per-subject metrics (no significance test)
    rows = []
    for s in subjs:
        m   = g_all == s
        row = {"subject": int(s), "n_test": int(m.sum())}
        for name in MODELS:
            tag           = name.lower()
            y_pred, y_prob = model_preds[name]
            acc, bal_acc, auc, kappa, sens, spec = compute_metrics(
                y_all[m], y_pred[m], y_prob[m]
            )
            row.update({
                f"{tag}_acc":         round(float(acc),     4),
                f"{tag}_bal_acc":     round(float(bal_acc), 4),
                f"{tag}_auc":         round(float(auc),     4),
                f"{tag}_kappa":       round(float(kappa),   4),
                f"{tag}_sensitivity": round(float(sens),    4),
                f"{tag}_specificity": round(float(spec),    4),
            })
            print(f"  S{s:03d} [{name:8s}]  acc={acc:.3f}  bal={bal_acc:.3f}  "
                  f"auc={auc:.3f}  kappa={kappa:.3f}", flush=True)
        rows.append(row)

    df = pd.DataFrame(rows)

    out = RESULTS_DIR / f"model3_eegnet_cross_{task_name}.csv"
    df.to_csv(out, index=False)

    # per-fold metrics
    fold_rows = []
    for fold in range(1, n_splits + 1):
        fm = fold_id_all == fold
        row = {"fold": fold, "n_test": int(fm.sum())}
        for name in MODELS:
            tag = name.lower()
            y_pred, y_prob = model_preds[name]
            f_acc, f_bal, f_auc, f_kap, f_sn, f_sp = compute_metrics(
                y_all[fm], y_pred[fm], y_prob[fm]
            )
            row.update({
                f"{tag}_acc": round(float(f_acc), 4), f"{tag}_bal_acc": round(float(f_bal), 4),
                f"{tag}_auc": round(float(f_auc), 4), f"{tag}_kappa": round(float(f_kap), 4),
                f"{tag}_sensitivity": round(float(f_sn), 4), f"{tag}_specificity": round(float(f_sp), 4),
            })
        fold_rows.append(row)
    fold_df  = pd.DataFrame(fold_rows)
    fold_out = RESULTS_DIR / f"model3_eegnet_cross_{task_name}_perfold.csv"
    fold_df.to_csv(fold_out, index=False)
    print(f"\n== {task_name} cross-subject PER-FOLD (GroupKFold-{n_splits}), Model 3 ==")
    for name in MODELS:
        tag = name.lower()
        print(f"  {name}: acc mean={fold_df[f'{tag}_acc'].mean():.4f} std={fold_df[f'{tag}_acc'].std():.4f}  "
              f"auc mean={fold_df[f'{tag}_auc'].mean():.4f} std={fold_df[f'{tag}_auc'].std():.4f}")
    print(f"Saved -> {fold_out}", flush=True)

    # permutation test on fixed predictions (labels shuffled within subject)
    if n_perm > 0:
        subj_idx = {int(s): np.where(g_all == s)[0] for s in subjs}
        pooled_rows = []
        for name in MODELS:
            tag = name.lower()
            y_pred, y_prob = model_preds[name]
            _, p_bal, p_auc, p_kap, p_sn, p_sp = compute_metrics(y_all, y_pred, y_prob)

            obs_fold_accs = [
                (y_pred[fold_id_all == f] == y_all[fold_id_all == f]).mean()
                for f in range(1, n_splits + 1)
            ]
            obs_mean_acc = float(np.mean(obs_fold_accs))

            rng    = np.random.default_rng(42)
            count  = 0
            t_perm = time.time()
            for i in range(n_perm):
                y_perm = y_all.copy()
                for idx in subj_idx.values():
                    y_perm[idx] = rng.permutation(y_all[idx])
                perm_fold_accs = [
                    (y_pred[fold_id_all == f] == y_perm[fold_id_all == f]).mean()
                    for f in range(1, n_splits + 1)
                ]
                if np.mean(perm_fold_accs) >= obs_mean_acc:
                    count += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{tag}] perm_{i+1}/{n_perm}_done  [{time.time()-t_perm:.0f}s elapsed]", flush=True)
            pval = (count + 1) / (n_perm + 1)

            pooled_rows.append(dict(
                model=name, acc=round(obs_mean_acc, 4), bal_acc=round(float(p_bal), 4),
                auc=round(float(p_auc), 4), kappa=round(float(p_kap), 4),
                sensitivity=round(float(p_sn), 4), specificity=round(float(p_sp), 4),
                perm_pvalue=round(float(pval), 4), n_total=len(y_all),
            ))
        pooled_df  = pd.DataFrame(pooled_rows)
        pooled_out = RESULTS_DIR / f"model3_eegnet_cross_{task_name}_pooled.csv"
        pooled_df.to_csv(pooled_out, index=False)
        print(f"\n== {task_name} cross-subject POOLED (GroupKFold-{n_splits}), Model 3 ==")
        for _, r in pooled_df.iterrows():
            print(f"  {r['model']}: acc={r['acc']:.4f}  auc={r['auc']:.4f}  "
                  f"kappa={r['kappa']:.4f}  perm_pvalue={r['perm_pvalue']:.4f}")
        print(f"Saved -> {pooled_out}", flush=True)

    # ROC data
    roc_out = RESULTS_DIR / f"model3_eegnet_cross_{task_name}_rocdata.npz"
    np.savez(roc_out, y_true=y_all,
             eegnet_score=model_preds["EEGNet"][1],
             fold_id=fold_id_all, subject=g_all)
    print(f"Saved -> {roc_out}", flush=True)

    # summary
    print(f"\n== {task_name} cross-subject (GroupKFold-{n_splits}), Model 3 ==")
    for name in MODELS:
        tag = name.lower()
        print(f"\n  {name}:")
        for metric in ["acc", "bal_acc", "auc", "kappa"]:
            sc = df[f"{tag}_{metric}"]
            print(f"    {metric:>8s}: mean={sc.mean():.3f}  median={sc.median():.3f}  "
                  f"std={sc.std():.3f}  >0.6={(sc > 0.6).mean():.2f}  >0.7={(sc > 0.7).mean():.2f}")
        print(f"    {'sens':>8s}: mean={df[f'{tag}_sensitivity'].mean():.3f}  "
              f"spec: mean={df[f'{tag}_specificity'].mean():.3f}")

    # compare vs M1 and M2
    m1 = M1_DIR / f"model1_{task_name}_cross_subject.csv"
    m2 = M2_DIR / f"model2_{task_name}_cross_subject.csv"
    for label, p, b_col in [
        ("M1 CSP+LDA",        m1, "acc"),
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
          f"N_PERM: {N_PERM}  SMOKE: {SMOKE}\n")

    if SMOKE:
        subs = SUBJECTS[:6]
        print(f"[SMOKE] {len(subs)} subjects, 2 folds, 5 epochs — sanity check only")
        for task_name, runs in TASKS.items():
            run_cross(task_name, runs, subs, n_splits=2, max_epochs=5, n_perm=0)
    else:
        for task_name, runs in TASKS.items():
            run_cross(task_name, runs, SUBJECTS,
                      n_splits=5, max_epochs=MAX_EPOCHS, n_perm=N_PERM)

    print("\nAll done.")


if __name__ == "__main__":
    main()
