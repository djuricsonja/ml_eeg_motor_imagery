#!/usr/bin/env python
"""Batch preprocessing of all subjects and task runs (parallel): raw EDF -> cleaned epochs."""

# single-threaded workers; must be set before numpy/mne are imported
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import csv, traceback, warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import numpy as np
import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import Ransac

mne.set_log_level("ERROR")
# limit torch threads (ICLabel backend)
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass

# ── config ─────────────────────────────────────────────────────────────────────
PROJECT     = Path(__file__).resolve().parents[1]
DATA_ROOT   = PROJECT / "mne_data"
EPOCHS_DIR  = PROJECT / "data" / "epochs"
MANIFEST_DIR = Path(__file__).resolve().parent
EPOCHS_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE  = {88, 92, 100}                            # 128 Hz subjects (see data_audit.ipynb)
SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDE]
RUNS     = list(range(3, 15))                       # all task runs

ICLABEL_THRESHOLD = 0.8
N_WORKERS         = 100      # set to available CPU cores

def epoch_path(subject, run):
    return EPOCHS_DIR / f"S{subject:03d}_R{run:02d}-epo.fif"

# ── preprocess one run ─────────────────────────────────────────────────────────
def process_one(args):
    subject, run = args
    qc = {"subject": subject, "run": run}
    stage = "load"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fname = eegbci.load_data(subject, [run], path=str(DATA_ROOT), update_path=True)[0]
            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)

            stage = "montage"
            eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

            stage = "highpass"
            raw.filter(1.0, None, fir_design="firwin", verbose=False)

            stage = "reference"
            raw.set_eeg_reference("average", projection=False, verbose=False)

            stage = "ransac"
            events, _ = mne.events_from_annotations(raw, verbose=False)
            tmp = mne.Epochs(raw, events, tmin=0.0, tmax=4.0,
                             baseline=None, preload=True, verbose=False)
            ransac = Ransac(verbose=False, n_jobs=1)
            ransac.fit(tmp)
            raw.info["bads"] = ransac.bad_chs_
            if ransac.bad_chs_:
                raw.interpolate_bads(reset_bads=True, verbose=False)
            qc["n_bad_chs"] = len(ransac.bad_chs_)
            qc["bad_chs"]   = ",".join(ransac.bad_chs_)

            stage = "ica"
            rank = mne.compute_rank(raw, verbose=False)["eeg"]
            ica = ICA(n_components=rank, method="picard", max_iter=1000,
                      random_state=42, fit_params=dict(ortho=False, extended=True))
            ica.fit(raw, verbose=False)

            stage = "iclabel"
            lab = label_components(raw, ica, method="iclabel")
            excl = [i for i, (l, p) in enumerate(zip(lab["labels"], lab["y_pred_proba"]))
                    if l not in ("brain", "other") and p > ICLABEL_THRESHOLD]
            ica.exclude = excl
            ica.apply(raw, verbose=False)
            qc["n_ica_excluded"] = len(excl)
            qc["ica_labels"]     = ",".join(lab["labels"][i] for i in excl)

            stage = "lowpass"
            raw.filter(None, 40.0, fir_design="firwin", verbose=False)

            stage = "resample"
            raw.resample(128.0, verbose=False)

            stage = "epoch"
            events, emap = mne.events_from_annotations(raw, verbose=False)
            event_id = {k: emap[k] for k in ("T1", "T2") if k in emap}
            epochs = mne.Epochs(raw, events, event_id=event_id,
                                tmin=-0.5, tmax=4.0, baseline=None,
                                preload=True, verbose=False)
            qc["n_epochs_in"]  = len(epochs)
            qc["n_epochs_out"] = len(epochs)
            qc["n_T1"] = int(np.sum(epochs.events[:, 2] == emap.get("T1", -1)))
            qc["n_T2"] = int(np.sum(epochs.events[:, 2] == emap.get("T2", -1)))

            stage = "save"
            epochs.save(epoch_path(subject, run), overwrite=True, verbose=False)

            msgs = [str(w.message) for w in caught
                    if "filtered between 1 and 100" not in str(w.message)]
            qc.update(status="ok", n_warnings=len(msgs), warnings="; ".join(msgs)[:500])
            return qc
        except Exception as e:
            return {"subject": subject, "run": run, "status": "failed", "stage": stage,
                    "error_type": type(e).__name__, "error_msg": str(e),
                    "traceback": traceback.format_exc()}

# ── main ───────────────────────────────────────────────────────────────────────
LOG_FIELDS = ["timestamp", "subject", "run", "status", "stage",
              "n_bad_chs", "bad_chs", "n_ica_excluded", "ica_labels",
              "n_epochs_in", "n_epochs_out", "n_T1", "n_T2",
              "n_warnings", "warnings", "error_type", "error_msg", "traceback"]

def main():
    log_path = MANIFEST_DIR / f"preprocessing_manifest_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    def log_row(rec):
        row = {k: "" for k in LOG_FIELDS}
        row["timestamp"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        row.update({k: v for k, v in rec.items() if k in LOG_FIELDS})
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)

    tasks = [(s, r) for s in SUBJECTS for r in RUNS if not epoch_path(s, r).exists()]
    done  = len(SUBJECTS) * len(RUNS) - len(tasks)
    print(f"{len(SUBJECTS)*len(RUNS)} total | {done} already on disk | {len(tasks)} to do")
    print(f"{N_WORKERS} workers, ICLabel threshold {ICLABEL_THRESHOLD}, manifest {log_path.name}\n")

    n_ok = n_failed = 0
    with mp.Pool(N_WORKERS) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_one, tasks), 1):
            log_row(rec)
            if rec["status"] == "ok":
                n_ok += 1
                print(f"[{i}/{len(tasks)}] S{rec['subject']:03d} R{rec['run']:02d}  ok    "
                      f"epochs={rec['n_epochs_out']} ({rec['n_T1']}/{rec['n_T2']})  "
                      f"bad={rec['n_bad_chs']}  ica_excl={rec['n_ica_excluded']}")
            else:
                n_failed += 1
                print(f"[{i}/{len(tasks)}] S{rec['subject']:03d} R{rec['run']:02d}  "
                      f"FAILED at {rec['stage']}: {rec['error_type']}: {rec['error_msg']}")
    print(f"\nDone. ok={n_ok}  failed={n_failed}  (skipped {done})")

if __name__ == "__main__":
    main()
