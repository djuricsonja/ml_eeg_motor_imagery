"""Shared experiment settings for all modeling scripts."""

from pathlib import Path

PROJECT    = Path(__file__).resolve().parents[1]
EPOCHS_DIR = PROJECT / "data" / "epochs"
M1_DIR     = PROJECT / "modeling" / "01_csp_lda"    / "results"
M2_DIR     = PROJECT / "modeling" / "02_riemannian" / "results"
M3_DIR     = PROJECT / "modeling" / "03_eegnet"     / "results"

EXCLUDE  = {88, 92, 100}       # 128 Hz subjects (see preprocessing/data_audit.ipynb)
SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDE]

TASKS = {
    "task1_LR": [4, 8, 12],    # imagined left vs right fist
    "task2_FF": [6, 10, 14],   # imagined fists vs feet
}
TRANSFER = {                   # executed -> imagined
    "task1_LR": {"train": [3, 7, 11], "test": [4, 8, 12]},
    "task2_FF": {"train": [5, 9, 13], "test": [6, 10, 14]},
}

FMIN, FMAX = 8.0, 30.0         # band-pass (Hz)
TMIN, TMAX = 0.5, 2.5          # analysis window (s)

N_PERM_WITHIN   = 200
N_PERM_CROSS    = 1000
N_PERM_TRANSFER = 1000

N_CSP      = 6                 # Model 1
MAX_EPOCHS = 200               # Model 3
