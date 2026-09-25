import os
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("FAIR_DATA_DIR", "/rds/fair_challenge/public_data"))

MASK_FILE = "WIDE12H_bin2_2arcmin_mask.npy"
KAPPA_FILE = "WIDE12H_bin2_2arcmin_kappa.npy"
LABEL_FILE = "label.npy"
TEST_KAPPA_FILE = "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
HOLDOUT_KAPPA_FILE = "WIDE12H_bin2_2arcmin_kappa_noisy_Phase1_holdout.npy"
HOLDOUT_LABEL_FILE = "Phase-1_holdout_labels_10020.npy"


class Data:
    """Public FAIR Universe weak lensing data. Maps are stored as unmasked pixels only and unpacked here."""

    Ncosmo = 101
    Nsys = 256
    Ntest = 4000
    Nholdout = 10020
    shape = (1424, 176)
    pixelsize_arcmin = 2
    ng = 30

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)

    def _load(self, name):
        return np.load(self.data_dir / name)

    def load_train_data(self):
        self.mask = self._load(MASK_FILE)
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:, :, self.mask] = self._load(KAPPA_FILE)
        self.label = self._load(LABEL_FILE)

    def load_test_data(self):
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:, self.mask] = self._load(TEST_KAPPA_FILE)

    def load_holdout_data(self):
        """Labelled Phase 1 holdout: rows 0-4019 are the Phase 1 test set, rows 4020-10019 cover all 101 training cosmologies."""
        self.kappa_holdout = np.zeros((self.Nholdout, *self.shape), dtype=np.float16)
        self.kappa_holdout[:, self.mask] = self._load(HOLDOUT_KAPPA_FILE)
        self.label_holdout = self._load(HOLDOUT_LABEL_FILE)
