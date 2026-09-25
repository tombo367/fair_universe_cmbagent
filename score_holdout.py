import json
import argparse
import zipfile
from pathlib import Path

import numpy as np

from utils.data import DATA_DIR, HOLDOUT_LABEL_FILE
from utils.score import per_sample_score, score_phase1

BLOCKS = {
    "all 10020": slice(0, 10020),
    "Phase 1 test (first 4020)": slice(0, 4020),
    "training grid (last 6000)": slice(4020, 10020),
}


def load_submission(path):
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read("result.json"))
    return np.array(d["means"], float), np.array(d["errorbars"], float)


def report(name, y, mu, sig, n_boot, rng):
    ps = per_sample_score(y, mu, sig)
    boot = ps[rng.integers(0, len(ps), size=(n_boot, len(ps)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    cosmos = np.unique(y, axis=0)
    gmean = np.array([ps[np.all(y == c, axis=1)].mean() for c in cosmos])
    sem_cosmo = gmean.std(ddof=1) / np.sqrt(len(cosmos))

    err = mu - y
    z = err / sig
    print(f"\n=== {name}: {len(y)} maps, {len(cosmos)} cosmologies ===")
    print(f"  score          : {score_phase1(y, mu, sig):.4f}")
    print(f"  95% CI (maps)  : [{lo:.4f}, {hi:.4f}]   bootstrap, B={n_boot}")
    print(f"  SEM (cosmology): +/- {sem_cosmo:.4f}")
    for i, p in enumerate(["Omega_m", "S8"]):
        print(f"  {p:8s} bias {err[:, i].mean():+.4f}  RMSE {np.sqrt((err[:, i] ** 2).mean()):.4f}  "
              f"mean sigma {sig[:, i].mean():.4f}  68% cov {np.mean(np.abs(z[:, i]) <= 1):.3f}  "
              f"95% cov {np.mean(np.abs(z[:, i]) <= 1.959964):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Score a holdout submission against the Phase 1 holdout labels.")
    parser.add_argument("submission", nargs="?", help="defaults to the newest submissions/Submission_holdout_*.zip")
    parser.add_argument("--labels", default=str(Path(DATA_DIR) / HOLDOUT_LABEL_FILE))
    parser.add_argument("--n_boot", type=int, default=50000)
    args = parser.parse_args()

    path = args.submission or max(Path("submissions").glob("Submission_holdout_*.zip"), key=lambda p: p.stat().st_mtime)
    mu, sig = load_submission(path)
    y = np.load(args.labels)
    assert mu.shape == y.shape, f"submission has {mu.shape[0]} rows, labels have {y.shape[0]}"
    print(f"Scoring {path}")

    rng = np.random.default_rng(0)
    for name, sl in BLOCKS.items():
        report(name, y[sl], mu[sl], sig[sl], args.n_boot, rng)


if __name__ == "__main__":
    main()
