# FAIR Universe weak lensing: CNN + WST ensemble

## Model

Ten ensemble members. Each one alternates between two backbones, `klite_inception` and `klite_inception_se` (an Inception-style CNN, with squeeze-and-excitation in the second). Batch sizes cycle through 32, 48, 64, 96 and 128. Each member uses its own 80/20 split over the 256 nuisance-parameter systems (`random_state=5566+i`).

Training happens in two stages:

1. **CNN.** Pretrain for 20 epochs on noiseless maps, then train for 50 epochs on maps with fresh shape noise added under the mask. Augmentation uses random H/V flips per sample plus a random dihedral view per batch. Optimiser is Adam, lr `2e-4 * bs/64`, weight decay `1e-4`, and the LR halves when validation loss plateaus.
2. **WST head.** 630 scattering coefficients are reduced to 64 PCA components. These feed a small MLP that adds a residual correction to the CNN output. A 3-class head on the S8 residual acts as an auxiliary loss. This stage runs for 25 epochs on the pre-noised HDF5 maps, with lr `1e-5` for the head and one tenth of that for the CNN.

At inference, each member's predictions are averaged over 8 dihedral views. A Gaussian likelihood per cosmology is calibrated on the validation predictions from all members combined, then covariance shrinkage, kernel smoothing and temperature scaling are applied to it. Test-time member weights come from the unsupervised marginal likelihood, and posterior means and error bars come from the cosmology grid.

## Setup

Needs Python 3.12 and an NVIDIA GPU with a CUDA 13 driver.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The public challenge data is read from `$FAIR_DATA_DIR`, which defaults to `/rds/fair_challenge/public_data`. The precomputed scattering files go in `/rds/fair_challenge/wavelet_scattering_float16`, which you can change with `--save_dir` and `WST_DATA_DIR` in `train_ensemble.py`.

## Layout

- `compute_scattering.py`, `train_ensemble.py`, `infer_ensemble.py`: the three pipeline steps, run in that order
- `score_holdout.py`: scores holdout predictions against the Phase 1 labels
- `utils/`: small helpers for loading data, adding shape noise, computing the scattering transform and writing submissions

## Reproducing the results

Run everything from this directory with the environment activated. Each step writes a log to `logs/`. Outputs go to `data/`, `models/` and `submissions/`, all gitignored.

```bash
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
```

### 1. Dataset

```bash
python -u compute_scattering.py --master_seed 500 --n_noisy 150 2>&1 | tee logs/scattering.log
```

This makes 150 noise realisations of every training map, seeds 500–649. Each realisation stores the float16 masked map and the 630 isotropic scattering-covariance coefficients from `foscat`, with 4 orientations and all scales. The files hold 10 realisations each, about 58 GB per file, so 875 GB in total. On one GPU each realisation takes about 1.5 minutes for the scattering transform alone.

### 2. Training

```bash
python -u train_ensemble.py cnn 2>&1 | tee logs/train_cnn.log
python -u train_ensemble.py wst 2>&1 | tee logs/train_wst.log
```

The CNN stage trains all 10 members on noise added on the fly, and writes `models/model_{i}_pretrain.pth` and `models/model_{i}.pth`. It also writes each member's split, label scaler and image statistics to `data/`. The WST stage fits the scattering PCA and saves it to `data/wst_pca.npz`. It then trains each member's residual head on the HDF5 files and writes `models/model_{i}_wst_residual_with_error.pth`. Running `python train_ensemble.py` with no argument does both stages in turn.

The WST stage takes its HDF5 files in `os.listdir` order: epochs 0–9 use the first file, 10–19 the second and 20–24 the third. Inference also validates on the first file. In the original run the first three files were `570-579`, `530-539` and `560-569`. If you regenerate the data somewhere else the listing order may differ, and each member will then see different noise realisations.

### 3. Inference

```bash
python -u infer_ensemble.py --split test    2>&1 | tee logs/infer_test.log
python -u infer_ensemble.py --split holdout 2>&1 | tee logs/infer_holdout.log
python score_holdout.py 2>&1 | tee logs/score_holdout.log
```

Both inference runs calibrate the ensemble likelihood on each member's validation systems, then predict the given split. They write `submissions/Submission_<split>_<date>.zip` and `data/<split>_eval.npz`.

- `--split test` predicts the 4,000 unlabelled public test maps.
- `--split holdout` predicts the 10,020 labelled Phase 1 holdout maps and prints the overall score. It needs `WIDE12H_bin2_2arcmin_kappa_noisy_Phase1_holdout.npy` and `Phase-1_holdout_labels_10020.npy` in `$FAIR_DATA_DIR`.
  - The first 4,020 rows are the Phase 1 test set, which has 10 cosmologies.
  - The last 6,000 cover all 101 training cosmologies.

`score_holdout.py` scores the newest holdout submission, or a zip you pass it. It reports the challenge score for the full holdout and for each of those two blocks. For each it also gives a bootstrap 95% interval, a cosmology-level error, and the bias, RMSE and 68%/95% coverage for Ω_m and S8. The final submission scores 11.34 on the full holdout, 10.79 on the Phase 1 test set and 11.70 on the last 6,000 maps.

## License

Released under the [MIT License](LICENSE).
