# FAIR Universe weak lensing: CNN + WST ensemble

## Model

Ten ensemble members. Each one alternates between two backbones, `klite_inception` and `klite_inception_se` (an Inception-style CNN, with squeeze-and-excitation in the second). Batch sizes cycle through 32, 48, 64, 96 and 128. Each member uses its own 80/20 split over the 256 nuisance-parameter systems (`random_state=5566+i`).

Training happens in two stages:

1. **CNN.** Pretrain for 20 epochs on noiseless maps, then train for 50 epochs on maps with fresh shape noise added under the mask. Augmentation uses random H/V flips per sample plus a random dihedral view per batch. Optimiser is Adam, lr `2e-4 * bs/64`, weight decay `1e-4`, and the LR halves when validation loss plateaus.
2. **WST head.** 630 scattering coefficients are reduced to 64 PCA components. These feed a small MLP that adds a residual correction to the CNN output. A 3-class head on the S8 residual acts as an auxiliary loss. This stage runs for 25 epochs on the pre-noised HDF5 maps, with lr `1e-5` for the head and one tenth of that for the CNN.

At inference, each member's predictions are averaged over 8 dihedral views. A Gaussian likelihood per cosmology is calibrated on the validation predictions from all members combined, then covariance shrinkage, kernel smoothing and temperature scaling are applied to it. Test-time member weights come from the unsupervised marginal likelihood, and posterior means and error bars come from the cosmology grid.

## Usage

Run from this directory. Outputs go to `data/`, `models/` and `submissions/`.

```bash
python compute_scattering.py    # noisy maps + scattering coefficients to HDF5
python train_ensemble.py        # both stages
python train_ensemble.py cnn    # CNN stage only
python train_ensemble.py wst    # WST head only (needs models/model_{i}.pth)
python infer_ensemble.py        # writes submissions/Submission_<date>.zip
```

`compute_scattering.py` needs to run first, and only once. It generates 150 noise realisations (seeds 500–649) of every training map on the GPU. For each one it saves the float16 masked map and the 630 isotropic scattering-covariance coefficients from `foscat`, using 4 orientations and all scales. They go into 15 HDF5 files of 10 realisations each, about 58 GB per file. The WST stage of training reads these files. With the default settings the output is bit-identical to the original dataset in `/rds/fair_challenge/wavelet_scattering_float16`.

To run inference with the original weights, copy `fair_universe_final/data` and `fair_universe_final/models` in here first.

Requires `fair_universe`, `foscat`, `torch`, `torchvision`, `scikit-learn`, `h5py`, `hdf5plugin` and `tqdm`. Data paths are set at the top of `train_ensemble.py`.
