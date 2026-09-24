import os
import argparse
import numpy as np
import torch
import h5py
import hdf5plugin
from tqdm import tqdm

from utils.data import DATA_DIR, Data
from utils.noise import add_noise
from utils.scattering import N_COEFFS, make_scat_op, scattering_coefficients

PER_FILE = 10
TILE = 32


def main():
    parser = argparse.ArgumentParser(description="Precompute noisy kappa maps and their scattering coefficients.")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--save_dir", default="/rds/fair_challenge/wavelet_scattering_float16")
    parser.add_argument("--n_noisy", type=int, default=150)
    parser.add_argument("--master_seed", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_float32", action="store_true")
    args = parser.parse_args()

    data = Data(args.data_dir)
    data.load_train_data()
    kappa, mask, label = data.kappa, data.mask, data.label
    scat_op = make_scat_op()
    mask_b = mask.astype(bool)
    n_pix = int(mask_b.sum())
    kappa_dtype = np.float32 if args.save_float32 else np.float16
    zstd = hdf5plugin.Zstd(clevel=1)
    os.makedirs(args.save_dir, exist_ok=True)

    for f in range((args.n_noisy + PER_FILE - 1) // PER_FILE):
        first = f * PER_FILE
        last = min(first + PER_FILE, args.n_noisy)
        path = os.path.join(args.save_dir, f"precomputed_dataset_seeds_{args.master_seed + first}-{args.master_seed + last - 1}.h5")
        print(f"Writing {path}")

        with h5py.File(path, "w") as h5f:
            dset_st = h5f.create_dataset("ST_coefficients", shape=(last - first, data.Ncosmo, data.Nsys, N_COEFFS), dtype=np.float32,
                                         chunks=(1, data.Ncosmo, TILE, N_COEFFS), compression=zstd, shuffle=True)
            dset_kappa = h5f.create_dataset("noisy_kappa", shape=(last - first, data.Ncosmo, data.Nsys, n_pix), dtype=kappa_dtype,
                                            chunks=(1, data.Ncosmo, TILE, n_pix), compression=zstd, shuffle=True)
            h5f.create_dataset("labels", data=label, compression=zstd, shuffle=True)
            h5f.create_dataset("mask", data=mask, compression=zstd, shuffle=True)

            for n in range(first, last):
                seed = args.master_seed + n
                print(f"Realisation {n + 1}/{args.n_noisy} (seed {seed})")
                noisy = add_noise(kappa, mask, data.ng, seed=seed)
                torch.cuda.empty_cache()
                dset_st[n % PER_FILE] = np.stack([
                    scattering_coefficients(scat_op, noisy[i], mask, args.batch_size)
                    for i in tqdm(range(data.Ncosmo), desc="ST")
                ])
                dset_kappa[n % PER_FILE] = noisy[:, :, mask_b].astype(kappa_dtype)
                h5f.flush()
                os.fsync(h5f.id.get_vfd_handle())
                del noisy
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
