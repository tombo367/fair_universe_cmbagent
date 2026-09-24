import os
import argparse
import numpy as np
import torch
import h5py
import hdf5plugin
import foscat.scat_cov2D as sc
from tqdm import tqdm

NCOSMO, NSYS = 101, 256
SHAPE = (1424, 176)
NG = 30
PIXEL_SIZE_ARCMIN = 2.0
N_ST = 630
PER_FILE = 10
TILE = 32


def load_train_data(data_dir):
    mask = np.load(os.path.join(data_dir, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.zeros((NCOSMO, NSYS, *SHAPE), dtype=np.float16)
    kappa[:, :, mask] = np.load(os.path.join(data_dir, "WIDE12H_bin2_2arcmin_kappa.npy"))
    label = np.load(os.path.join(data_dir, "label.npy"))
    return kappa, mask, label


def add_noise(data, mask, seed, device="cuda", chunk_ncosmo=32, chunk_nsys=256):
    """Shape noise under the mask, generated on the GPU from a single seeded stream."""
    out = np.empty_like(data, dtype=np.float32)
    scale = torch.tensor(0.4 / np.sqrt(2.0 * NG * PIXEL_SIZE_ARCMIN ** 2), dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, device=device)[None, None].to(torch.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    with torch.no_grad():
        for i0 in range(0, data.shape[0], chunk_ncosmo):
            for j0 in range(0, data.shape[1], chunk_nsys):
                d = torch.as_tensor(data[i0:i0 + chunk_ncosmo, j0:j0 + chunk_nsys], device=device, dtype=torch.float32)
                noise = torch.empty_like(d).normal_(mean=0.0, std=1.0, generator=gen)
                out[i0:i0 + chunk_ncosmo, j0:j0 + chunk_nsys] = (d + (noise * scale) * mask_t).cpu().numpy()
    torch.cuda.synchronize()
    return out


def compute_st(scat_op, kappa, mask, batch_size=8):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st = np.zeros((NCOSMO, NSYS, N_ST))
    for i in tqdm(range(NCOSMO), desc="ST"):
        for j0 in range(0, NSYS, batch_size):
            ref = scat_op.eval(torch.tensor(kappa[i, j0:j0 + batch_size]), mask=mask)
            st[i, j0:j0 + batch_size] = ref.iso_mean().flattenMask().detach().cpu().numpy()
    return st


def main():
    parser = argparse.ArgumentParser(description="Precompute noisy kappa maps and their scattering coefficients.")
    parser.add_argument("--data_dir", default="/rds/fair_challenge/public_data")
    parser.add_argument("--save_dir", default="/rds/fair_challenge/wavelet_scattering_float16")
    parser.add_argument("--n_noisy", type=int, default=150)
    parser.add_argument("--master_seed", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_float32", action="store_true")
    args = parser.parse_args()

    kappa, mask, label = load_train_data(args.data_dir)
    scat_op = sc.funct(NORIENT=4, JmaxDelta=0, padding="same", BACKEND="torch", all_type="float32")
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
            dset_st = h5f.create_dataset("ST_coefficients", shape=(last - first, NCOSMO, NSYS, N_ST), dtype=np.float32,
                                         chunks=(1, NCOSMO, TILE, N_ST), compression=zstd, shuffle=True)
            dset_kappa = h5f.create_dataset("noisy_kappa", shape=(last - first, NCOSMO, NSYS, n_pix), dtype=kappa_dtype,
                                            chunks=(1, NCOSMO, TILE, n_pix), compression=zstd, shuffle=True)
            h5f.create_dataset("labels", data=label, compression=zstd, shuffle=True)
            h5f.create_dataset("mask", data=mask, compression=zstd, shuffle=True)

            for n in range(first, last):
                seed = args.master_seed + n
                print(f"Realisation {n + 1}/{args.n_noisy} (seed {seed})")
                noisy = add_noise(kappa, mask, seed)
                dset_st[n % PER_FILE] = compute_st(scat_op, noisy, mask.reshape(1, *mask.shape), args.batch_size).astype(np.float32)
                dset_kappa[n % PER_FILE] = noisy[:, :, mask_b].astype(kappa_dtype)
                h5f.flush()
                os.fsync(h5f.id.get_vfd_handle())
                del noisy
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
