import os
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import foscat.scat_cov2D as sc
from fair_universe import Data, Utility

from train_ensemble import (
    CosmoNet, CosmoNetWSTwithError, CosmologyDataset, KappaTransform, WstPCATransform,
    build_ensemble_config, get_h5_paths, load_scaler_npz, load_wst_pca_npz,
    set_global_determinism, strip_prefix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBMISSIONS_DIR = "submissions"
WORKERS_VAL = 0
WORKERS_TEST = 4

SHRINKAGE_LAMBDA = 0.15
SMOOTH_BW_FRAC = 0.20
TARGET_DF = 2.2

TTA_VIEWS = ("id", "flipH", "flipV", "rot180", "transpose", "rot90", "rot270", "transpose_flip")


# ---------------- likelihood ----------------

def ensure_spd(cov, eps=1e-8):
    cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
    tr = np.trace(cov, axis1=-2, axis2=-1)
    ridge = (1e-4 * tr / cov.shape[-1] + eps)[..., None, None]
    return cov + ridge * np.eye(cov.shape[-1], dtype=cov.dtype)[None, ...]


def shrink_covariance(cov, lam):
    cov = cov.astype(np.float64)
    diag = np.zeros_like(cov)
    diag[..., 0, 0] = cov[..., 0, 0]
    diag[..., 1, 1] = cov[..., 1, 1]
    return ensure_spd((1.0 - lam) * cov + lam * diag)


def pairwise_sq_dists(X):
    XX = np.sum(X * X, axis=1, keepdims=True)
    D = XX + XX.T - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    return D


def kernel_smooth_mu_cov(mu, cov, theta, bw_frac):
    G = theta.shape[0]
    D = pairwise_sq_dists(theta)
    nn5_sq = D[np.arange(G), np.argsort(D, axis=1)[:, 5]]
    h2 = (bw_frac * float(np.median(np.sqrt(nn5_sq + 1e-12)))) ** 2 + 1e-12
    W = np.exp(-0.5 * D / h2)
    W /= W.sum(axis=1, keepdims=True) + 1e-12
    mu_sm = W @ mu
    M2_sm = np.einsum("gh,hij->gij", W, cov + np.einsum("gi,gj->gij", mu, mu))
    return mu_sm, ensure_spd(M2_sm - np.einsum("gi,gj->gij", mu_sm, mu_sm))


def logpdf_mvn(d, mean, cov):
    L = np.linalg.cholesky(cov.astype(np.float64))
    y = np.linalg.solve(L, (d - mean)[..., None]).squeeze(-1)
    maha = np.sum(y ** 2, axis=-1)
    logdet = 2.0 * np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(axis=-1)
    return -0.5 * (maha + logdet + 2 * np.log(2 * np.pi))


def calibrate_likelihood(y_pred, y_true, grid, ridge=1e-4):
    row_to_i = {tuple(grid[i]): i for i in range(grid.shape[0])}
    groups = [[] for _ in range(grid.shape[0])]
    for k in range(y_true.shape[0]):
        groups[row_to_i[tuple(y_true[k])]].append(k)

    G, p = grid.shape[0], 2
    mean_d = np.zeros((G, p))
    cov_d = np.zeros((G, p, p))
    for i in range(G):
        di = y_pred[np.array(groups[i], dtype=np.int64)]
        mu = di.mean(axis=0)
        Xc = di - mu
        cov = (Xc.T @ Xc) / float(max(len(di) - p - 2, 1))
        mean_d[i] = mu
        cov_d[i] = cov + np.eye(p) * (ridge * np.trace(cov) / p + 1e-8)
    return mean_d, cov_d


def temperature_from_residuals(mean_d, cov_d, y_pred, y_true, grid, target_df):
    idx_map = {(float(g[0]), float(g[1])): i for i, g in enumerate(grid)}
    gi = np.array([idx_map[(float(t[0]), float(t[1]))] for t in y_true], dtype=int)
    L = np.linalg.cholesky(ensure_spd(cov_d[gi]))
    y = y_pred - mean_d[gi]
    z0 = y[:, 0] / L[:, 0, 0]
    z1 = (y[:, 1] - L[:, 1, 0] * z0) / L[:, 1, 1]
    return np.sqrt(max(float(np.mean(z0 * z0 + z1 * z1)) / target_df, 1e-6))


def member_weights_unsup(mean_d, cov_d, preds_list):
    nll = []
    for yp in preds_list:
        ll = logpdf_mvn(yp[:, None, :], mean_d[None], cov_d[None])
        a = ll.max(axis=1, keepdims=True)
        nll.append(-float(np.mean(a + np.log(np.exp(ll - a).mean(axis=1, keepdims=True) + 1e-300))))
    nll = np.array(nll)
    w = np.exp(-nll - nll.min())
    return w / w.sum()


def grid_posterior(mean_d, cov_d, grid, d_obs, eps=1e-10):
    G, N = grid.shape[0], d_obs.shape[0]
    mu_out = np.empty((N, 2))
    sigma_out = np.empty((N, 2))
    for i in range(N):
        ll = logpdf_mvn(d_obs[i:i + 1, None, :], mean_d[None], cov_d[None])[0]
        w = np.exp(ll - ll.max())
        ws = w.sum()
        w = np.ones(G) / G if (not np.isfinite(ws) or ws <= 0.0) else w / ws
        tmean = (w[:, None] * grid).sum(axis=0)
        c = grid - tmean
        tcov = c.T @ (c * w[:, None])
        mu_out[i] = tmean
        sigma_out[i] = np.sqrt(np.clip([tcov[0, 0], tcov[1, 1]], eps, None))
    return mu_out, sigma_out


# ---------------- model inference ----------------

def apply_tta(xb, view):
    if view == "flipH":
        return xb.flip(-1)
    if view == "flipV":
        return xb.flip(-2)
    if view == "rot180":
        return xb.flip(-1).flip(-2)
    if view == "transpose":
        return xb.transpose(-1, -2)
    if view == "rot90":
        return torch.rot90(xb, 1, dims=(-2, -1))
    if view == "rot270":
        return torch.rot90(xb, 3, dims=(-2, -1))
    if view == "transpose_flip":
        return xb.transpose(-1, -2).flip(-1)
    return xb


@torch.no_grad()
def predict(model, loader, scaler):
    outs = []
    for batch in loader:
        xb = batch[0].to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        wb = batch[1].to(DEVICE, non_blocking=True)
        preds = [scaler.inverse_transform(model(apply_tta(xb, v), wb)[0].cpu().numpy()) for v in TTA_VIEWS]
        outs.append(np.mean(preds, axis=0))
    return np.concatenate(outs, axis=0)


def load_member(cfg, wst_dim):
    base_cnn = CosmoNet(cfg["backbone"]).to(DEVICE).to(memory_format=torch.channels_last)
    m = CosmoNetWSTwithError(base_cnn, wst_dim=wst_dim, hidden=64, alpha_init=0.01,
                             freeze_cnn=True, dropout=0.5).to(DEVICE).to(memory_format=torch.channels_last)
    state = torch.load(f"models/model_{cfg['model_id']}_wst_residual_with_error.pth", map_location=DEVICE)
    m.load_state_dict(strip_prefix(state), strict=True)
    return m.eval()


def compute_st(kappa, mask, batch_size=8):
    scat_op = sc.funct(NORIENT=4, JmaxDelta=0, padding="same", BACKEND="torch", all_type="float32")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st = np.zeros((kappa.shape[0], 630), dtype=np.float32)
    mask = mask.reshape(1, *mask.shape)
    for i0 in range(0, kappa.shape[0], batch_size):
        ref = scat_op.eval(torch.tensor(kappa[i0:i0 + batch_size]), mask=mask)
        st[i0:i0 + batch_size] = ref.iso_mean().flattenMask().detach().cpu().numpy()
    return st


class TestKappaDataset(Dataset):
    """Test maps are already noisy, so they are only masked and standardised."""
    def __init__(self, kappa_test, mask, mean_img, std_img, st):
        self.X = kappa_test.astype(np.float32)
        self.mask = mask.astype(np.float32)
        self.mean = float(mean_img)
        self.std = float(std_img) if abs(std_img) > 0 else 1e-6
        self.st = st

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img = (self.X[idx] * self.mask - self.mean) / self.std
        return torch.from_numpy(img[None, ...]), torch.from_numpy(self.st[idx])


def main():
    set_global_determinism(1234)
    h5_paths = get_h5_paths()

    data = Data()
    data.load_train_data()
    data.load_test_data()
    grid = data.label[:, 0, :2].astype(np.float64)

    wst_scaler, wst_pca = load_wst_pca_npz("data/wst_pca.npz")
    wst_transform = WstPCATransform(wst_scaler, wst_pca)
    wst_dim = wst_pca.n_components_
    cfgs = build_ensemble_config()

    # validation predictions, used to calibrate the likelihood
    val_preds, val_truth, scalers, stats = [], [], [], []
    for cfg in cfgs:
        mid = cfg["model_id"]
        val_idx = np.load(f"data/member_{mid:02d}_internal_split.npz")["val_internal"].astype(int)
        s = np.load(f"data/img_stats_noisy_{mid}.npz")
        mean_img, std_img = float(s["mean"]), float(s["std"])
        scaler = load_scaler_npz(f"data/label_scaler_{mid}.npz")

        ds = CosmologyDataset(h5_paths, val_idx, 0, KappaTransform(mean_img, std_img), wst_transform,
                              scaler, rng=np.random.default_rng(2))
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=WORKERS_VAL, pin_memory=True)
        val_preds.append(predict(load_member(cfg, wst_dim), loader, scaler))
        val_truth.append(data.label[:, val_idx, :2].reshape(-1, 2).astype(np.float64))
        scalers.append(scaler)
        stats.append((mean_img, std_img))

    y_pred_val = np.concatenate(val_preds, axis=0)
    y_val = np.concatenate(val_truth, axis=0)
    mean_d, cov_d = calibrate_likelihood(y_pred_val, y_val, grid)
    cov_d = shrink_covariance(cov_d, SHRINKAGE_LAMBDA)
    mean_d, cov_d = kernel_smooth_mu_cov(mean_d, cov_d, grid, SMOOTH_BW_FRAC)
    tau = temperature_from_residuals(mean_d, cov_d, y_pred_val, y_val, grid, TARGET_DF)
    cov_d = cov_d * tau ** 2
    print(f"Temperature tau = {tau:.3f}")

    # test predictions
    st_test = wst_transform.transform(compute_st(data.kappa_test.astype(np.float32) * data.mask.astype(np.float32)[None], data.mask)).astype(np.float32)
    test_preds = []
    for cfg, (mean_img, std_img), scaler in zip(cfgs, stats, scalers):
        ds = TestKappaDataset(data.kappa_test, data.mask, mean_img, std_img, st_test)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=WORKERS_TEST, pin_memory=True)
        test_preds.append(predict(load_member(cfg, wst_dim), loader, scaler))

    w = member_weights_unsup(mean_d, cov_d, test_preds)
    print("Member weights:", np.round(w, 4))
    y_pred_test = np.tensordot(w, np.stack(test_preds, axis=0), axes=(0, 0))
    mu, sigma = grid_posterior(mean_d, cov_d, grid, y_pred_test)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    zip_path = Utility.save_json_zip(
        submission_dir=SUBMISSIONS_DIR,
        json_file_name="result.json",
        zip_file_name=f"Submission_{stamp}.zip",
        data={"means": mu.tolist(), "errorbars": sigma.tolist()},
    )
    np.savez("data/test_eval.npz", ens_pred_test=y_pred_test, ens_mu_test=mu, ens_sigma_test=sigma,
             cosmology_grid=grid, ens_mean=mean_d, ens_cov=cov_d, member_weights=w)

    print(f"Saved {zip_path}")
    print(f"Omega_m: mean {mu[:, 0].mean():.4f}, mean sigma {sigma[:, 0].mean():.4f}")
    print(f"S8:      mean {mu[:, 1].mean():.4f}, mean sigma {sigma[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
