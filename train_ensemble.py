import os
import sys
import time
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import h5py
import hdf5plugin

from utils.data import Data
from utils.noise import noise_scale

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

WST_DATA_DIR = "/rds/fair_challenge/wavelet_scattering_float16"

NUM_ENSEMBLE = 10
EPOCHS = 50
PRETRAIN_EPOCHS = 20
BACKBONES = ["klite_inception", "klite_inception_se"]
BATCH_SIZES = [32, 48, 64, 96, 128]
INTERNAL_VAL_FRAC = 0.20

BASE_LR = 2e-4
BASE_LR_WST = 1e-5
WEIGHT_DECAY = 1e-4
DATALOADER_WORKERS = 4
PIN_MEMORY = True

TRAIN_VIEWS = ("id", "flipH", "flipV", "rot180", "transpose", "rot90", "rot270", "transpose_flip")


def set_global_determinism(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stdsafe(s):
    return s if s > 1e-8 else 1e-8


def get_h5_paths(data_dir=WST_DATA_DIR):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]


def strip_prefix(state_dict, prefix="_orig_mod."):
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def build_ensemble_config():
    return [
        {
            "model_id": i,
            "backbone": BACKBONES[i % len(BACKBONES)],
            "batch_size": BATCH_SIZES[i % len(BATCH_SIZES)],
            "epochs": EPOCHS,
            "pretrain_epochs": PRETRAIN_EPOCHS,
        }
        for i in range(NUM_ENSEMBLE)
    ]


# ---------------- data ----------------

class CosmoDataset(Dataset):
    """Kappa maps with optional on-the-fly shape noise (inside the mask) and random H/V flips."""
    def __init__(self, kappa, mask, idx_pairs, labels, ng, pixel_size_arcmin,
                 mean_img, std_img, label_scaler, add_noise, train_mode, rng_seed=0):
        self.kappa = kappa
        self.mask = mask.astype(np.float32)
        self.idx_pairs = idx_pairs.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.scale = float(noise_scale(float(ng), float(pixel_size_arcmin)))
        self.mean_img = float(mean_img)
        self.std_img = float(stdsafe(std_img))
        self.label_scaler = label_scaler
        self.add_noise = add_noise
        self.train_mode = train_mode
        self.to_tensor = transforms.ToTensor()
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self):
        return self.idx_pairs.shape[0]

    def __getitem__(self, idx):
        ic, isys = int(self.idx_pairs[idx, 0]), int(self.idx_pairs[idx, 1])
        arr = self.kappa[ic, isys].astype(np.float32)
        if self.add_noise:
            noise = self.rng.standard_normal(arr.shape).astype(np.float32) * self.scale
            arr = arr + noise * self.mask
        if self.train_mode:
            if self.rng.random() > 0.5:
                arr = np.flip(arr, axis=1).copy()
            if self.rng.random() > 0.5:
                arr = np.flip(arr, axis=0).copy()
        t = (self.to_tensor(arr) - self.mean_img) / self.std_img
        y = self.label_scaler.transform(self.labels[ic, isys, :2].reshape(1, -1)).astype(np.float32).squeeze(0)
        return t, torch.from_numpy(y)


class CosmologyDataset(Dataset):
    """
    Pre-noised kappa maps + scattering coefficients from the HDF5 files. Each epoch uses a
    different noise realisation (and the file changes every 10 epochs); the next epoch is
    prefetched in a background thread.
    """
    def __init__(self, h5_paths, n_sys_indices, epoch=0, kappa_transform=None,
                 st_transform=None, label_transform=None, rng=None):
        self.h5_paths = h5_paths
        self.n_sys_indices = sorted(n_sys_indices)
        self.n_sys_subset = len(self.n_sys_indices)

        with h5py.File(self.h5_paths[0], "r") as f:
            self.st_shape = f["ST_coefficients"].shape
            self.kappa_shape = f["noisy_kappa"].shape
            self.kappa_dtype = f["noisy_kappa"].dtype
            self.st_dtype = f["ST_coefficients"].dtype
            self.n_noisy, self.n_cosmo, self.n_sys_total, _ = self.st_shape
            self.total_samples = self.n_cosmo * self.n_sys_subset
            self.mask = f["mask"][:].astype(bool)
            self.labels = f["labels"][:]

        self.epoch = epoch
        self.rng = rng or np.random.default_rng()
        self.n_indices = self.rng.permutation(10)
        self.path_idx = self.epoch // 10 % len(h5_paths)
        self.prefetch_ready = threading.Event()
        self.kappa_transform = kappa_transform
        self.st_transform = st_transform
        self.label_transform = label_transform
        self._load_epoch_data()
        self.prefetch_ready.set()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        cosmo_idx = index // self.n_sys_subset
        sys_idx = self.n_sys_indices[index % self.n_sys_subset]

        kappa = np.zeros(self.mask.shape, dtype=np.float32)
        kappa[self.mask] = self.cached_kappa[cosmo_idx, sys_idx, :]
        st = self.cached_st[cosmo_idx, sys_idx, :].reshape(1, -1)
        label = self.labels[cosmo_idx, sys_idx, :2].reshape(-1, 2)

        kappa_t = self.kappa_transform(kappa) if self.kappa_transform is not None else torch.from_numpy(kappa)
        if self.st_transform is not None:
            st = self.st_transform.transform(st)
        if self.label_transform is not None:
            label = self.label_transform.transform(label)

        return kappa_t, torch.from_numpy(st[0].astype(np.float32)), torch.from_numpy(label[0].astype(np.float32))

    def _read(self, path, noisy_idx):
        kappa = np.zeros(self.kappa_shape[1:], dtype=self.kappa_dtype)
        st = np.zeros(self.st_shape[1:], dtype=self.st_dtype)
        with h5py.File(path, "r") as f:
            for i in range(0, self.st_shape[2], 32):
                kappa[:, i:i + 32] = f["noisy_kappa"][noisy_idx, :, i:i + 32]
                st[:, i:i + 32] = f["ST_coefficients"][noisy_idx, :, i:i + 32]
        return kappa, st

    def _load_epoch_data(self):
        t0 = time.time()
        self.cached_kappa, self.cached_st = self._read(
            self.h5_paths[self.path_idx], self.n_indices[self.epoch % 10])
        print(f"Loaded {self.cached_kappa.nbytes / 1e9:.2f} GB in {time.time() - t0:.1f}s")

    def _prefetch_next_epoch(self):
        nxt = self.epoch + 1
        path = self.h5_paths[nxt // 10 % len(self.h5_paths)]
        self.prefetch_kappa, self.prefetch_st = self._read(path, self.n_indices[nxt % 10])
        self.prefetch_ready.set()

    def set_epoch(self, epoch):
        self.prefetch_ready.wait()
        self.epoch = epoch
        if hasattr(self, "prefetch_kappa"):
            self.cached_kappa = self.prefetch_kappa
            self.cached_st = self.prefetch_st
        if self.epoch % 10 == 0 and self.epoch > 0:
            self.n_indices = self.rng.permutation(10)
        self.prefetch_ready.clear()
        threading.Thread(target=self._prefetch_next_epoch, daemon=True).start()


class KappaTransform:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(stdsafe(std))
        self.to_tensor = transforms.ToTensor()

    def __call__(self, arr):
        return (self.to_tensor(arr.astype(np.float32)) - self.mean) / self.std


class WstPCATransform:
    def __init__(self, scaler, pca):
        self.scaler = scaler
        self.pca = pca

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))


def fit_wst_pca(h5_paths, n_sys_indices, n_components=64, max_samples=100000):
    rng = np.random.default_rng(12345)
    with h5py.File(h5_paths[0], "r") as f:
        st_ds = f["ST_coefficients"]
        n_cosmo = st_ds.shape[1]
        all_pairs = [(i, j) for i in range(n_cosmo) for j in n_sys_indices]
        sel = rng.choice(len(all_pairs), size=min(max_samples, len(all_pairs)), replace=False)
        X = np.stack([st_ds[0, all_pairs[k][0], all_pairs[k][1], :] for k in sel], axis=0).astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, whiten=False)
    pca.fit(Xs)
    return scaler, pca


def save_wst_pca_npz(path, scaler, pca):
    np.savez(path, scaler_mean=scaler.mean_, scaler_var=scaler.var_,
             pca_components=pca.components_, pca_explained_variance=pca.explained_variance_,
             pca_explained_variance_ratio=pca.explained_variance_ratio_, pca_mean=pca.mean_)


def load_wst_pca_npz(path):
    z = np.load(path)
    sc = StandardScaler()
    sc.mean_ = z["scaler_mean"]
    sc.var_ = z["scaler_var"]
    sc.scale_ = np.sqrt(sc.var_)
    sc.n_features_in_ = sc.mean_.shape[0]
    pca = PCA()
    pca.components_ = z["pca_components"]
    pca.explained_variance_ = z["pca_explained_variance"]
    pca.explained_variance_ratio_ = z["pca_explained_variance_ratio"]
    pca.mean_ = z["pca_mean"]
    pca.n_components_ = pca.components_.shape[0]
    pca.n_features_ = pca.components_.shape[1]
    return sc, pca


def save_scaler_npz(path, scaler):
    np.savez(path, mean=scaler.mean_, var=scaler.var_)


def load_scaler_npz(path):
    z = np.load(path)
    sc = StandardScaler()
    sc.mean_ = z["mean"]
    sc.var_ = z["var"]
    sc.scale_ = np.sqrt(sc.var_)
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc


def compute_img_stats_sampled(kappa, mask, idx_pairs, ng, pixel_size_arcmin,
                              max_samples=4000, seed=7777, add_noise=True):
    rng = np.random.default_rng(seed)
    sel = rng.choice(idx_pairs.shape[0], size=min(max_samples, idx_pairs.shape[0]), replace=False)
    scale = float(noise_scale(float(ng), float(pixel_size_arcmin)))
    maskf = mask.astype(np.float32)
    s, s2, n = np.float64(0.0), np.float64(0.0), 0
    for k in sel:
        arr = kappa[int(idx_pairs[k, 0]), int(idx_pairs[k, 1])].astype(np.float32)
        if add_noise:
            arr = arr + rng.standard_normal(arr.shape).astype(np.float32) * scale * maskf
        arr = arr.astype(np.float64, copy=False)
        s += arr.sum()
        s2 += (arr * arr).sum()
        n += arr.size
    mean = float(s / max(n, 1))
    std = math.sqrt(float(max((s2 - s * mean) / (n - 1), 0.0))) if n > 1 else 0.0
    return mean, float(stdsafe(std))


# ---------------- models ----------------

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 4
        self.b1 = ConvBNRelu(in_channels, c, kernel_size=1)
        self.b2 = ConvBNRelu(in_channels, c, kernel_size=3, padding=1)
        self.b3 = ConvBNRelu(in_channels, c, kernel_size=5, padding=2)
        self.b4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_conv = ConvBNRelu(in_channels, c, kernel_size=1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4_conv(self.b4_pool(x))], dim=1)


class KLiteInception(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.stem = ConvBNRelu(1, 32, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.inc2 = InceptionBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.inc3 = InceptionBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.inc4 = InceptionBlock(128, 256)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dim))

    def forward(self, x):
        x = self.pool1(self.stem(x))
        x = self.pool2(self.inc2(x))
        x = self.pool3(self.inc3(x))
        x = self.pool_final(self.inc4(x))
        return self.head(x)


class KLiteInceptionSE(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.stem = ConvBNRelu(1, 48, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.inc2 = InceptionBlock(48, 96)
        self.se2 = SEBlock(96)
        self.pool2 = nn.MaxPool2d(2)
        self.inc3 = InceptionBlock(96, 192)
        self.se3 = SEBlock(192)
        self.pool3 = nn.MaxPool2d(2)
        self.inc4 = InceptionBlock(192, 256)
        self.se4 = SEBlock(256)
        self.pool_final = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 160), nn.ReLU(inplace=True), nn.Linear(160, out_dim))

    def forward(self, x):
        x = self.pool1(self.stem(x))
        x = self.pool2(self.se2(self.inc2(x)))
        x = self.pool3(self.se3(self.inc3(x)))
        x = self.pool_final(self.se4(self.inc4(x)))
        return self.head(x)


class CosmoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.back = {"klite_inception": KLiteInception, "klite_inception_se": KLiteInceptionSE}[backbone](out_dim=2)

    def forward(self, x):
        return self.back(x)


class CosmoNetWSTwithError(nn.Module):
    """CNN prediction plus a small WST-conditioned residual, and a 3-class head on the S8 residual."""
    def __init__(self, base_cnn, wst_dim, hidden=64, alpha_init=0.01, freeze_cnn=True, dropout=0.5, n_classes=3):
        super().__init__()
        self.cnn = base_cnn
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        self.wst_mlp = nn.Sequential(
            nn.Linear(wst_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(wst_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        nn.init.zeros_(self.wst_mlp[-1].bias)

    def forward(self, x_kappa, x_wst):
        y_cnn = self.cnn(x_kappa)
        inp = torch.cat([x_wst, y_cnn], dim=-1)
        y_pred = y_cnn + self.alpha * self.wst_mlp(inp)
        return y_pred, self.cls_head(inp), y_cnn


# ---------------- training ----------------

def apply_view(xb, view):
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


def apply_random_view_batch(xb, views=TRAIN_VIEWS):
    v = views[torch.randint(low=0, high=len(views), size=(1,)).item()]
    return apply_view(xb, v).contiguous(memory_format=torch.channels_last)


def residual_to_class(y_true, y_cnn, delta=0.5, param_idx=1):
    e = (y_true - y_cnn)[:, param_idx]
    cls = torch.empty_like(e, dtype=torch.long)
    cls[e < -delta] = 0
    cls[torch.abs(e) <= delta] = 1
    cls[e > delta] = 2
    return cls


def train_one(model, train_loader, val_loader, epochs, base_lr, weight_decay, device, model_path):
    lr = base_lr * (train_loader.batch_size / 64.0)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        total, cnt = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = apply_random_view_batch(xb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())
            cnt += 1

        model.eval()
        vtotal, vcnt = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                vtotal += float(loss_fn(model(xb), yb).cpu().item())
                vcnt += 1
        val_loss = vtotal / max(vcnt, 1)
        sch.step(val_loss)
        print(f"Epoch {ep+1}/{epochs} | train {total / max(cnt, 1):.6f} | val {val_loss:.6f} | lr {opt.param_groups[0]['lr']}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), model_path)
    return best


def eval_cnn_only(cnn_model, val_loader, device):
    cnn_model.eval()
    loss_fn = nn.MSELoss()
    total, cnt = 0.0, 0
    with torch.no_grad():
        for xb, _, yb in val_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            total += float(loss_fn(cnn_model(xb), yb).cpu().item())
            cnt += 1
    return total / max(cnt, 1)


def train_one_wst(model, train_loader, val_loader, epochs, base_lr, weight_decay, device, model_path):
    lr_wst = base_lr * (train_loader.batch_size / 64.0)
    lr_cnn = lr_wst * 0.1
    lambda_res, lambda_cls, delta_thresh = 1e-3, 0.1, 0.5

    cnn_params, wst_params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (cnn_params if name.startswith("cnn.") else wst_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": wst_params, "lr": lr_wst}, {"params": cnn_params, "lr": lr_cnn}],
        weight_decay=weight_decay,
    )
    sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for ep in range(epochs):
        train_loader.dataset.set_epoch(ep)
        val_loader.dataset.set_epoch(ep)

        model.train()
        total, cnt = 0.0, 0
        for xb, wb, yb in train_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = apply_random_view_batch(xb)

            y_pred, cls_logits, y_cnn = model(xb, wb)
            loss_reg = loss_fn(y_pred, yb)
            loss_res = ((y_pred - y_cnn) ** 2).mean()
            cls_targets = residual_to_class(yb, y_cnn.detach(), delta=delta_thresh, param_idx=1)
            loss_cls = F.cross_entropy(cls_logits, cls_targets)
            loss = loss_reg + lambda_res * loss_res + lambda_cls * loss_cls

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())
            cnt += 1

        model.eval()
        vtotal, vcnt = 0.0, 0
        with torch.no_grad():
            for xb, wb, yb in val_loader:
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                wb = wb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_pred, _, _ = model(xb, wb)
                vtotal += float(loss_fn(y_pred, yb).cpu().item())
                vcnt += 1
        val_loss = vtotal / max(vcnt, 1)
        sch.step(val_loss)
        print(f"[WST] Epoch {ep+1}/{epochs} | train {total / max(cnt, 1):.6f} | val {val_loss:.6f} | lr {opt.param_groups[0]['lr']}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), model_path)
    return best


def train_cnn_members(device):
    set_global_determinism(1234)
    data = Data()
    data.load_train_data()
    kappa = data.kappa.astype(np.float32)
    mask = data.mask.astype(np.float32)
    labels = data.label.astype(np.float32)
    Ncosmo, Nsys = data.Ncosmo, data.Nsys
    all_sys_idx = np.arange(Nsys)

    def make_ds(pairs, mean, std, scaler, add_noise, train_mode, seed):
        return CosmoDataset(kappa, mask, pairs, labels, data.ng, data.pixelsize_arcmin,
                            mean, std, scaler, add_noise=add_noise, train_mode=train_mode, rng_seed=seed)

    def make_loader(ds, bs, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=DATALOADER_WORKERS,
                          pin_memory=PIN_MEMORY, persistent_workers=False)

    best_losses = []
    for cfg in build_ensemble_config():
        mid, bs = cfg["model_id"], cfg["batch_size"]
        print(f"\n=== CNN member {mid} | {cfg['backbone']} | batch {bs} ===")
        set_global_determinism(1234 + mid)

        tr_sys_idx, val_sys_idx = train_test_split(all_sys_idx, test_size=INTERNAL_VAL_FRAC, random_state=5566 + mid)
        np.savez(f"data/member_{mid:02d}_internal_split.npz", train=tr_sys_idx, val_internal=val_sys_idx)
        pairs_train = np.array([(i, j) for i in range(Ncosmo) for j in tr_sys_idx], dtype=np.int64)
        pairs_val = np.array([(i, j) for i in range(Ncosmo) for j in val_sys_idx], dtype=np.int64)

        label_scaler = StandardScaler()
        label_scaler.fit(labels[:, tr_sys_idx, :2].reshape(-1, 2).astype(np.float32))
        save_scaler_npz(f"data/label_scaler_{mid}.npz", label_scaler)

        mean_clean, std_clean = compute_img_stats_sampled(
            kappa, mask, pairs_train, data.ng, data.pixelsize_arcmin, seed=10000 + mid, add_noise=False)
        np.savez(f"data/img_stats_clean_{mid}.npz", mean=mean_clean, std=std_clean)
        mean_noisy, std_noisy = compute_img_stats_sampled(
            kappa, mask, pairs_train, data.ng, data.pixelsize_arcmin, seed=20000 + mid, add_noise=True)
        np.savez(f"data/img_stats_noisy_{mid}.npz", mean=mean_noisy, std=std_noisy)

        model = CosmoNet(cfg["backbone"]).to(device).to(memory_format=torch.channels_last)
        model = torch.compile(model)

        # pretrain on noiseless maps
        pre_path = f"models/model_{mid}_pretrain.pth"
        train_one(
            model,
            make_loader(make_ds(pairs_train, mean_clean, std_clean, label_scaler, False, True, 31000 + mid), bs, True),
            make_loader(make_ds(pairs_val, mean_clean, std_clean, label_scaler, False, False, 32000 + mid), bs, False),
            cfg["pretrain_epochs"], BASE_LR, WEIGHT_DECAY, device, pre_path,
        )
        model.load_state_dict(torch.load(pre_path, map_location=device))

        # then on noisy maps
        best_val = train_one(
            model,
            make_loader(make_ds(pairs_train, mean_noisy, std_noisy, label_scaler, True, True, 41000 + mid), bs, True),
            make_loader(make_ds(pairs_val, mean_noisy, std_noisy, label_scaler, True, False, 42000 + mid), bs, False),
            cfg["epochs"], BASE_LR, WEIGHT_DECAY, device, f"models/model_{mid}.pth",
        )
        best_losses.append(best_val)
        np.savez(
            f"data/member_{mid:02d}_final_meta.npz",
            best_val=np.array(best_val, dtype=np.float32),
            train_sys_idx=tr_sys_idx, val_sys_idx=val_sys_idx,
            img_mean_clean=mean_clean, img_std_clean=std_clean,
            img_mean_noisy=mean_noisy, img_std_noisy=std_noisy,
        )

    print("Best val MSE per member:", [round(x, 6) for x in best_losses])
    np.savez("data/ensemble_summary.npz", best_losses=np.array(best_losses, dtype=np.float32))


def train_wst_members(device):
    set_global_determinism(1234)
    h5_paths = get_h5_paths()

    for cfg in build_ensemble_config():
        mid, bs = cfg["model_id"], cfg["batch_size"]
        print(f"\n=== WST head member {mid} | {cfg['backbone']} | batch {bs} ===")

        split = np.load(f"data/member_{mid:02d}_internal_split.npz")
        stats = np.load(f"data/img_stats_noisy_{mid}.npz")
        label_scaler = load_scaler_npz(f"data/label_scaler_{mid}.npz")

        # PCA is fit once, on member 0's training systems, and reused
        wst_pca_path = "data/wst_pca.npz"
        if os.path.exists(wst_pca_path):
            wst_scaler, wst_pca = load_wst_pca_npz(wst_pca_path)
        else:
            wst_scaler, wst_pca = fit_wst_pca(h5_paths, split["train"], n_components=64)
            save_wst_pca_npz(wst_pca_path, wst_scaler, wst_pca)
        wst_transform = WstPCATransform(wst_scaler, wst_pca)

        kappa_transform = KappaTransform(float(stats["mean"]), float(stats["std"]))
        train_ds = CosmologyDataset(h5_paths, split["train"], 0, kappa_transform, wst_transform,
                                    label_scaler, rng=np.random.default_rng(1000 + mid))
        val_ds = CosmologyDataset(h5_paths, split["val_internal"], 0, kappa_transform, wst_transform,
                                  label_scaler, rng=np.random.default_rng(2000 + mid))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=PIN_MEMORY)

        base_cnn = CosmoNet(cfg["backbone"]).to(device).to(memory_format=torch.channels_last)
        base_cnn.load_state_dict(strip_prefix(torch.load(f"models/model_{mid}.pth", map_location=device)))
        print(f"CNN-only val MSE: {eval_cnn_only(base_cnn, val_loader, device):.6f}")

        model = CosmoNetWSTwithError(base_cnn, wst_dim=wst_pca.n_components_, hidden=64,
                                     alpha_init=0.01, freeze_cnn=False, dropout=0.5).to(device)
        best = train_one_wst(model, train_loader, val_loader, cfg["epochs"] // 2, BASE_LR_WST, WEIGHT_DECAY,
                             device, f"models/model_{mid}_wst_residual_with_error.pth")
        print(f"Member {mid}: best WST val MSE = {best:.6f}")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if stage in ("cnn", "all"):
        train_cnn_members(device)
    if stage in ("wst", "all"):
        train_wst_members(device)


if __name__ == "__main__":
    main()
