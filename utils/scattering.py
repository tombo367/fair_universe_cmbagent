import numpy as np
import torch
import foscat.scat_cov2D as sc

N_COEFFS = 630


def make_scat_op():
    return sc.funct(NORIENT=4, JmaxDelta=0, padding="same", BACKEND="torch", all_type="float32")


def scattering_coefficients(scat_op, maps, mask, batch_size=8):
    """Isotropic scattering-covariance coefficients for maps of shape (N, H, W); returns (N, 630)."""
    mask = mask.reshape(1, *mask.shape)
    out = np.zeros((maps.shape[0], N_COEFFS), dtype=np.float32)
    for i0 in range(0, maps.shape[0], batch_size):
        ref = scat_op.eval(torch.tensor(maps[i0:i0 + batch_size]), mask=mask)
        out[i0:i0 + batch_size] = ref.iso_mean().flattenMask().detach().cpu().numpy()
    return out
