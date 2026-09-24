import numpy as np
import torch


def noise_scale(ng, pixel_size_arcmin):
    return 0.4 / np.sqrt(2.0 * ng * pixel_size_arcmin ** 2)


def add_noise(data, mask, ng, pixel_size=2.0, seed=None, device="cuda", chunk_ncosmo=32, chunk_nsys=256):
    """
    Gaussian shape noise added under the mask on the GPU, for maps of shape (Ncosmo, Nsys, H, W).
    Noise comes from one seeded stream in chunk order, so the chunk sizes are part of the result.
    """
    out = np.empty_like(data, dtype=np.float32)
    scale = torch.tensor(noise_scale(ng, pixel_size), dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, device=device)[None, None].to(torch.float32)
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    with torch.no_grad():
        for i0 in range(0, data.shape[0], chunk_ncosmo):
            for j0 in range(0, data.shape[1], chunk_nsys):
                d = torch.as_tensor(data[i0:i0 + chunk_ncosmo, j0:j0 + chunk_nsys], device=device, dtype=torch.float32)
                noise = torch.empty_like(d).normal_(mean=0.0, std=1.0, generator=gen)
                out[i0:i0 + chunk_ncosmo, j0:j0 + chunk_nsys] = (d + (noise * scale) * mask_t).cpu().numpy()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out
