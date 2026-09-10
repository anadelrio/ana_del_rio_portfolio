import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def batch_ssim_psnr(recon: torch.Tensor, original: torch.Tensor):
    """recon, original: (B, 1, 28, 28) en [0, 1]. Devuelve (ssim_medio, psnr_medio)."""
    recon_np = recon.detach().cpu().numpy()
    orig_np = original.detach().cpu().numpy()
    ssim_vals, psnr_vals = [], []
    for i in range(recon_np.shape[0]):
        r, o = recon_np[i, 0], orig_np[i, 0]
        ssim_vals.append(ssim(o, r, data_range=1.0))
        psnr_vals.append(psnr(o, r, data_range=1.0))
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))
