"""Evaluación unificada de los 3 modelos sobre el conjunto de test.

A diferencia del informe original, aquí NO se suman términos de naturaleza
distinta en un único "Total Loss" (el original comparaba ELBO=BCE+KL del
VAE contra BCE+commitment del VQ-VAE contra solo-BCE del Geo-VQ, lo cual
favorecía artificialmente a este último). En su lugar se reporta cada
término por separado, más SSIM/PSNR como métricas de calidad perceptual
independientes de cómo se entrenó cada modelo.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import MNISTDataset
from models.vae import VAE
from models.vqvae import VQVAE
from evaluation.metrics import batch_ssim_psnr
from utils.paths import CHECKPOINTS_DIR, LATENTS_DIR, FIGURES_DIR, LOGS_DIR


@torch.no_grad()
def evaluate_vae(vae, loader, device):
    bce_tot = kl_tot = n = 0.0
    ssim_vals, psnr_vals = [], []
    for x, _ in loader:
        x = x.to(device)
        recon, mu, logvar = vae(x)
        bce = F.binary_cross_entropy(recon, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        bce_tot += bce.item()
        kl_tot += kl.item()
        n += x.size(0)
        s, p = batch_ssim_psnr(recon, x)
        ssim_vals.append(s); psnr_vals.append(p)
    return {"bce_per_img": bce_tot / n, "kl_per_img": kl_tot / n,
           "ssim": float(np.mean(ssim_vals)), "psnr": float(np.mean(psnr_vals))}


@torch.no_grad()
def evaluate_vqvae(vqvae, loader, device):
    bce_tot = commit_tot = n = 0.0
    ssim_vals, psnr_vals, perp_vals = [], [], []
    for x, _ in loader:
        x = x.to(device)
        recon, commitment, _, perplexity = vqvae(x)
        bce = F.binary_cross_entropy(recon, x, reduction="sum")
        bce_tot += bce.item()
        commit_tot += commitment.item() * x.size(0)
        n += x.size(0)
        perp_vals.append(perplexity.item())
        s, p = batch_ssim_psnr(recon, x)
        ssim_vals.append(s); psnr_vals.append(p)
    return {"bce_per_img": bce_tot / n, "commit_per_img": commit_tot / n,
           "ssim": float(np.mean(ssim_vals)), "psnr": float(np.mean(psnr_vals)),
           "perplexity": float(np.mean(perp_vals))}


@torch.no_grad()
def evaluate_geovq(vae, loader, device, centroids: np.ndarray):
    """Reconstruye cada imagen de test asignándola a su centroide geodésico
    más cercano en el espacio latente del VAE, y decodificando ese centroide."""
    centroids_t = torch.from_numpy(centroids).to(device)
    bce_tot = n = 0.0
    ssim_vals, psnr_vals = [], []
    for x, _ in loader:
        x = x.to(device)
        mu, _ = vae.encode(x)
        dists = torch.cdist(mu, centroids_t)
        nearest = dists.argmin(dim=1)
        z_q = centroids_t[nearest]
        recon = vae.decode(z_q)
        bce = F.binary_cross_entropy(recon, x, reduction="sum")
        bce_tot += bce.item()
        n += x.size(0)
        s, p = batch_ssim_psnr(recon, x)
        ssim_vals.append(s); psnr_vals.append(p)
    return {"bce_per_img": bce_tot / n,
           "ssim": float(np.mean(ssim_vals)), "psnr": float(np.mean(psnr_vals))}


def make_comparison_figure(vae, vqvae, geo_centroids, euc_centroids, dataset, device, n=8):
    idx = np.random.RandomState(0).choice(len(dataset), n, replace=False)
    x = torch.stack([dataset[i][0] for i in idx]).to(device)

    with torch.no_grad():
        vae_recon, mu, _ = vae(x)
        vqvae_recon, _, _, _ = vqvae(x)

        geo_c = torch.from_numpy(geo_centroids).to(device)
        nearest_geo = torch.cdist(mu, geo_c).argmin(dim=1)
        geo_recon = vae.decode(geo_c[nearest_geo])

        euc_c = torch.from_numpy(euc_centroids).to(device)
        nearest_euc = torch.cdist(mu, euc_c).argmin(dim=1)
        euc_recon = vae.decode(euc_c[nearest_euc])

    rows = [("Original", x), ("VAE", vae_recon), ("VQ-VAE", vqvae_recon),
           ("Geo-VQ", geo_recon), ("Eucl-VQ", euc_recon)]

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.2, len(rows) * 1.3))
    for r, (name, imgs) in enumerate(rows):
        imgs = imgs.cpu().numpy()
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=11, rotation=0, ha="right", va="center")
    plt.tight_layout()
    out = FIGURES_DIR / "comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Figura guardada en", out)


def main(args):
    device = torch.device("cpu")
    dataset = MNISTDataset(train=False)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    vae = VAE(latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(CHECKPOINTS_DIR / "vae_mnist.pt", map_location=device))
    vae.eval()

    vqvae = VQVAE(embedding_dim=args.embedding_dim, num_embeddings=args.embeddings).to(device)
    vqvae.load_state_dict(torch.load(CHECKPOINTS_DIR / "vqvae_mnist.pt", map_location=device))
    vqvae.eval()

    geo_centroids = np.load(LATENTS_DIR / "geodesic_centroids.npy")
    euc_centroids = np.load(LATENTS_DIR / "euclidean_centroids.npy")

    print("Evaluando VAE...")
    vae_metrics = evaluate_vae(vae, loader, device)
    print("Evaluando VQ-VAE...")
    vqvae_metrics = evaluate_vqvae(vqvae, loader, device)
    print("Evaluando Geo-VQ (VAE + centroides geodésicos)...")
    geovq_metrics = evaluate_geovq(vae, loader, device, geo_centroids)
    print("Evaluando Eucl-VQ (VAE + centroides euclídeos, control)...")
    euclvq_metrics = evaluate_geovq(vae, loader, device, euc_centroids)

    results = {"VAE": vae_metrics, "VQ-VAE": vqvae_metrics,
              "Geo-VQ": geovq_metrics, "Eucl-VQ": euclvq_metrics}

    print("\n=== Comparación justa (términos por separado, no sumados) ===")
    print(f"{'Modelo':<10}{'BCE/img':>12}{'SSIM':>10}{'PSNR':>10}{'Otro término':>20}")
    for name, m in results.items():
        extra = ""
        if "kl_per_img" in m:
            extra = f"KL={m['kl_per_img']:.2f}"
        elif "commit_per_img" in m:
            extra = f"Commit={m['commit_per_img']:.4f} | Perp={m['perplexity']:.1f}/{args.embeddings}"
        print(f"{name:<10}{m['bce_per_img']:>12.2f}{m['ssim']:>10.4f}{m['psnr']:>10.2f}{extra:>20}")

    with open(LOGS_DIR / "results_table.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nGenerando figura comparativa...")
    make_comparison_figure(vae, vqvae, geo_centroids, euc_centroids, dataset, device, n=args.n_examples)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dim", type=int, default=20)
    p.add_argument("--embedding_dim", type=int, default=16)
    p.add_argument("--embeddings", type=int, default=64)
    p.add_argument("--n_examples", type=int, default=8)
    main(p.parse_args())
