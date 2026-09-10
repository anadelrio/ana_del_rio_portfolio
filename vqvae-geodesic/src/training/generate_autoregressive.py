"""Genera dígitos nuevos: el prior autoregresivo genera una rejilla de
códigos desde cero, y el decoder del VQ-VAE la convierte en una imagen."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt

from models.vqvae import VQVAE
from models.autoregressive import CodeGridAR
from utils.paths import CHECKPOINTS_DIR, FIGURES_DIR


def main(args):
    device = torch.device("cpu")

    vqvae = VQVAE(embedding_dim=args.embedding_dim, num_embeddings=args.embeddings).to(device)
    vqvae.load_state_dict(torch.load(CHECKPOINTS_DIR / "vqvae_mnist.pt", map_location=device))
    vqvae.eval()

    ckpt = torch.load(CHECKPOINTS_DIR / "autoregressive_model.pt", map_location=device)
    ar = CodeGridAR(vocab_size=ckpt["vocab_size"]).to(device)
    ar.load_state_dict(ckpt["state_dict"])
    seq_len = ckpt["seq_len"]
    grid_side = int(seq_len ** 0.5)
    assert grid_side * grid_side == seq_len, "seq_len debe ser un cuadrado perfecto (rejilla NxN)"

    print(f"Generando {args.n} muestras nuevas (rejilla {grid_side}x{grid_side} códigos)...")
    codes = ar.generate(batch_size=args.n, seq_len=seq_len, device=device, temperature=args.temperature)
    codes_grid = codes.view(args.n, grid_side, grid_side)

    with torch.no_grad():
        z_q = vqvae.quantizer.embedding[codes_grid]        # (n, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2)                       # (n, D, H, W)
        images = torch.sigmoid(vqvae.decoder(z_q))

    n_cols = min(args.n, 8)
    n_rows = (args.n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    axes = axes.reshape(n_rows, n_cols) if n_rows > 1 else axes.reshape(1, n_cols)
    for i in range(args.n):
        r, c = divmod(i, n_cols)
        axes[r, c].imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for i in range(args.n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / "ar_generated.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print("Guardado en", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--embedding_dim", type=int, default=16)
    p.add_argument("--embeddings", type=int, default=64)
    main(p.parse_args())
