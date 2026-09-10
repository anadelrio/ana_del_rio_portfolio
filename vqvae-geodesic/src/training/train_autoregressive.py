"""Entrena el prior autoregresivo sobre las rejillas de códigos del VQ-VAE
(no sobre secuencias artificiales de imágenes distintas — ver
models/autoregressive.py para la explicación del cambio)."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import MNISTDataset
from models.vqvae import VQVAE
from models.autoregressive import CodeGridAR
from utils.paths import CHECKPOINTS_DIR, LOGS_DIR


def extract_code_grids(vqvae, dataset, device, batch_size=512):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_codes = []
    vqvae.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, _, indices, _ = vqvae(x)          # (B, 7, 7)
            all_codes.append(indices.reshape(indices.size(0), -1).cpu())  # (B, 49) raster order
    return torch.cat(all_codes, dim=0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dispositivo: {device}")

    vqvae = VQVAE(embedding_dim=args.embedding_dim, num_embeddings=args.embeddings).to(device)
    vqvae.load_state_dict(torch.load(CHECKPOINTS_DIR / "vqvae_mnist.pt", map_location=device))

    print("Extrayendo rejillas de códigos del VQ-VAE entrenado...")
    dataset = MNISTDataset(train=True)
    codes = extract_code_grids(vqvae, dataset, device)
    seq_len = codes.shape[1]
    print(f"Códigos extraídos: {codes.shape} (una secuencia de {seq_len} tokens por imagen)")

    code_dataset = TensorDataset(codes)
    loader = DataLoader(code_dataset, batch_size=args.batch_size, shuffle=True)

    model = CodeGridAR(vocab_size=args.embeddings, embed_dim=args.embed_dim,
                       hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = CHECKPOINTS_DIR / "autoregressive_model.pt"
    resume_path = CHECKPOINTS_DIR / "autoregressive_model_resume.pt"
    start_epoch = 1
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        print(f"Reanudando desde la época {start_epoch}")

    loss_fn = nn.CrossEntropyLoss()

    log_csv = LOGS_DIR / "autoregressive_train_log.csv"
    if not (args.resume and log_csv.exists()):
        log_csv.write_text("epoch,avg_loss,nll_per_token\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0.0
        for (batch_codes,) in loader:
            batch_codes = batch_codes.to(device)
            logits = model(batch_codes)  # (B, L, vocab)
            loss = loss_fn(logits.reshape(-1, args.embeddings), batch_codes.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_codes.size(0)

        avg_loss = total_loss / len(code_dataset)
        print(f"E{epoch:02d} | NLL/token {avg_loss:.4f}")
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{avg_loss * len(code_dataset)},{avg_loss}\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, resume_path)
        torch.save({"state_dict": model.state_dict(), "seq_len": seq_len,
                   "vocab_size": args.embeddings}, ckpt_path)

    print("Guardado en", ckpt_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embedding_dim", type=int, default=16)
    p.add_argument("--embeddings", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())
