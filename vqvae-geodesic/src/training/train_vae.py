import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from data.dataset import MNISTDataset
from models.vae import VAE, vae_loss
from utils.paths import CHECKPOINTS_DIR, LOGS_DIR


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Dispositivo: {device}")

    model = VAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = CHECKPOINTS_DIR / "vae_mnist.pt"          # state_dict "limpio" (lo usan otros scripts)
    resume_path = CHECKPOINTS_DIR / "vae_mnist_resume.pt"  # checkpoint con optimizador, solo para reanudar
    start_epoch = 1
    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        print(f"Reanudando desde la época {start_epoch}")

    dataset = MNISTDataset(train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    log_csv = LOGS_DIR / "vae_train_log.csv"
    if not (args.resume and log_csv.exists()):
        log_csv.write_text("epoch,total_loss,bce,kl\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        tot = bce_tot = kl_tot = 0.0
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, bce, kl = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            tot += loss.item()
            bce_tot += bce.item()
            kl_tot += kl.item()

        n = len(dataset)
        print(f"E{epoch:02d} | Loss/img {tot/n:.2f} | BCE/img {bce_tot/n:.2f} | KL/img {kl_tot/n:.2f}")
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{tot},{bce_tot},{kl_tot}\n")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, resume_path)
        torch.save(model.state_dict(), ckpt_path)

    print("Guardado en", ckpt_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true", help="Continuar desde el último checkpoint guardado")
    train(p.parse_args())
