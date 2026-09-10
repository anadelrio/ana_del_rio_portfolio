"""Extrae los latentes del VAE entrenado y ejecuta la cuantización geodésica
"a posteriori" — esta vez de verdad (ver src/geodesic/graph_tools.py).

Además, como control, ejecuta también K-means euclídeo estándar sobre los
MISMOS latentes (misma K, mismos datos) y guarda ambos resultados. Esto
permite comparar de forma justa si la distancia geodésica aporta algo
frente al euclídeo simple sobre este dataset — que es la pregunta que el
proyecto original planteaba pero nunca llegó a responder con código real.

Calidad del clustering: al tener las etiquetas reales de MNIST disponibles,
medimos además la pureza y el NMI de cada clustering frente al dígito real
— una métrica objetiva que el proyecto original no tenía.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import normalized_mutual_info_score

from data.dataset import MNISTDataset
from models.vae import VAE
from geodesic.graph_tools import geodesic_kmeans, euclidean_kmeans
from utils.paths import CHECKPOINTS_DIR, LATENTS_DIR


def cluster_purity(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Fracción de puntos cuyo cluster mayoritario coincide con su clase real."""
    correct = 0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        if mask.sum() == 0:
            continue
        majority = np.bincount(labels_true[mask]).argmax()
        correct += (labels_true[mask] == majority).sum()
    return correct / len(labels_true)


def main(args):
    device = torch.device("cpu")  # extracción de latentes es barata, no hace falta GPU

    print("Cargando VAE entrenado...")
    vae = VAE(latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(CHECKPOINTS_DIR / "vae_mnist.pt", map_location=device))
    vae.eval()

    print("Extrayendo latentes de TODO el dataset de entrenamiento...")
    dataset = MNISTDataset(train=True)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    latents, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            mu, _ = vae.encode(x.to(device))
            latents.append(mu.cpu().numpy())
            labels.append(y.numpy())
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Latentes: {latents.shape} (dataset completo, no una submuestra de 1.000)")

    np.save(LATENTS_DIR / "vae_latents.npy", latents)
    np.save(LATENTS_DIR / "vae_labels.npy", labels)

    # ---------- Clustering geodésico (real) ----------
    geo_labels, geo_centroids, _ = geodesic_kmeans(
        latents, n_clusters=args.n_clusters, n_neighbors=args.n_neighbors,
        n_landmarks=args.n_landmarks, random_state=args.seed,
    )
    np.save(LATENTS_DIR / "geodesic_labels.npy", geo_labels)
    np.save(LATENTS_DIR / "geodesic_centroids.npy", geo_centroids)

    # ---------- Clustering euclídeo (control / ablation) ----------
    print("🔹 Ejecutando K-means euclídeo (control) sobre los mismos latentes...")
    euc_labels, euc_centroids = euclidean_kmeans(latents, n_clusters=args.n_clusters, random_state=args.seed)
    np.save(LATENTS_DIR / "euclidean_labels.npy", euc_labels)
    np.save(LATENTS_DIR / "euclidean_centroids.npy", euc_centroids)

    # ---------- Comparación cuantitativa ----------
    geo_purity = cluster_purity(geo_labels, labels)
    euc_purity = cluster_purity(euc_labels, labels)
    geo_nmi = normalized_mutual_info_score(labels, geo_labels)
    euc_nmi = normalized_mutual_info_score(labels, euc_labels)

    print("\n=== Ablation: geodésico vs. euclídeo (mismos latentes, mismo K) ===")
    print(f"{'Método':<12}{'Pureza':>10}{'NMI':>10}")
    print(f"{'Geodésico':<12}{geo_purity:>10.4f}{geo_nmi:>10.4f}")
    print(f"{'Euclídeo':<12}{euc_purity:>10.4f}{euc_nmi:>10.4f}")

    with open(LATENTS_DIR / "clustering_ablation.csv", "w") as f:
        f.write("method,purity,nmi\n")
        f.write(f"geodesic,{geo_purity},{geo_nmi}\n")
        f.write(f"euclidean,{euc_purity},{euc_nmi}\n")
    print(f"\nGuardado en {LATENTS_DIR / 'clustering_ablation.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latent_dim", type=int, default=20)
    p.add_argument("--n_clusters", type=int, default=64)
    p.add_argument("--n_neighbors", type=int, default=10)
    p.add_argument("--n_landmarks", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
