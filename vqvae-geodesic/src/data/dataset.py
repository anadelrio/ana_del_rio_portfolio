"""Dataset de MNIST a partir de los archivos idx-ubyte originales.

Se implementa un lector manual en vez de depender de `torchvision.datasets`
para reducir dependencias — el formato idx es muy simple y los archivos
crudos ya vienen incluidos en `data/MNIST/raw/`.
"""
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.paths import DATA_DIR

MNIST_RAW_DIR = DATA_DIR / "MNIST" / "raw"

_FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte",
}

_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def _download_if_missing():
    MNIST_RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in _FILES.values() if not (MNIST_RAW_DIR / f).exists()]
    if not missing:
        return
    for fname in missing:
        gz_name = fname + ".gz"
        ok = False
        for mirror in _MIRRORS:
            try:
                url = mirror + gz_name
                gz_path = MNIST_RAW_DIR / gz_name
                urllib.request.urlretrieve(url, gz_path)
                with gzip.open(gz_path, "rb") as f_in, open(MNIST_RAW_DIR / fname, "wb") as f_out:
                    f_out.write(f_in.read())
                gz_path.unlink()
                ok = True
                break
            except Exception:
                continue
        if not ok:
            raise RuntimeError(
                f"No se pudo descargar {fname}. Coloca manualmente los 4 archivos "
                f"idx-ubyte de MNIST en {MNIST_RAW_DIR}"
            )


def _read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number inesperado en {path}: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number inesperado en {path}: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


class MNISTDataset(Dataset):
    """MNIST en memoria, normalizado a [0, 1], shape (1, 28, 28)."""

    def __init__(self, train: bool = True, download: bool = True):
        if download:
            _download_if_missing()
        prefix = "train" if train else "test"
        images = _read_idx_images(MNIST_RAW_DIR / _FILES[f"{prefix}_images"])
        labels = _read_idx_labels(MNIST_RAW_DIR / _FILES[f"{prefix}_labels"])
        self.images = torch.from_numpy(images.copy()).float().div_(255.0).unsqueeze(1)  # (N,1,28,28)
        self.labels = torch.from_numpy(labels.copy()).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
