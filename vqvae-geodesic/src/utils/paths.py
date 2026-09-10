"""Rutas centralizadas del proyecto.

Todas se calculan a partir de la ubicación de este archivo, no del directorio
desde el que se ejecute un script — así todo funciona igual desde una
terminal, desde VS Code, o importado como módulo.
"""
from pathlib import Path

# src/utils/paths.py -> sube dos niveles para llegar a la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
LATENTS_DIR = EXPERIMENTS_DIR / "latents"

for d in (DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, FIGURES_DIR, LATENTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
