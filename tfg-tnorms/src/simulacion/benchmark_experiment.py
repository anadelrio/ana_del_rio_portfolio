from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ruta de salida -------------------------------------------------------
try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_FILE = os.path.join(BASE_DIR, "benchmark_experiment_resultados.txt")

# configuracion del experimento ----------------------------------------
@dataclass(frozen=True)
class BenchmarkConfig:
    sample_size: int = 100
    n_args: int = 2
    n_iter: int = 10000
    alpha: float = 0.05
    base_seed: int = 123

@dataclass(frozen=True)
class BenchmarkStats:
    p: int
    p_prime: int
    rechazo: float
    no_rechazo: float
    no_rechazo_fuerte: float

# definición de t-norma de Yager ---------------------------------------
def t_yager_binaria(x: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if p == 0:
        resultado = np.zeros_like(x)
        x_es_uno = np.isclose(x, 1.0)
        y_es_uno = np.isclose(y, 1.0)
        resultado[x_es_uno & y_es_uno]  = 1.0
        resultado[x_es_uno & ~y_es_uno] = y[x_es_uno & ~y_es_uno]
        resultado[y_es_uno & ~x_es_uno] = x[y_es_uno & ~x_es_uno]
        return resultado

    if np.isinf(p):
        return np.minimum(x, y)

    radio = ((1.0 - x) ** p + (1.0 - y) ** p) ** (1.0 / p)
    return 1.0 - np.minimum(1.0, radio)

def yager_n_aria(matriz: np.ndarray, p: float) -> np.ndarray:
    matriz = np.asarray(matriz, dtype=float)
    agg = matriz[:, 0]
    for col in range(1, matriz.shape[1]):
        agg = t_yager_binaria(agg, matriz[:, col], p)
    return agg

# simulación de un par (p, p') -----------------------------------------
def simular_par_yager(
    p: int,
    p_prime: int,
    config: BenchmarkConfig,
    seed: int | None = None,
) -> BenchmarkStats:
    rng = np.random.default_rng(seed)
    p_values = np.empty(config.n_iter, dtype=float)

    for i in range(config.n_iter):
        muestra_p      = rng.random((config.sample_size, config.n_args))
        muestra_prime  = rng.random((config.sample_size, config.n_args))
        salida_p       = yager_n_aria(muestra_p,     p)
        salida_p_prime = yager_n_aria(muestra_prime, p_prime)
        _, p_values[i] = mannwhitneyu(salida_p, salida_p_prime, alternative="two-sided")

    return BenchmarkStats(
        p=p,
        p_prime=p_prime,
        rechazo=float(np.mean(p_values <= config.alpha) * 100.0),
        no_rechazo=float(np.mean(p_values > config.alpha) * 100.0),
        no_rechazo_fuerte=float(np.mean(p_values > 0.5) * 100.0),
    )

# promedio de un bloque de resultados ----------------------------------
def promedio_resultados(resultados: Iterable[BenchmarkStats]) -> BenchmarkStats:
    resultados = list(resultados)
    return BenchmarkStats(
        p=-1,
        p_prime=-1,
        rechazo=float(np.mean([r.rechazo for r in resultados])),
        no_rechazo=float(np.mean([r.no_rechazo for r in resultados])),
        no_rechazo_fuerte=float(np.mean([r.no_rechazo_fuerte for r in resultados])),
    )

# formato LaTeX --------------------------------------------------------
def fila_latex_comparacion(label: str, stats: BenchmarkStats) -> str:
    return (
        f" {label} & "
        f"{stats.rechazo:.2f}\\% & "
        f"{stats.no_rechazo:.2f}\\% & "
        f"{stats.no_rechazo_fuerte:.2f}\\% \\\\"
    )

def fila_latex_yager(stats: BenchmarkStats) -> str:
    etiqueta = f"Yager ($p$ = {stats.p}) vs Yager ($p'$ = {stats.p_prime})"
    return fila_latex_comparacion(etiqueta, stats)

# ejecución del experimento --------------------------------------------
def ejecutar_benchmark_principal(config: BenchmarkConfig | None = None) -> List[str]:
    if config is None:
        config = BenchmarkConfig()

    filas_latex: List[str] = []
    ps_explicitos = [0, 1, 2, 3, 4]
    resultados_p_mas_1 = {}

    total = len(ps_explicitos) + 26 + (300 - 26)
    with tqdm(total=total, desc="Simulando benchmark") as pbar:

        for indice, p in enumerate(ps_explicitos):
            stats = simular_par_yager(p, p + 1, config, seed=config.base_seed + indice)
            filas_latex.append(fila_latex_yager(stats))
            resultados_p_mas_1[p] = stats
            pbar.update(1)

        bloque_p_mas_1 = []
        for p in range(0, 26):
            if p in resultados_p_mas_1:
                stats = resultados_p_mas_1[p]
            else:
                stats = simular_par_yager(p, p + 1, config, seed=config.base_seed + 1000 + p)
            bloque_p_mas_1.append(stats)
            pbar.update(1)

        bloque_p_mas_10 = []
        for p in range(26, 300):
            stats = simular_par_yager(p, p + 10, config, seed=config.base_seed + 2000 + p)
            bloque_p_mas_10.append(stats)
            pbar.update(1)

    promedio_p_mas_1  = promedio_resultados(bloque_p_mas_1)
    promedio_p_mas_10 = promedio_resultados(bloque_p_mas_10)

    filas_latex.append(fila_latex_comparacion("Yager ($p$ = p) vs Yager ($p'$ = p+1)",  promedio_p_mas_1))
    filas_latex.append(fila_latex_comparacion("Yager ($p$ = p) vs Yager ($p'$ = p+10)", promedio_p_mas_10))

    return filas_latex

# main ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    filas = ejecutar_benchmark_principal()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("% Filas para la tabla resumen\n")
        for fila in filas:
            f.write(fila + "\n")

    print(f"Resultados guardados en: {OUTPUT_FILE}")