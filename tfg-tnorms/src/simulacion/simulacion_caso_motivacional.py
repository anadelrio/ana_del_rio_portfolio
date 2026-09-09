import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# definición de t-normas -----------------------------------------------

def tnorm_product(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * y

def tnorm_yager(x, y, p):
    if p <= 0:
        raise ValueError("Se requiere p > 0.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inner = (1.0 - x) ** p + (1.0 - y) ** p
    return np.maximum(0.0, 1.0 - inner ** (1.0 / p))

# generación de muestra ------------------------------------------------

def generar_muestra(sample_size=100, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(sample_size, 2))
    return X[:, 0], X[:, 1]

# figura 1: histogramas ------------------------------------------------

def dibujar_histogramas(sample_size=100, seed=123):
    x1, x2 = generar_muestra(sample_size=sample_size, seed=seed)

    y_prod     = tnorm_product(x1, x2)
    y_yager_25 = tnorm_yager(x1, x2, p=2.5)
    y_yager_1  = tnorm_yager(x1, x2, p=1.0)
    y_yager_5  = tnorm_yager(x1, x2, p=5.0)

    datos = [
        (y_prod,     "Product",        "#404040"),
        (y_yager_25, r"Yager-$p$=2.5", "#909090"),
        (y_yager_1,  r"Yager-$p$=1",   "#909090"),
        (y_yager_5,  r"Yager-$p$=5",   "#909090"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))  

    for ax, (vals, titulo, color) in zip(axes.flat, datos):
        max_val = vals.max()
        bins = np.linspace(0.0, max_val, 9)  
        ax.hist(vals, bins=bins, color=color, edgecolor="white")
        ax.set_title(titulo, fontsize=11)
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, max_val * 1.05)         
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "motivacional_histogramas.jpg")
    plt.savefig(out, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()
    return out

# figura 2: QQ-plot ----------------------------------------------------

def dibujar_qqplot(sample_size=100, seed=123):
    x1, x2 = generar_muestra(sample_size=sample_size, seed=seed)

    sorted_prod    = np.sort(tnorm_product(x1, x2))
    sorted_yager25 = np.sort(tnorm_yager(x1, x2, p=2.5))

    plt.figure(figsize=(5, 5))
    plt.scatter(sorted_prod, sorted_yager25, s=20, facecolors="none", edgecolors="black")
    plt.plot([0, 1], [0, 1], "k-")            
    plt.xlabel("Product")
    plt.ylabel("Yager t-norm (p=2.5)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "motivacional_qqplot.jpg")
    plt.savefig(out, format="jpg", dpi=150, bbox_inches="tight")
    plt.close()
    return out

# simulación Mann-Whitney U --------------------------------------------

def simular_product_vs_yager(p, n_iter=10000, sample_size=100, seed=42):
    rng = np.random.default_rng(seed)
    p_values = np.empty(n_iter, dtype=float)

    for i in range(n_iter):
        X_prod  = rng.uniform(0.0, 1.0, size=(sample_size, 2))
        X_yager = rng.uniform(0.0, 1.0, size=(sample_size, 2))
        y_prod  = tnorm_product(X_prod[:, 0],  X_prod[:, 1])
        y_yager = tnorm_yager(X_yager[:, 0], X_yager[:, 1], p=p)
        _, p_values[i] = mannwhitneyu(y_prod, y_yager, alternative="two-sided")

    reject_rate        = 100.0 * np.mean(p_values <= 0.05)
    accept_rate        = 100.0 * np.mean(p_values > 0.05)
    strong_accept_rate = 100.0 * np.mean(p_values > 0.5)
    return reject_rate, accept_rate, strong_accept_rate

# tabla de tasas de rechazo --------------------------------------------

def mostrar_tabla_motivacional():
    valores_p = [2.5, 1.0, 5.0]
    sep = "-" * 72

    print("Comparacion Product vs. Yager")
    print("Entradas ~ Uniforme(0,1), n=100, 2 argumentos, 10 000 iteraciones")
    print(sep)

    for p in tqdm(valores_p, desc="Simulando"):
        reject_rate, accept_rate, strong_accept_rate = simular_product_vs_yager(p=p)
        print(f"Yager (p = {p})")
        print(f"  Rechazo    p-valor < 0.05 : {reject_rate:.1f}%")
        print(f"  Aceptacion p-valor >= 0.05: {accept_rate:.1f}%")
        print(f"  Aceptacion p-valor >= 0.5 : {strong_accept_rate:.1f}%")
        print(sep)

# main -----------------------------------------------------------------

if __name__ == "__main__":
    tareas = [
        "Histogramas",
        "QQ-plot",
        "Tabla motivacional",
    ]

    with tqdm(total=len(tareas), desc="Generando figuras") as pbar:
        dibujar_histogramas()
        pbar.update(1)

        dibujar_qqplot()
        pbar.update(1)

    mostrar_tabla_motivacional()

    print(f"Figuras guardadas en: {BASE_DIR}")