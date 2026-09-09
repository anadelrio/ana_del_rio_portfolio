import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import LogLocator, FuncFormatter
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# definición de t-normas -----------------------------------------------

def tnorm_frank(x, y, s, eps=1e-12):
    x = np.asarray(x)
    y = np.asarray(y)

    if abs(s - 1.0) < 1e-12:
        return x * y
    if s <= 0:
        return np.minimum(x, y)

    ls     = np.log(s)
    sx     = np.exp(x * ls)
    sy     = np.exp(y * ls)
    inside = np.maximum(1.0 + ((sx - 1.0) * (sy - 1.0)) / (s - 1.0), eps)
    return np.log(inside) / ls

def tnorm_yager(x, y, p, eps=1e-15):
    x = np.asarray(x)
    y = np.asarray(y)

    x = np.clip(x, 0.0, 1.0 - eps)
    y = np.clip(y, 0.0, 1.0 - eps)

    if p > 1e9:
        return np.minimum(x, y)

    ax   = np.exp(p * np.log1p(-x))
    ay   = np.exp(p * np.log1p(-y))
    s    = ax + ay
    spow = np.zeros_like(s)
    mask = s > 0
    spow[mask] = np.exp((1.0 / p) * np.log(s[mask]))
    return np.maximum(1.0 - spow, 0.0)

# test de Mann-Whitney ---------------------------

def pvalor_mannwhitney_aprox(a, b):
    a  = np.asarray(a)
    b  = np.asarray(b)
    n1 = a.size
    n2 = b.size

    combinado = np.concatenate([a, b])
    orden     = np.argsort(combinado, kind="mergesort")
    rangos    = np.empty_like(orden, dtype=float)
    rangos[orden] = np.arange(1, n1 + n2 + 1, dtype=float)

    R1    = rangos[:n1].sum()
    U1    = R1 - n1 * (n1 + 1) / 2.0
    mu    = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return 1.0
    return float(2.0 * (1.0 - norm.cdf(abs((U1 - mu) / sigma))))

# agregación n-aria ----------------------------------------------------

def agregacion_naria(tnorm_func, X, param):
    res = X[:, 0]
    for k in range(1, X.shape[1]):
        res = tnorm_func(res, X[:, k], param)
    return res

def experimento_un_punto(p_yager, s_frank, tam_muestra, n_args, rng):
    X_y = rng.random((tam_muestra, n_args))
    X_f = rng.random((tam_muestra, n_args))
    return pvalor_mannwhitney_aprox(
        agregacion_naria(tnorm_yager, X_y, p_yager),
        agregacion_naria(tnorm_frank, X_f, s_frank),
    )

# barrido de la malla --------------------------------------------------

def barrido_malla(n_malla, n_experimentos, tam_muestra, n_args, semilla):
    rng    = np.random.default_rng(semilla)
    s_vals = np.logspace(-4, 2, n_malla)
    p_vals = np.logspace( 0, 5, n_malla)
    pmedio = np.zeros((n_malla, n_malla), dtype=float)

    for i, p in enumerate(tqdm(p_vals, desc=f"Yager p (n={n_args})")):
        for j, s in enumerate(s_vals):
            acc = 0.0
            for _ in range(n_experimentos):
                acc += experimento_un_punto(p, s, tam_muestra, n_args, rng)
            pmedio[i, j] = acc / n_experimentos

    return s_vals, p_vals, pmedio

# formateador de ticks -------------------------------------------------

def sci_1e_formatter(val, pos=None):
    if val <= 0:
        return ""
    exp = int(np.round(np.log10(val)))
    if not np.isclose(val, 10 ** exp):
        return ""
    sign = "+" if exp >= 0 else "-"
    return f"1e{sign}{abs(exp):02d}"

# figura resumen -------------------------------------------------------

def plot_figura(carpeta_salida, s_vals, p_vals, grids, n_args_list):
    os.makedirs(carpeta_salida, exist_ok=True)
    niveles = [0.1, 0.3, 0.5, 0.7, 0.9]
    S, P    = np.meshgrid(s_vals, p_vals)

    xticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    yticks = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, grid, n_args in zip(axes, grids, n_args_list):
        ax.set_facecolor("white")
        Z  = np.nan_to_num(np.asarray(grid, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        cs = ax.contour(S, P, Z, levels=niveles, colors="k", linewidths=1.0)
        ax.clabel(cs, fmt="%.1f", inline=True, fontsize=8, colors="k")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-4, 1e2); ax.set_ylim(1e0, 1e5)
        ax.set_xticks(xticks); ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(FuncFormatter(sci_1e_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(sci_1e_formatter))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.set_title(f"{n_args} argumentos", fontsize=11)
        ax.label_outer()

    fig.text(0.5,  0.04, "Frank",  ha="center", va="center", fontsize=12)
    fig.text(0.06, 0.5,  "Yager",  ha="center", va="center", rotation="vertical", fontsize=12)
    fig.tight_layout(rect=[0.08, 0.06, 1, 1])

    outpath = os.path.join(carpeta_salida, "fig_yager_vs_frank_aridad.png")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    return outpath

# main -----------------------------------------------------------------

def main(
    carpeta_salida,
    n_malla=100,
    n_experimentos=1000,
    tam_muestra=100,
    semilla=1234,
    n_args_list=(3, 5, 7, 10),
    guardar_npz=True,
):
    t0     = time.time()
    grids  = []
    s_ref  = None
    p_ref  = None

    os.makedirs(carpeta_salida, exist_ok=True)

    for n_args in n_args_list:
        s_vals, p_vals, pmedio = barrido_malla(
            n_malla, n_experimentos, tam_muestra, n_args, semilla)
        grids.append(pmedio)

        if s_ref is None:
            s_ref, p_ref = s_vals, p_vals

        if guardar_npz:
            np.savez(
                os.path.join(carpeta_salida, f"datos_yager_frank_n{n_args}.npz"),
                s_vals=s_vals, p_vals=p_vals, pmedio=pmedio,
                n_args=n_args, n_malla=n_malla,
                n_experimentos=n_experimentos, tam_muestra=tam_muestra,
            )

    plot_figura(carpeta_salida, s_ref, p_ref, grids, n_args_list)

    print(f"Resultados guardados en: {carpeta_salida}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    carpeta_script = os.path.dirname(os.path.abspath(__file__))
    main(
        carpeta_salida=os.path.join(carpeta_script, "salida_yager_vs_frank_aridad"),
        n_malla=100,
        n_experimentos=1000,
        tam_muestra=100,
        semilla=1234,
        n_args_list=(3, 5, 7, 10),
        guardar_npz=True,
    )