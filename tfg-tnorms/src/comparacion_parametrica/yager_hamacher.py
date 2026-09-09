import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from tqdm import tqdm

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# definiciones de t-normas -------------------------------------------------------------------------------

def tnorm_yager(x, y, p):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p > 1e6:
        return np.minimum(x, y)
    ax   = np.exp(p * np.log1p(-x))
    ay   = np.exp(p * np.log1p(-y))
    s    = ax + ay
    spow = np.zeros_like(s)
    mask = s > 0
    spow[mask] = np.exp((1.0 / p) * np.log(s[mask]))
    return np.maximum(1.0 - spow, 0.0)

def tnorm_hamacher(x1, x2, r):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if np.isclose(r, 0.0, atol=1e-10):
        return x1 * x2
    den = r + (1.0 - r) * (x1 + x2 - x1 * x2)
    res = np.zeros_like(den)
    mask = den > 1e-15
    res[mask] = (x1[mask] * x2[mask]) / den[mask]
    return res

# test estadístico ----------------------------------------------------------------------

def pvalor_mannwhitney_asymptotic(x, y):
    n1, n2 = len(x), len(y)
    all_data = np.concatenate([x, y])
    argsorted = np.argsort(all_data)
    ranks = np.empty_like(argsorted)
    ranks[argsorted] = np.arange(1, len(all_data) + 1)
    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    mu    = (n1 * n2) / 2.0
    sigma = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
    z     = (u1 - mu) / sigma
    return 2.0 * (1.0 - norm.cdf(abs(z)))

# realización del experimento y del barrido de la malla ---------------------------------------------------------
def experimento_un_punto(p_yager, r_hamacher, tam_muestra, rng):
    U_y = rng.random(tam_muestra); V_y = rng.random(tam_muestra)
    U_h = rng.random(tam_muestra); V_h = rng.random(tam_muestra)
    vals_yager    = tnorm_yager(U_y, V_y, p_yager)
    vals_hamacher = tnorm_hamacher(U_h, V_h, r_hamacher)
    return pvalor_mannwhitney_asymptotic(vals_yager, vals_hamacher)

def barrido_malla(n_malla, n_experimentos, tam_muestra, semilla):
    rng = np.random.default_rng(semilla)
    r_vals = np.logspace(-2, 2, n_malla)
    p_vals = np.logspace(-1, 2, n_malla)

    pmedio_grid   = np.zeros((n_malla, n_malla))
    norech50_grid = np.zeros((n_malla, n_malla))
    norech05_grid = np.zeros((n_malla, n_malla))

    for i in tqdm(range(n_malla), desc="Simulando filas de la malla"):
        p_yager = p_vals[i]
        for j in range(n_malla):
            r_hamacher = r_vals[j]
            pvals_iter = np.array([
                experimento_un_punto(p_yager, r_hamacher, tam_muestra, rng)
                for _ in range(n_experimentos)
            ])
            pmedio_grid[i, j]   = np.mean(pvals_iter)
            norech50_grid[i, j] = np.mean(pvals_iter >= 0.50)
            norech05_grid[i, j] = np.mean(pvals_iter >= 0.05)

    return r_vals, p_vals, pmedio_grid, norech50_grid, norech05_grid

# generación de la imagen --------------------------------------------------------------------------------------
def figura_resumen(carpeta_salida, r_vals, p_vals, pmedio_grid, norech50_grid, norech05_grid):
    niveles_contorno = [0.1, 0.2, 0.3, 0.4, 0.5]
    pmedio_suave = gaussian_filter(pmedio_grid, sigma=1.0)
    S, P = np.meshgrid(r_vals, p_vals)

    def fmt_exp(x, pos=None):
        if x <= 0: return ""
        return f"1e{int(round(np.log10(x))):+d}"

    def estilo(ax, show_ylabel=True):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(r_vals[0], r_vals[-1])
        ax.set_ylim(p_vals[0], p_vals[-1])
        ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
        ax.set_yticks([1e-1, 1e0, 1e1, 1e2])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_exp))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_exp))
        ax.tick_params(axis="both", which="major", labelsize=8.5, length=3.5)
        ax.set_xlabel("Hamacher", fontsize=9.5, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=9.5, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    fig = plt.figure(figsize=(6.2, 7.8), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig, height_ratios=[1.25, 1.0], hspace=0.22, wspace=0.20)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Contornos del p-valor medio
    cont = ax1.contour(S, P, pmedio_suave, levels=niveles_contorno, colors="k", linewidths=0.85)
    ax1.clabel(cont, fmt="%.1f", inline=True, fontsize=7.5)
    estilo(ax1)

    # Mapas de densidad 
    ax2.pcolormesh(S, P, norech50_grid, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo(ax2, show_ylabel=True)

    ax3.pcolormesh(S, P, norech05_grid, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo(ax3, show_ylabel=False)

    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.07, top=0.96)
    ruta = os.path.join(carpeta_salida, "fig_resumen_yager_vs_hamacher.png")
    fig.savefig(ruta, dpi=450)
    plt.close(fig)
    print(f"Figura guardada en: {ruta}")

# guardado de resultados y main -----------------------------------------------------------------------------------------------------------

def guardar_resultados(carpeta_salida, r_vals, p_vals, pmedio, norech50, norech05):
    os.makedirs(carpeta_salida, exist_ok=True)
    np.savez(
        os.path.join(carpeta_salida, "datos_yager_vs_hamacher.npz"),
        r_vals=r_vals, p_vals=p_vals,
        pmedio=pmedio, norech50=norech50, norech05=norech05
    )

def main(carpeta_salida, n_malla=100, n_experimentos=1000, tam_muestra=100, semilla=1234):
    t0 = time.time()
    os.makedirs(carpeta_salida, exist_ok=True)

    r_vals, p_vals, pmedio, norech50, norech05 = barrido_malla(
        n_malla, n_experimentos, tam_muestra, semilla
    )
    guardar_resultados(carpeta_salida, r_vals, p_vals, pmedio, norech50, norech05)
    figura_resumen(carpeta_salida, r_vals, p_vals, pmedio, norech50, norech05)

    print(f"Proceso finalizado en: {carpeta_salida}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.2f} minutos")

if __name__ == "__main__":
    carpeta_script = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    main(
        carpeta_salida=os.path.join(carpeta_script, "salida_yager_vs_hamacher"),
        n_malla=100,
        n_experimentos=1000,
        tam_muestra=100,
        semilla=1234
    )