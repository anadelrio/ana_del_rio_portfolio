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

# definiciones t-normas usadas --------------------------------------------------------------------------------------
def tnorm_yager(x, y, p):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if p > 1e6:
        return np.minimum(x, y)
    ax   = np.exp(p * np.log1p(-np.clip(x, 0.0, 1.0 - 1e-15)))
    ay   = np.exp(p * np.log1p(-np.clip(y, 0.0, 1.0 - 1e-15)))
    s    = ax + ay
    spow = np.zeros_like(s)
    mask = s > 0
    spow[mask] = np.exp((1.0 / p) * np.log(s[mask]))
    return np.maximum(1.0 - spow, 0.0)

def tnorm_schweizer_sklar(x, y, q):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isclose(q, 0.0):
        return x * y
    eps = 1e-15
    x = np.clip(x, eps, 1.0)
    y = np.clip(y, eps, 1.0)
    if q > 0:
        inner  = np.maximum(0.0, x**q + y**q - 1.0)
        result = np.zeros_like(inner)
        mask   = inner > 0
        result[mask] = inner[mask] ** (1.0 / q)
        return result
    inner = x ** q + y ** q - 1.0
    inner = np.maximum(inner, 1e-300)
    return inner ** (1.0 / q)

# test estadístico --------------------------------------------------------------------------------------------------------------
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

# realizacion del experimento y barrido de la malla -------------------------------------------------------------
def experimento_un_punto(p_yager, q_ss, tam_muestra, rng):
    U_y = rng.random(tam_muestra); V_y = rng.random(tam_muestra)
    U_s = rng.random(tam_muestra); V_s = rng.random(tam_muestra)
    vals_yager = tnorm_yager(U_y, V_y, p_yager)
    vals_ss    = tnorm_schweizer_sklar(U_s, V_s, q_ss)
    return pvalor_mannwhitney_asymptotic(vals_yager, vals_ss)

def barrido_malla(n_malla, n_experimentos, tam_muestra, semilla):
    rng = np.random.default_rng(semilla)

    p_vals = np.logspace(-1, 4, n_malla)          
    q_pos  =  np.logspace(-3, 5, n_malla)          
    q_neg  = -np.logspace(-3, 5, n_malla)[::-1]    

    def _barrer(q_vals, desc):
        pm  = np.zeros((n_malla, n_malla))
        n50 = np.zeros((n_malla, n_malla))
        n05 = np.zeros((n_malla, n_malla))
        for i in tqdm(range(n_malla), desc=desc):
            p_y = p_vals[i]
            for j in range(n_malla):
                pv = np.array([
                    experimento_un_punto(p_y, q_vals[j], tam_muestra, rng)
                    for _ in range(n_experimentos)
                ])
                pm[i, j]  = np.mean(pv)
                n50[i, j] = np.mean(pv >= 0.50)
                n05[i, j] = np.mean(pv >= 0.05)
        return pm, n50, n05

    print("Barriendo mitad negativa (q<0)...")
    pm_neg, n50_neg, n05_neg = _barrer(q_neg, "  q<0 filas Yager")
    print("Barriendo mitad positiva (q>0)...")
    pm_pos, n50_pos, n05_pos = _barrer(q_pos, "  q>0 filas Yager")

    return (p_vals, q_neg, q_pos,
            pm_neg, n50_neg, n05_neg,
            pm_pos, n50_pos, n05_pos)

# creación de la imagen resumen ---------------------------------------------------------------
def figura_resumen(carpeta_salida, p_vals, q_neg, q_pos,
                   pm_neg, n50_neg, n05_neg,
                   pm_pos, n50_pos, n05_pos):

    niveles  = [0.1, 0.2, 0.3, 0.4, 0.5]
    pm_neg_s = gaussian_filter(pm_neg, sigma=1.0)
    pm_pos_s = gaussian_filter(pm_pos, sigma=1.0)

    S_neg, P_neg = np.meshgrid(np.abs(q_neg), p_vals)
    S_pos, P_pos = np.meshgrid(q_pos,          p_vals)

    def fmt_neg(x, pos=None):
        if x <= 0: return ""
        return f"-1e{int(round(np.log10(x))):+d}"

    def fmt_pos(x, pos=None):
        if x <= 0: return ""
        return f"1e{int(round(np.log10(x))):+d}"

    def fmt_y(x, pos=None):
        if x <= 0: return ""
        return f"1e{int(round(np.log10(x))):+d}"

    def estilo_neg(ax, show_ylabel=True):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(np.abs(q_neg[0]), np.abs(q_neg[-1]))  
        ax.set_ylim(1e-1, 1e4)
        ax.set_xticks([1e5, 1e3, 1e1, 1e-1, 1e-3])
        ax.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_neg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=7, length=3)
        ax.set_xlabel("Schweizer-Sklar", fontsize=8, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=8, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    def estilo_pos(ax, show_ylabel=False):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(q_pos[0], q_pos[-1])
        ax.set_ylim(1e-1, 1e4)
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
        ax.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_pos))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=7, length=3)
        ax.set_xlabel("Schweizer-Sklar", fontsize=8, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=8, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig = plt.figure(figsize=(10.0, 12.0), facecolor="white")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.15)

    ax_c_neg = fig.add_subplot(gs[0, 0])
    ax_c_pos = fig.add_subplot(gs[0, 1])
    ax_a_neg = fig.add_subplot(gs[1, 0])
    ax_a_pos = fig.add_subplot(gs[1, 1])
    ax_n_neg = fig.add_subplot(gs[2, 0])
    ax_n_pos = fig.add_subplot(gs[2, 1])

    c1 = ax_c_neg.contour(S_neg, P_neg, pm_neg_s, levels=niveles, colors="k", linewidths=0.85)
    ax_c_neg.clabel(c1, fmt="%.1f", inline=True, fontsize=7)
    estilo_neg(ax_c_neg, show_ylabel=True)

    c2 = ax_c_pos.contour(S_pos, P_pos, pm_pos_s, levels=niveles, colors="k", linewidths=0.85)
    ax_c_pos.clabel(c2, fmt="%.1f", inline=True, fontsize=7)
    estilo_pos(ax_c_pos, show_ylabel=False)

    ax_a_neg.pcolormesh(S_neg, P_neg, n50_neg, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_neg(ax_a_neg, show_ylabel=True)
    ax_a_pos.pcolormesh(S_pos, P_pos, n50_pos, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_pos(ax_a_pos, show_ylabel=False)

    ax_n_neg.pcolormesh(S_neg, P_neg, n05_neg, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_neg(ax_n_neg, show_ylabel=True)
    ax_n_pos.pcolormesh(S_pos, P_pos, n05_pos, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_pos(ax_n_pos, show_ylabel=False)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.06, top=0.97)
    ruta = os.path.join(carpeta_salida, "fig_resumen_yager_vs_schweizer_sklar.png")
    fig.savefig(ruta, dpi=450)
    plt.close(fig)
    print(f"Figura guardada en: {ruta}")

# guardado de resultados y main ---------------------------------------------------------------------------------------
def guardar_resultados(carpeta_salida, p_vals, q_neg, q_pos,
                       pm_neg, n50_neg, n05_neg,
                       pm_pos, n50_pos, n05_pos):
    os.makedirs(carpeta_salida, exist_ok=True)
    np.savez_compressed(
        os.path.join(carpeta_salida, "datos_yager_vs_schweizer_sklar.npz"),
        p_vals=p_vals, q_neg=q_neg, q_pos=q_pos,
        pm_neg=pm_neg, n50_neg=n50_neg, n05_neg=n05_neg,
        pm_pos=pm_pos, n50_pos=n50_pos, n05_pos=n05_pos
    )

def main(carpeta_salida, n_malla=100, n_experimentos=1000, tam_muestra=100, semilla=1234):
    t0 = time.time()
    os.makedirs(carpeta_salida, exist_ok=True)
    resultados = barrido_malla(n_malla, n_experimentos, tam_muestra, semilla)
    guardar_resultados(carpeta_salida, *resultados)
    figura_resumen(carpeta_salida, *resultados)
    print(f"Proceso finalizado en: {carpeta_salida}")
    print(f"Tiempo total: {(time.time() - t0) / 60:.2f} minutos")

if __name__ == "__main__":
    carpeta_script = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    main(
        carpeta_salida=os.path.join(carpeta_script, "salida_yager_vs_schweizer_sklar"),
        n_malla=100,
        n_experimentos=1000,
        tam_muestra=100,
        semilla=1234
    )