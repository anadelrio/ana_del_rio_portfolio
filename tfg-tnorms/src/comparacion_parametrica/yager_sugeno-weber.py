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

def tnorm_sugeno_weber(x, y, t):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Caso límite t -> inf: producto
    if t > 1e6:
        return x * y
    return np.maximum(0.0, (x + y - 1.0 + t * x * y) / (1.0 + t))

# test estadístico -----------------------------------------------------------------------------
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

# realización del experimento y barrido de malla --------------------------------------------------------------
def experimento_un_punto(p_yager, t_sw, tam_muestra, rng):
    U_y = rng.random(tam_muestra); V_y = rng.random(tam_muestra)
    U_w = rng.random(tam_muestra); V_w = rng.random(tam_muestra)
    vals_yager = tnorm_yager(U_y, V_y, p_yager)
    vals_sw    = tnorm_sugeno_weber(U_w, V_w, t_sw)
    return pvalor_mannwhitney_asymptotic(vals_yager, vals_sw)

def barrido_malla(n_malla, n_experimentos, tam_muestra, semilla):
    rng = np.random.default_rng(semilla)
    p_vals = np.logspace(-1, 1, n_malla)
    t_neg = -np.logspace(-3, 0, n_malla)[::-1]  
    t_pos = np.logspace(-3, 3, n_malla)

    def _barrido_mitad(t_vals):
        pm   = np.zeros((n_malla, n_malla))
        n50  = np.zeros((n_malla, n_malla))
        n05  = np.zeros((n_malla, n_malla))
        for i in tqdm(range(n_malla), desc="  Filas Yager"):
            p_y = p_vals[i]
            for j in range(n_malla):
                t_w = t_vals[j]
                pv  = np.array([
                    experimento_un_punto(p_y, t_w, tam_muestra, rng)
                    for _ in range(n_experimentos)
                ])
                pm[i, j]  = np.mean(pv)
                n50[i, j] = np.mean(pv >= 0.50)
                n05[i, j] = np.mean(pv >= 0.05)
        return pm, n50, n05

    print("Barriendo mitad negativa...")
    pm_neg, n50_neg, n05_neg = _barrido_mitad(t_neg)
    print("Barriendo mitad positiva...")
    pm_pos, n50_pos, n05_pos = _barrido_mitad(t_pos)

    return (p_vals, t_neg, t_pos,
            pm_neg, n50_neg, n05_neg,
            pm_pos, n50_pos, n05_pos)

# generación de la imagen ---------------------------------------------------------------------------------
def figura_resumen(carpeta_salida, p_vals, t_neg, t_pos,
                   pm_neg, n50_neg, n05_neg,
                   pm_pos, n50_pos, n05_pos):

    niveles = [0.1, 0.2, 0.3, 0.4, 0.5]

    pm_neg_s = gaussian_filter(pm_neg, sigma=1.0)
    pm_pos_s = gaussian_filter(pm_pos, sigma=1.0)

    S_neg, P_neg = np.meshgrid(np.abs(t_neg), p_vals)
    S_pos, P_pos = np.meshgrid(t_pos,          p_vals)

    def fmt_neg(x, pos=None):
        if x <= 0: return ""
        e = int(round(np.log10(x)))
        return f"-1e{e:+d}" if e != 0 else "-1"

    def fmt_pos(x, pos=None):
        if x <= 0: return ""
        e = int(round(np.log10(x)))
        return f"1e{e:+d}"

    def fmt_y(x, pos=None):
        if x <= 0: return ""
        e = int(round(np.log10(x)))
        return f"1e{e:+d}"

    def estilo_neg(ax, show_ylabel=True):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(np.abs(t_neg[0]), np.abs(t_neg[-1]))
        ax.set_ylim(p_vals[0], p_vals[-1])
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
        ax.set_yticks([0.1, 0.5, 1, 2, 5, 10])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_neg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=8, length=3.5)
        ax.set_xlabel("Sugeno-Weber", fontsize=9, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=9, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    def estilo_pos(ax, show_ylabel=False):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(t_pos[0], t_pos[-1])
        ax.set_ylim(p_vals[0], p_vals[-1])
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
        ax.set_yticks([0.1, 0.5, 1, 2, 5, 10])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_pos))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=8, length=3.5)
        ax.set_xlabel("Sugeno-Weber", fontsize=9, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=9, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig = plt.figure(figsize=(10.0, 10.0), facecolor="white")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.15)

    ax_c_neg = fig.add_subplot(gs[0, 0])   # contornos negativo
    ax_c_pos = fig.add_subplot(gs[0, 1])   # contornos positivo
    ax_a_neg = fig.add_subplot(gs[1, 0])   # acceptance negativo
    ax_a_pos = fig.add_subplot(gs[1, 1])   # acceptance positivo
    ax_n_neg = fig.add_subplot(gs[2, 0])   # non-rejection negativo
    ax_n_pos = fig.add_subplot(gs[2, 1])   # non-rejection positivo

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

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.07, top=0.96)
    ruta = os.path.join(carpeta_salida, "fig_resumen_yager_vs_sugeno_weber.png")
    fig.savefig(ruta, dpi=450)
    plt.close(fig)
    print(f"Figura guardada en: {ruta}")

# guardado de resultados y main ------------------------------------------------------------------------------
def guardar_resultados(carpeta_salida, p_vals, t_neg, t_pos,
                       pm_neg, n50_neg, n05_neg,
                       pm_pos, n50_pos, n05_pos):
    os.makedirs(carpeta_salida, exist_ok=True)
    np.savez(
        os.path.join(carpeta_salida, "datos_yager_vs_sugeno_weber.npz"),
        p_vals=p_vals, t_neg=t_neg, t_pos=t_pos,
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
        carpeta_salida=os.path.join(carpeta_script, "salida_yager_vs_sugeno_weber"),
        n_malla=100,
        n_experimentos=1000,
        tam_muestra=100,
        semilla=1234
    )