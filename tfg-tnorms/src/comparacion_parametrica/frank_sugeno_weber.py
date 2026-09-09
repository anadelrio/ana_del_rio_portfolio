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

# t-normas ---------------------------------------------------------------------------------------------------

def tnorm_frank(x, y, s, eps=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isclose(s, 1.0, atol=1e-12, rtol=0):
        return x * y
    if s <= 0:
        raise ValueError("Frank requiere s > 0.")
    ls     = np.log(s)
    sx     = np.exp(x * ls)
    sy     = np.exp(y * ls)
    inside = np.maximum(1.0 + (sx - 1.0) * (sy - 1.0) / (s - 1.0), eps)
    return np.log(inside) / ls

def tnorm_sugeno_weber(x, y, t):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t > 1e6:
        return x * y
    return np.maximum(0.0, (x + y - 1.0 + t * x * y) / (1.0 + t))

#  test estadístico -----------------------------------------------------------------
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

# experimento y barrido de malla -----------------------------------------------------------------
def experimento_un_punto(s_frank, t_sw, tam_muestra, rng):
    U_f = rng.random(tam_muestra); V_f = rng.random(tam_muestra)
    U_w = rng.random(tam_muestra); V_w = rng.random(tam_muestra)
    vals_frank = tnorm_frank(U_f, V_f, s_frank)
    vals_sw    = tnorm_sugeno_weber(U_w, V_w, t_sw)
    return pvalor_mannwhitney_asymptotic(vals_frank, vals_sw)

def barrer_mitad(frank_grid, t_vals, n_experimentos, tam_muestra, rng, desc):
    n = len(frank_grid)
    pm  = np.zeros((n, n)); n50 = np.zeros((n, n)); n05 = np.zeros((n, n))
    for i in tqdm(range(n), desc=desc):
        s = frank_grid[i]
        for j in range(n):
            pv = np.array([
                experimento_un_punto(s, t_vals[j], tam_muestra, rng)
                for _ in range(n_experimentos)
            ])
            pm[i, j]  = np.mean(pv)
            n50[i, j] = np.mean(pv >= 0.50)
            n05[i, j] = np.mean(pv >= 0.05)
    return pm, n50, n05

def barrido_malla(n_malla, n_experimentos, tam_muestra, semilla):
    rng = np.random.default_rng(semilla)

    frank_grid = np.logspace(-4, 4, n_malla)
    sw_abs     = np.logspace(-1, 3, n_malla)
    sw_neg     = -sw_abs[::-1]   # de -1e+3 a -1e-1
    sw_pos     =  sw_abs.copy()  # de  1e-1 a  1e+3

    print("Barriendo mitad negativa (t<0)...")
    pm_neg, n50_neg, n05_neg = barrer_mitad(
        frank_grid, sw_neg, n_experimentos, tam_muestra, rng, "  Frank (t<0)")
    print("Barriendo mitad positiva (t>0)...")
    pm_pos, n50_pos, n05_pos = barrer_mitad(
        frank_grid, sw_pos, n_experimentos, tam_muestra, rng, "  Frank (t>0)")

    return frank_grid, sw_neg, sw_pos, pm_neg, n50_neg, n05_neg, pm_pos, n50_pos, n05_pos

# figura ---------------------------------------------------------------------------------------------------------------------------

def figura_resumen(carpeta_salida, frank_grid, sw_neg, sw_pos,
                   pm_neg, n50_neg, n05_neg, pm_pos, n50_pos, n05_pos):

    niveles = [0.1, 0.2, 0.3, 0.4, 0.5]

    pm_neg_s = gaussian_filter(pm_neg, sigma=1.0)
    pm_pos_s = gaussian_filter(pm_pos, sigma=1.0)

    S_neg, F_neg = np.meshgrid(np.abs(sw_neg), frank_grid)
    S_pos, F_pos = np.meshgrid(sw_pos,          frank_grid)

    def fmt_neg(x, pos=None):
        if x <= 0: return ""
        e = int(round(np.log10(x)))
        return f"-1e{e:+d}"

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
        ax.set_xlim(np.abs(sw_neg[0]), np.abs(sw_neg[-1])) 
        ax.set_ylim(frank_grid[0], frank_grid[-1])
        ax.set_xticks([1e3, 1e1, 1e-1])
        ax.set_yticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_neg))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=7, length=3)
        ax.set_xlabel("Sugeno-Weber", fontsize=8, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Frank", fontsize=8, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    def estilo_pos(ax, show_ylabel=False):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(sw_pos[0], sw_pos[-1])
        ax.set_ylim(frank_grid[0], frank_grid[-1])
        ax.set_xticks([1e-1, 1e1, 1e3])
        ax.set_yticks([1e-4, 1e-2, 1e0, 1e2, 1e4])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_pos))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_y))
        ax.tick_params(axis="both", which="major", labelsize=7, length=3)
        ax.set_xlabel("Sugeno-Weber", fontsize=8, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Frank", fontsize=8, labelpad=4)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig = plt.figure(figsize=(7.6, 10.2), facecolor="white")
    gs  = GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.0, 1.0], hspace=0.34, wspace=0.22)

    ax11 = fig.add_subplot(gs[0, 0]); ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0]); ax22 = fig.add_subplot(gs[1, 1])
    ax31 = fig.add_subplot(gs[2, 0]); ax32 = fig.add_subplot(gs[2, 1])

    c1 = ax11.contour(S_neg, F_neg, pm_neg_s, levels=niveles, colors="k", linewidths=0.85)
    ax11.clabel(c1, fmt="%.1f", inline=True, fontsize=7); estilo_neg(ax11, True)

    c2 = ax12.contour(S_pos, F_pos, pm_pos_s, levels=niveles, colors="k", linewidths=0.85)
    ax12.clabel(c2, fmt="%.1f", inline=True, fontsize=7); estilo_pos(ax12, False)

    ax21.pcolormesh(S_neg, F_neg, n50_neg, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_neg(ax21, True)
    ax22.pcolormesh(S_pos, F_pos, n50_pos, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_pos(ax22, False)

    ax31.pcolormesh(S_neg, F_neg, n05_neg, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_neg(ax31, True)
    ax32.pcolormesh(S_pos, F_pos, n05_pos, cmap="gray_r", vmin=0, vmax=1, shading="auto")
    estilo_pos(ax32, False)

    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.06, top=0.97)
    ruta = os.path.join(carpeta_salida, "fig_resumen_frank_vs_sugeno_weber.png")
    fig.savefig(ruta, dpi=450)
    plt.close(fig)
    print(f"Figura guardada en: {ruta}")

# guardado y main -------------------------------------------------------------------------------------

def guardar_resultados(carpeta_salida, frank_grid, sw_neg, sw_pos,
                       pm_neg, n50_neg, n05_neg, pm_pos, n50_pos, n05_pos):
    os.makedirs(carpeta_salida, exist_ok=True)
    np.savez_compressed(
        os.path.join(carpeta_salida, "datos_frank_vs_sugeno_weber.npz"),
        frank_grid=frank_grid, sw_neg=sw_neg, sw_pos=sw_pos,
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
        carpeta_salida=os.path.join(carpeta_script, "salida_frank_vs_sugeno_weber"),
        n_malla=100,
        n_experimentos=1000,
        tam_muestra=100,
        semilla=1234
    )
