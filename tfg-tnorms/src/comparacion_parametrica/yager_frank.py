import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from tqdm import tqdm

# definciones de las t-normas ----------------------------------------------------------------------
def tnorm_frank(x, y, s, eps=1e-12):
    x = np.asarray(x)
    y = np.asarray(y)

    if abs(s - 1.0) < 1e-6:
        return x * y

    if s <= 0:
        return np.minimum(x, y)

    ls = np.log(s)
    sx = np.exp(x * ls)
    sy = np.exp(y * ls)

    num = (sx - 1.0) * (sy - 1.0)
    den = s - 1.0
    inside = 1.0 + num / den
    inside = np.where(inside < eps, eps, inside)
    return np.log(inside) / ls

def tnorm_yager(x, y, p, eps=1e-12):
    x = np.asarray(x)
    y = np.asarray(y)

    ax = np.where(x >= 1.0, 0.0, np.power(1.0 - x, p))
    ay = np.where(y >= 1.0, 0.0, np.power(1.0 - y, p))
    
    suma = ax + ay
    res = 1.0 - np.power(suma, 1.0 / p)
    return np.where(res < 0.0, 0.0, res)

# test estadístico ---------------------------------------------------------------------------------
def pvalor_mannwhitney_asymptotic(x, y):
    n1 = len(x)
    n2 = len(y)
    
    all_data = np.concatenate([x, y])
    argsorted = np.argsort(all_data)
    ranks = np.empty_like(argsorted)
    ranks[argsorted] = np.arange(1, len(all_data) + 1)
    
    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1)) / 2.0
    
    mu = (n1 * n2) / 2.0
    sigma = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12.0)
    
    z = (u1 - mu) / sigma
    p_val = 2.0 * (1.0 - norm.cdf(abs(z)))
    return p_val

# experimento y barrido de la malla -------------------------------------------------
def experimento_un_punto(p_yager, s_frank, tam_muestra, rng):
    U_yager = rng.random(tam_muestra)
    V_yager = rng.random(tam_muestra)
    
    U_frank = rng.random(tam_muestra)
    V_frank = rng.random(tam_muestra)
    
    valores_yager = tnorm_yager(U_yager, V_yager, p_yager)
    valores_frank = tnorm_frank(U_frank, V_frank, s_frank)
    
    return pvalor_mannwhitney_asymptotic(valores_yager, valores_frank)

def barrido_malla(n_malla, n_experimentos, tam_muestra, semilla):
    rng = np.random.default_rng(semilla)
    
    s_vals = np.logspace(-4, 2, n_malla)
    p_vals = np.logspace(0, 4, n_malla) 

    pmedio_grid = np.zeros((n_malla, n_malla))
    norech50_grid = np.zeros((n_malla, n_malla))
    norech05_grid = np.zeros((n_malla, n_malla))

    for i in tqdm(range(n_malla), desc="Simulando filas de la malla"):
        p_yager = p_vals[i]
        for j in range(n_malla):
            s_frank = s_vals[j]
            
            pvals_iter = np.empty(n_experimentos)
            for k in range(n_experimentos):
                pvals_iter[k] = experimento_un_punto(p_yager, s_frank, tam_muestra, rng)
            
            pmedio_grid[i, j] = np.mean(pvals_iter)
            norech50_grid[i, j] = np.mean(pvals_iter >= 0.50)
            norech05_grid[i, j] = np.mean(pvals_iter >= 0.05)

    return s_vals, p_vals, pmedio_grid, norech50_grid, norech05_grid

# generacón de la figura resumen --------------------------------------------------
def figura_resumen(carpeta_salida, s_vals, p_vals, pmedio_grid, norech50_grid, norech05_grid):
    from scipy.ndimage import gaussian_filter
    niveles = [0.1, 0.2, 0.3, 0.4, 0.5] # no hay niveles que llegen a 0.6 así que se excluyen
    pmedio_grid = gaussian_filter(pmedio_grid, sigma=1.0)
    S, P = np.meshgrid(s_vals, p_vals)

    x_ticks = [1e-4, 1e-2, 1e0, 1e2]
    y_ticks = [1, 10, 100, 1000, 10000]

    def fmt_1e(x, pos=None):
        if x <= 0: return ""
        exp = int(np.round(np.log10(x)))
        return f"1e{exp:+d}"

    def fmt_int(y, pos=None):
        if y <= 0: return ""
        return str(int(np.round(y)))

    def _estilo_axes(ax, show_ylabel=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(s_vals), max(s_vals))
        ax.set_ylim(1, 10000)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_1e))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))
        
        ax.tick_params(axis="x", which="major", labelsize=9, length=5)
        ax.tick_params(axis="y", which="major", labelsize=9, length=5, labelrotation=90)
        
        ax.set_xlabel("Frank", fontsize=11, labelpad=4)
        
        if show_ylabel:
            ax.set_ylabel("Yager", fontsize=11, labelpad=8)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=True, labelleft=False)

    fig = plt.figure(figsize=(6.5, 10.2), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1.0], hspace=0.04, wspace=0.25)

    # --- Panel superior: Contornos del p-valor medio ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("white")
    cont = ax1.contour(S, P, pmedio_grid, levels=niveles, colors="k", linewidths=1.0)
    ax1.clabel(cont, fmt="%.1f", inline=True, fontsize=9)
    _estilo_axes(ax1, show_ylabel=True)

    # --- Panel inferior izq: Tasa no rechazo (p >= 0.50) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("white")
    ax2.pcolormesh(S, P, norech50_grid, shading="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
    _estilo_axes(ax2, show_ylabel=True)
    ax2.set_box_aspect(1.0)  

    # --- Panel inferior dcha: Tasa no rechazo (p >= 0.05) ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("white")
    ax3.pcolormesh(S, P, norech05_grid, shading="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
    _estilo_axes(ax3, show_ylabel=False)
    ax3.set_box_aspect(1.0)  

    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.08, top=0.96)

    ruta_grafica = os.path.join(carpeta_salida, "fig_resumen_frank_vs_yager.png")
    fig.savefig(ruta_grafica, dpi=450)
    plt.close(fig)

# main y guardado de resultados ---------------------------------------------------
def guardar_resultados(carpeta_salida, s_vals, p_vals, pmedio, norech50, norech05):
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_datos = os.path.join(carpeta_salida, "datos_yager_vs_frank.npz")
    np.savez(
        ruta_datos,
        s_vals=s_vals,
        p_vals=p_vals,
        pmedio=pmedio,
        norech50=norech50,
        norech05=norech05
    )

def main(carpeta_salida, n_malla=100, n_experimentos=1000, tam_muestra=100, semilla=1234):
    t0 = time.time()
    os.makedirs(carpeta_salida, exist_ok=True)

    s_vals, p_vals, pmedio, norech50, norech05 = barrido_malla(
        n_malla, n_experimentos, tam_muestra, semilla
    )

    guardar_resultados(carpeta_salida, s_vals, p_vals, pmedio, norech50, norech05)
    figura_resumen(carpeta_salida, s_vals, p_vals, pmedio, norech50, norech05)
    
    print(f"\nProceso finalizado. Gráficos idénticos generados en: {carpeta_salida}")
    print(f"Tiempo de ejecución total: {(time.time() - t0) / 60:.2f} minutos")

if __name__ == "__main__":
    try:
        carpeta_script = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        carpeta_script = os.getcwd()

    main(
        carpeta_salida=os.path.join(carpeta_script, "salida_yager_vs_frank"),
        n_malla=100,          
        n_experimentos=3000,  
        tam_muestra=500,      
        semilla=1234,
    )