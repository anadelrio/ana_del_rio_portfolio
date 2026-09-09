import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# definición de t-normas -----------------------------------------------

def t_schweizer_sklar(x, y, q):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.isclose(q, -1.0):
        return np.minimum(x, y)
    if np.isclose(q, 0.0):
        return x * y
    if np.isclose(q, 1.0):
        return np.where((x == 1) | (y == 1), np.minimum(x, y), 0.0)

    inner = x ** q + y ** q - 1.0

    if q > 0:
        inner = np.maximum(0.0, inner)
        out   = np.zeros_like(inner)
        m     = inner > 0
        out[m] = inner[m] ** (1.0 / q)
        return out

    inner = np.maximum(inner, 1e-300)
    return inner ** (1.0 / q)

def t_sugeno_weber(x, y, t, eps=1e-12):
    x = np.asarray(x)
    y = np.asarray(y)

    if np.isclose(t, -1.0):
        return np.where((x == 1) | (y == 1), np.minimum(x, y), 0.0)
    if np.isclose(t, 1.0):
        return x * y

    den = 1.0 + t
    if abs(den) < eps:
        den = eps if den >= 0 else -eps
    return np.maximum(0.0, (x + y - 1.0 + t * x * y) / den)

# test de Mann-Whitney (aproximación normal) ---------------------------

def pvalor_mannwhitney_aprox(a, b):
    a  = np.asarray(a)
    b  = np.asarray(b)
    n1 = a.size
    n2 = b.size

    combinado = np.concatenate([a, b])
    orden     = np.argsort(combinado, kind="mergesort")
    rangos    = np.empty_like(orden, dtype=float)
    rangos[orden] = np.arange(1, combinado.size + 1)

    r1      = rangos[:n1].sum()
    U1      = r1 - n1 * (n1 + 1) / 2.0
    media_U = n1 * n2 / 2.0
    var_U   = n1 * n2 * (n1 + n2 + 1) / 12.0
    z       = (U1 - media_U) / np.sqrt(var_U)
    return float(2.0 * (1.0 - norm.cdf(abs(z))))

# construcción de mallas de parámetros ---------------------------------

def build_param_grids(n_malla=50):
    q_neg_lin = -np.linspace(1.0, 1e-2, n_malla)
    t_neg_lin = -np.linspace(1.0, 1e-2, n_malla)
    # Evitar q=-1, q=1, t=1 exactos en la malla (causan discontinuidades)
    q_neg_log = -np.logspace(-2, 4, n_malla)
    q_neg_log[np.isclose(q_neg_log, -1.0, rtol=1e-9)] *= 1.0001
    t_neg_log = -np.logspace(-2, 4, n_malla)
    t_neg_log = -np.logspace(-2, 4, n_malla)
    q_pos     = np.logspace(-2, 4, n_malla)
    q_pos[np.isclose(q_pos, 1.0, rtol=1e-9)] *= 1.0001
    t_pos     = np.logspace(-2, 4, n_malla)
    t_pos[np.isclose(t_pos, 1.0, rtol=1e-9)] *= 1.0001
    return q_neg_lin, q_neg_log, q_pos, t_neg_lin, t_neg_log, t_pos

# simulación en un punto -----------------------------------------------

def simular_punto(q, t, n_exp, sample_size, rng):
    pvals = np.empty(n_exp, dtype=float)
    for i in range(n_exp):
        # Muestras para Schweizer-Sklar
        x_ss = rng.random(sample_size)
        y_ss = rng.random(sample_size)
        
        # Muestras totalmente independientes para Sugeno-Weber
        x_sw = rng.random(sample_size)
        y_sw = rng.random(sample_size)
        
        pvals[i] = pvalor_mannwhitney_aprox(
            t_schweizer_sklar(x_ss, y_ss, q),
            t_sugeno_weber(x_sw, y_sw, t),
        )
    return float(pvals.mean()), float(np.mean(pvals >= 0.50)), float(np.mean(pvals >= 0.05))

# barrido de la malla --------------------------------------------------

def barrer(q_vals, t_vals, n_exp, sample_size, rng, desc=""):
    nY     = len(q_vals)
    nX     = len(t_vals)
    p_mean = np.empty((nY, nX), dtype=float)
    acc    = np.empty_like(p_mean)
    nonrej = np.empty_like(p_mean)

    for iy, q in enumerate(tqdm(q_vals, desc=desc, total=nY)):
        for ix, t in enumerate(t_vals):
            p_mean[iy, ix], acc[iy, ix], nonrej[iy, ix] = \
                simular_punto(q, t, n_exp, sample_size, rng)

    return p_mean, acc, nonrej

# funciones de ticks ---------------------------------------------------

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

def ticks_neg_lin_paper():
    ticks  = [-1.00, -0.50, -0.20, -0.01]
    labels = ["-1.00", "-0.50", "-0.20", "-0.01"]
    return ticks, labels

def ticks_neg_log_paper():
    ticks  = [-1e-2, -1e0, -1e2, -1e4]
    labels = ["-1e-02", "-1e+00", "-1e+02", "-1e+04"]
    return ticks, labels

def ticks_pos_paper():
    ticks  = [1e-2, 1e0, 1e2, 1e4]
    labels = ["1e-02", "1e+00", "1e+02", "1e+04"]
    return ticks, labels

# estilo de ejes -------------------------------------------------------

def _style_ax_shared(ax, x_type, y_type, hide_ylabel=False):
    ax.set_xlabel("Sugeno-Weber", fontsize=8)
    if hide_ylabel:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.set_ylabel("Schweizer-Sklar", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=6, length=3, width=0.8, pad=1)
    ax.minorticks_off()
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    if x_type == "pos_log":
        ax.set_xscale("log")
        xt, xl = ticks_pos_paper()
        ax.set_xticks(xt); ax.set_xticklabels(xl)
    elif x_type == "neg_log":
        ax.set_xscale("symlog", linthresh=1e-3)
        xt, xl = ticks_neg_log_paper()
        ax.set_xticks(xt); ax.set_xticklabels(xl)
        ax.invert_xaxis()
    else:
        xt, xl = ticks_neg_lin_paper()
        ax.set_xticks(xt); ax.set_xticklabels(xl)

    if y_type == "pos_log":
        ax.set_yscale("log")
        yt, yl = ticks_pos_paper()
        ax.set_yticks(yt); ax.set_yticklabels(yl)
    elif y_type == "neg_log":
        ax.set_yscale("symlog", linthresh=1e-3)
        yt, yl = ticks_neg_log_paper()
        ax.set_yticks(yt); ax.set_yticklabels(yl)
        ax.invert_yaxis()
    else:
        yt, yl = ticks_neg_lin_paper()
        ax.set_yticks(yt); ax.set_yticklabels(yl)

# definición de cuadrantes ---------------------------------------------

def _get_blocks(q_neg_log, q_pos, t_neg_lin, t_pos, Z_tl, Z_tr, Z_bl, Z_br):
    return [
        (0, 0, t_neg_lin, q_pos,     Z_tl, "neg_lin", "pos_log"),
        (0, 1, t_pos,     q_pos,     Z_tr, "pos_log", "pos_log"),
        (1, 0, t_neg_lin, q_neg_log, Z_bl, "neg_lin", "neg_log"),
        (1, 1, t_pos,     q_neg_log, Z_br, "pos_log", "neg_log"),
    ]

# figura de contornos --------------------------------------------------

def plot_2x2_contours(q_neg_log, q_pos, t_neg_lin, t_pos,
                      P_tl, P_tr, P_bl, P_br, outpath):
    fig = plt.figure(figsize=(7.4, 7.4), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
    blocks = _get_blocks(q_neg_log, q_pos, t_neg_lin, t_pos, P_tl, P_tr, P_bl, P_br)

    for r, c, t_vals, q_vals, Z, x_type, y_type in blocks:
        ax      = fig.add_subplot(gs[r, c])
        T, Q    = np.meshgrid(t_vals, q_vals)
        cs      = ax.contour(T, Q, Z, levels=LEVELS, colors="k", linewidths=0.8)
        ax.clabel(cs, fmt="%.1f", inline=True, fontsize=6.5)
        _style_ax_shared(ax, x_type, y_type, hide_ylabel=(c == 1))
        if y_type == "neg_log":
            ax.invert_yaxis()

    fig.savefig(outpath, dpi=500, bbox_inches="tight")
    plt.close(fig)

# figura de mapas de calor ---------------------------------------------

def plot_2x2_heatmaps(q_neg_log, q_pos, t_neg_lin, t_pos,
                      Z_tl, Z_tr, Z_bl, Z_br, outpath, vmin=0.0, vmax=1.0):
    fig = plt.figure(figsize=(7.4, 7.4), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
    blocks = _get_blocks(q_neg_log, q_pos, t_neg_lin, t_pos, Z_tl, Z_tr, Z_bl, Z_br)

    for r, c, t_vals, q_vals, Z, x_type, y_type in blocks:
        ax = fig.add_subplot(gs[r, c])
        T, Q = np.meshgrid(t_vals, q_vals)
        ax.pcolormesh(T, Q, Z, cmap="gray_r", vmin=vmin, vmax=vmax, shading="auto")

        ax.set_xlabel("Sugeno-Weber", fontsize=8)
        if c == 1:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
        else:
            ax.set_ylabel("Schweizer-Sklar", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6, length=3, width=0.8, pad=1)
        ax.minorticks_off()
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

        _style_ax_shared(ax, x_type, y_type, hide_ylabel=(c == 1))
        if y_type == "neg_log":
            ax.invert_yaxis()

    fig.savefig(outpath, dpi=500, bbox_inches="tight")
    plt.close(fig)

# main -----------------------------------------------------------------

def main(
    output_folder="salida_schweizer_sklar_vs_sugeno_weber",
    n_malla=50,
    n_exp=1000,
    sample_size=100,
    seed=1234,
):
    os.makedirs(output_folder, exist_ok=True)
    rng = np.random.default_rng(seed)

    q_neg_lin, q_neg_log, q_pos, t_neg_lin, t_neg_log, t_pos = build_param_grids(n_malla)

    p_tl, a_tl, nr_tl = barrer(q_pos,     t_neg_lin, n_exp, sample_size, rng, desc="q>0, t<0")
    p_tr, a_tr, nr_tr = barrer(q_pos,     t_pos,     n_exp, sample_size, rng, desc="q>0, t>0")
    p_bl, a_bl, nr_bl = barrer(q_neg_log, t_neg_lin, n_exp, sample_size, rng, desc="q<0, t<0")
    p_br, a_br, nr_br = barrer(q_neg_log, t_pos,     n_exp, sample_size, rng, desc="q<0, t>0")

    # guardado de datos ------------------------------------------------
    np.savez_compressed(
        os.path.join(output_folder, "datos_ss_vs_sw.npz"),
        q_neg_lin=q_neg_lin, q_neg_log=q_neg_log, q_pos=q_pos,
        t_neg_lin=t_neg_lin, t_neg_log=t_neg_log, t_pos=t_pos,
        p_tl=p_tl, p_tr=p_tr, p_bl=p_bl, p_br=p_br,
        a_tl=a_tl, a_tr=a_tr, a_bl=a_bl, a_br=a_br,
        nr_tl=nr_tl, nr_tr=nr_tr, nr_bl=nr_bl, nr_br=nr_br,
        n_malla=n_malla, n_exp=n_exp, sample_size=sample_size, seed=seed,
    )

    plot_2x2_contours(
        q_neg_log, q_pos, t_neg_lin, t_pos,
        P_tl=p_tl, P_tr=p_tr, P_bl=p_bl, P_br=p_br,
        outpath=os.path.join(output_folder, "fig15_ss_vs_sw_pmean_contours.png"),
    )
    plot_2x2_heatmaps(
        q_neg_log, q_pos, t_neg_lin, t_pos,
        Z_tl=a_tl, Z_tr=a_tr, Z_bl=a_bl, Z_br=a_br,
        outpath=os.path.join(output_folder, "fig16_ss_vs_sw_acceptance.png"),
    )
    plot_2x2_heatmaps(
        q_neg_log, q_pos, t_neg_lin, t_pos,
        Z_tl=nr_tl, Z_tr=nr_tr, Z_bl=nr_bl, Z_br=nr_br,
        outpath=os.path.join(output_folder, "fig17_ss_vs_sw_nonrejection.png"),
    )

    print(f"Resultados guardados en: {output_folder}")

if __name__ == "__main__":
    main(
        output_folder="salida_schweizer_sklar_vs_sugeno_weber",
        n_malla=100,
        n_exp=1000,
        sample_size=100,
        seed=1234,
    )