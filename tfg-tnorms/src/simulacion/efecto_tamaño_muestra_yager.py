import os
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ruta de salida -------------------------------------------------------
try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_FILE = os.path.join(BASE_DIR, "efecto_tamano_muestra_resultados.txt")

# definición de t-norma de Yager ---------------------------------------
def T_yager_binary(x, y, p):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if p == 0:
        res = np.zeros_like(x)
        mask_x1 = np.isclose(x, 1.0)
        mask_y1 = np.isclose(y, 1.0)
        res[mask_x1 & mask_y1]  = 1.0
        res[mask_x1 & ~mask_y1] = y[mask_x1 & ~mask_y1]
        res[mask_y1 & ~mask_x1] = x[mask_y1 & ~mask_x1]
        return res

    if np.isinf(p):
        return np.minimum(x, y)

    r = ((1.0 - x) ** p + (1.0 - y) ** p) ** (1.0 / p)
    return 1.0 - np.minimum(1.0, r)

def yager_tnorm_multi(X, p):
    X = np.asarray(X, dtype=float)
    agg = X[:, 0]
    for j in range(1, X.shape[1]):
        agg = T_yager_binary(agg, X[:, j], p)
    return agg

# simulación de un par (p, p') -----------------------------------------

def simulate_yager_pair(p, p2, N=100, n_args=2, n_iter=10000, alpha=0.05, seed=None):
    rng = np.random.default_rng(seed)
    p_values = []

    for _ in range(n_iter):
        X1 = rng.uniform(0.0, 1.0, size=(N, n_args))
        X2 = rng.uniform(0.0, 1.0, size=(N, n_args))
        _, pval = mannwhitneyu(yager_tnorm_multi(X1, p), yager_tnorm_multi(X2, p2), alternative="two-sided")
        p_values.append(pval)

    p_values = np.array(p_values)
    return {
        "p": p, "p2": p2,
        "rechazo":           float(np.mean(p_values <= alpha) * 100.0),
        "no_rechazo":        float(np.mean(p_values > alpha)  * 100.0),
        "no_rechazo_fuerte": float(np.mean(p_values > 0.5)    * 100.0),
    }

# promedio de un bloque ------------------------------------------------

def promedio_stats(lista_stats):
    return {
        "rechazo":           np.mean([d["rechazo"]           for d in lista_stats]),
        "no_rechazo":        np.mean([d["no_rechazo"]        for d in lista_stats]),
        "no_rechazo_fuerte": np.mean([d["no_rechazo_fuerte"] for d in lista_stats]),
    }

# formato LaTeX --------------------------------------------------------

def latex_row_yager(p, p2, stats):
    return (
        f"Yager ($p={p}$) vs Yager ($p'={p2}$) & "
        f"{stats['no_rechazo_fuerte']:.2f}\\% & "
        f"{stats['no_rechazo']:.2f}\\% & "
        f"{stats['rechazo']:.2f}\\% \\\\"
    )

def latex_row_generic(label_left, label_right, stats):
    return (
        f"Yager ($p={label_left}$) vs Yager ($p'={label_right}$) & "
        f"{stats['no_rechazo_fuerte']:.2f}\\% & "
        f"{stats['no_rechazo']:.2f}\\% & "
        f"{stats['rechazo']:.2f}\\% \\\\"
    )

def tabla_latex(sample_size, filas_latex):
    caption = (
        f"Comparacion entre dos t-normas Yager, con distribucion Uniforme, "
        f"tamano de muestra {sample_size} y dos argumentos."
    )
    label = f"tab:efecto_muestra_{sample_size}"
    lineas = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"T-normas comparadas & No rechazo $p$-valor $> 0.5$ (\%) & No rechazo $p$-valor $> 0.05$ (\%) & Rechazo $p$-valor $\leq 0.05$ (\%) \\",
        r"\hline",
    ] + filas_latex + [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lineas)

# ejecución del experimento para un tamaño de muestra -----------------

def run_sample_size_case(sample_size, ps_tabla, n_args=2, N_ITER=10000, ALPHA=0.05, BASE_SEED=1234):
    filas_latex = []
    stats_p1_dict = {}
    total = len(ps_tabla) + 26 + (300 - 26)

    with tqdm(total=total, desc=f"Muestra n={sample_size}") as pbar:

        for idx, p in enumerate(ps_tabla):
            stats = simulate_yager_pair(p, p + 1, sample_size, n_args, N_ITER, ALPHA,
                                        seed=BASE_SEED + 10000 * sample_size + idx)
            filas_latex.append(latex_row_yager(p, p + 1, stats))
            stats_p1_dict[p] = stats
            pbar.update(1)

        bloque_p1 = []
        for p in range(0, 26):
            if p in stats_p1_dict:
                stats = stats_p1_dict[p]
            else:
                stats = simulate_yager_pair(p, p + 1, sample_size, n_args, N_ITER, ALPHA,
                                            seed=BASE_SEED + 10000 * sample_size + 1000 + p)
            bloque_p1.append(stats)
            pbar.update(1)

        bloque_p10 = []
        for p in range(26, 300):
            stats = simulate_yager_pair(p, p + 10, sample_size, n_args, N_ITER, ALPHA,
                                        seed=BASE_SEED + 10000 * sample_size + 2000 + p)
            bloque_p10.append(stats)
            pbar.update(1)

    filas_latex.append(latex_row_generic("p", "p+1",  promedio_stats(bloque_p1)))
    filas_latex.append(latex_row_generic("p", "p+10", promedio_stats(bloque_p10)))
    return filas_latex

# main ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    casos = [
        (50,   [0, 1, 2, 3]),
        (1000, [0, 1, 2, 3, 4, 5, 6, 7]),
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample_size, ps_tabla in casos:
            filas = run_sample_size_case(sample_size, ps_tabla)
            f.write(f"% Tabla para tamano de muestra {sample_size}\n")
            f.write(tabla_latex(sample_size, filas) + "\n\n")

    print(f"Resultados guardados en: {OUTPUT_FILE}")