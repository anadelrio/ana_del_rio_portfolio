import os
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_FILE = os.path.join(BASE_DIR, "efecto_aridad_resultados.txt")

# definición de t-norma de Yager n-aria -------------------------------------------------------
def yager_tnorm_multi(X, p):
    X = np.asarray(X, dtype=float)
    if p == 0:
        return np.min(X, axis=1)
    if np.isinf(p):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            otros = np.delete(X, i, axis=1)
            todos_uno = np.all(np.isclose(otros, 1.0), axis=1)
            result[todos_uno] = X[todos_uno, i]
        return result
    suma = np.sum((1.0 - X) ** p, axis=1)
    return np.maximum(0.0, 1.0 - suma ** (1.0 / p))

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

# promedio de un bloque --------------------------------------------------------

def promedio_stats(lista_stats):
    return {
        "rechazo":          np.mean([d["rechazo"]          for d in lista_stats]),
        "no_rechazo":       np.mean([d["no_rechazo"]       for d in lista_stats]),
        "no_rechazo_fuerte": np.mean([d["no_rechazo_fuerte"] for d in lista_stats]),
    }

# formato LaTeX --------------------------------------------------------

def latex_row_yager(p, p2, stats):
    return (
        f"{p} & {p2} & "
        f"{stats['no_rechazo_fuerte']:.1f} & "
        f"{stats['no_rechazo']:.1f} & "
        f"{stats['rechazo']:.1f} \\\\"
    )

def latex_row_generic(label_left, label_right, stats):
    return (
        f"{label_left} & {label_right} & "
        f"{stats['no_rechazo_fuerte']:.1f} & "
        f"{stats['no_rechazo']:.1f} & "
        f"{stats['rechazo']:.1f} \\\\"
    )

def tabla_latex(num_args, filas_latex):
    titulo = f"Resultados del experimento del efecto de la aridad para {num_args} argumentos"
    lineas = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + titulo + r".}",
        r"\label{tab:case2_yager_" + str(num_args) + r"args}",
        r"\begin{tabular}{ccccc}",
        r"\hline",
        r"$p$ & $p'$ & No rechazo $p$-valor $> 0.5$ (\%) & No rechazo $p$-valor $> 0.05$ (\%) & Rechazo $p$-valor $\leq 0.05$ (\%) \\",
        r"\multicolumn{5}{l}{\textit{Tests results: can output distributions be assimilated?}} \\",
    ] + filas_latex + [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lineas)

# ejecución del experimento para un número de argumentos -----------------------------------------------------------------
def run_case(num_args, ps_tabla, N=100, N_ITER=10000, ALPHA=0.05, BASE_SEED=123):
    filas_latex = []
    stats_p1_dict = {}
    total = len(ps_tabla) + 26 + (300 - 26)

    with tqdm(total=total, desc=f"Aridad {num_args} args") as pbar:

        for idx, p in enumerate(ps_tabla):
            stats = simulate_yager_pair(p, p + 1, N, num_args, N_ITER, ALPHA,
                                        seed=BASE_SEED + 100 * num_args + idx)
            filas_latex.append(latex_row_yager(p, p + 1, stats))
            stats_p1_dict[p] = stats
            pbar.update(1)

        bloque_p1 = []
        for p in range(0, 26):
            if p in stats_p1_dict:
                stats = stats_p1_dict[p]
            else:
                stats = simulate_yager_pair(p, p + 1, N, num_args, N_ITER, ALPHA,
                                            seed=BASE_SEED + 100 * num_args + 1000 + p)
            bloque_p1.append(stats)
            pbar.update(1)

        bloque_p10 = []
        for p in range(26, 300):
            stats = simulate_yager_pair(p, p + 10, N, num_args, N_ITER, ALPHA,
                                        seed=BASE_SEED + 100 * num_args + 2000 + p)
            bloque_p10.append(stats)
            pbar.update(1)

    filas_latex.append(latex_row_generic(r"$p$", r"$p+1$",  promedio_stats(bloque_p1)))
    filas_latex.append(latex_row_generic(r"$p$", r"$p+10$", promedio_stats(bloque_p10)))
    return filas_latex

# main -------------------------------------------------------------------
if __name__ == "__main__":
    casos = [
        (4, list(range(0, 9))),
        (6, list(range(0, 14))),
        (8, list(range(0, 16))),
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for num_args, ps_tabla in casos:
            filas = run_case(num_args, ps_tabla)
            f.write(f"% Tabla para {num_args} argumentos\n")
            f.write(tabla_latex(num_args, filas) + "\n\n")

    print(f"Resultados guardados en: {OUTPUT_FILE}")