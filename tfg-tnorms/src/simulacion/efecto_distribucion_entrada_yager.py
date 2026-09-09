import os
import numpy as np
from scipy.stats import mannwhitneyu
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_FILE = os.path.join(BASE_DIR, "efecto_distribucion_entrada_resultados.txt")

# definición de t-norma de Yager (n-aria) -------------------------------------------------------
def T_yager_nary(x, p):
    x = np.asarray(x, dtype=float)
    if p == 0:
        return np.min(x, axis=-1)
    if np.isinf(p):
        result = np.zeros(x.shape[0])
        for i in range(x.shape[1]):
            otros = np.delete(x, i, axis=1)
            todos_uno = np.all(np.isclose(otros, 1.0), axis=1)
            result[todos_uno] = x[todos_uno, i]
        return result
    one_minus = 1.0 - x
    s = np.sum(one_minus ** p, axis=-1)
    return np.maximum(1.0 - np.power(s, 1.0 / p), 0.0)

# samplers por distribución --------------------------------------------
def make_beta_sampler(a, b, N, n_args):
    def sampler(rng):
        return rng.beta(a, b, size=(N, n_args))
    return sampler

# simulación de un par (p, p') con samplers arbitrarios ---------------
def run_single_comparison(N, p, p_prime, n_args, N_ITER, sampler_1, sampler_2, ALPHA, BASE_SEED):
    seed = int(BASE_SEED + 1000 * p + 10 * p_prime)
    rng = np.random.default_rng(seed)

    count_gt_05    = 0
    count_gt_alpha = 0

    for _ in range(N_ITER):
        x1 = sampler_1(rng)
        x2 = sampler_2(rng)
        _, pvalue = mannwhitneyu(T_yager_nary(x1, p), T_yager_nary(x2, p_prime), alternative="two-sided")
        if pvalue > 0.5:
            count_gt_05 += 1
        if pvalue > ALPHA:
            count_gt_alpha += 1

    count_le_alpha = N_ITER - count_gt_alpha
    return count_gt_05, count_gt_alpha, count_le_alpha

def counts_to_pct(counts, total):
    c_gt_05, c_gt_alpha, c_le_alpha = counts
    return 100.0 * c_gt_05 / total, 100.0 * c_gt_alpha / total, 100.0 * c_le_alpha / total

# promedio sobre un rango de p con un desplazamiento fijo --------------
def run_range_summary(ps, shift, N, n_args, N_ITER, sampler_1, sampler_2, ALPHA, BASE_SEED):
    sum_gt_05 = sum_gt_alpha = sum_le_alpha = 0
    for p in ps:
        c = run_single_comparison(N, p, p + shift, n_args, N_ITER, sampler_1, sampler_2, ALPHA, BASE_SEED)
        sum_gt_05    += c[0]
        sum_gt_alpha += c[1]
        sum_le_alpha += c[2]
    return counts_to_pct((sum_gt_05, sum_gt_alpha, sum_le_alpha), N_ITER * len(list(ps)))

# formato LaTeX --------------------------------------------------------

def format_row(p_label, pprime_label, pct_gt_05, pct_gt_alpha, pct_le_alpha):
    return (
        f"{p_label} & {pprime_label} & "
        f"{pct_gt_05:5.1f} & {pct_gt_alpha:5.1f} & {pct_le_alpha:5.1f} \\\\"
    )

# tabla 11: p' = p+1, 0 <= p <= 25 ------------------------------------

def run_tabla11(N=100, ps_beta22=(0,1,2,3,4,5,6), ps_beta05=(0,1,2,3),
                n_args=2, N_ITER=10000, ALPHA=0.05, BASE_SEED=321):

    ps_range = range(0, 26)
    results = {}

    # Beta(2,2) vs Beta(2,2)
    s1 = make_beta_sampler(2.0, 2.0, N, n_args)
    s2 = make_beta_sampler(2.0, 2.0, N, n_args)
    rows = []
    for p in tqdm(ps_beta22, desc="Tabla 11 Beta(2,2)"):
        c = run_single_comparison(N, p, p+1, n_args, N_ITER, s1, s2, ALPHA, BASE_SEED)
        rows.append(format_row(str(p), str(p+1), *counts_to_pct(c, N_ITER)))
    pct = run_range_summary(ps_range, 1, N, n_args, N_ITER, s1, s2, ALPHA, BASE_SEED)
    rows.append(format_row(r"$p$", r"$p+1$", *pct))
    results["beta_2_2"] = rows

    # Beta(0.5,0.5) vs Beta(0.5,0.5)
    s1 = make_beta_sampler(0.5, 0.5, N, n_args)
    s2 = make_beta_sampler(0.5, 0.5, N, n_args)
    rows = []
    for p in tqdm(ps_beta05, desc="Tabla 11 Beta(0.5,0.5)"):
        c = run_single_comparison(N, p, p+1, n_args, N_ITER, s1, s2, ALPHA, BASE_SEED)
        rows.append(format_row(str(p), str(p+1), *counts_to_pct(c, N_ITER)))
    pct = run_range_summary(ps_range, 1, N, n_args, N_ITER, s1, s2, ALPHA, BASE_SEED)
    rows.append(format_row(r"$p$", r"$p+1$", *pct))
    results["beta_0_5"] = rows

    # Beta(5,1) vs Beta(10,1): solo fila agregada
    pct = run_range_summary(ps_range, 1, N, n_args, N_ITER,
                            make_beta_sampler(5.0, 1.0, N, n_args),
                            make_beta_sampler(10.0, 1.0, N, n_args),
                            ALPHA, BASE_SEED)
    results["beta_5_1_10_1"] = [format_row(r"$p$", r"$p+1$", *pct)]

    # Beta(1,5) vs Beta(1,10): solo fila agregada
    pct = run_range_summary(ps_range, 1, N, n_args, N_ITER,
                            make_beta_sampler(1.0, 5.0, N, n_args),
                            make_beta_sampler(1.0, 10.0, N, n_args),
                            ALPHA, BASE_SEED)
    results["beta_1_5_1_10"] = [format_row(r"$p$", r"$p+1$", *pct)]

    return results

# tabla 12: p' = p+10, 25 < p <= 300 ----------------------------------
def run_tabla12(N=100, n_args=2, N_ITER=10000, ALPHA=0.05, BASE_SEED=321):
    ps_range = range(26, 301)
    results = {}

    distribuciones = [
        ("beta_5_1_10_1", 5.0, 1.0, 10.0, 1.0),
        ("beta_1_5_1_10", 1.0, 5.0, 1.0, 10.0),
        ("beta_2_2",      2.0, 2.0, 2.0, 2.0),
        ("beta_0_5",      0.5, 0.5, 0.5, 0.5),
    ]

    for key, a1, b1, a2, b2 in tqdm(distribuciones, desc="Tabla 12"):
        pct = run_range_summary(ps_range, 10, N, n_args, N_ITER,
                                make_beta_sampler(a1, b1, N, n_args),
                                make_beta_sampler(a2, b2, N, n_args),
                                ALPHA, BASE_SEED)
        results[key] = format_row(r"$p$", r"$p+10$", *pct)

    return results

# main -----------------------------------------------------------------
if __name__ == "__main__":
    tabla11 = run_tabla11()
    tabla12 = run_tabla12()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("% Tabla 11 - Beta(2,2) vs Beta(2,2)\n")
        for row in tabla11["beta_2_2"]:
            f.write(row + "\n")

        f.write("\n% Tabla 11 - Beta(0.5,0.5) vs Beta(0.5,0.5)\n")
        for row in tabla11["beta_0_5"]:
            f.write(row + "\n")

        f.write("\n% Tabla 11 - Beta(5,1) vs Beta(10,1)\n")
        for row in tabla11["beta_5_1_10_1"]:
            f.write(row + "\n")

        f.write("\n% Tabla 11 - Beta(1,5) vs Beta(1,10)\n")
        for row in tabla11["beta_1_5_1_10"]:
            f.write(row + "\n")

        f.write("\n% Tabla 12 - Beta(5,1) vs Beta(10,1)\n")
        f.write(tabla12["beta_5_1_10_1"] + "\n")

        f.write("\n% Tabla 12 - Beta(1,5) vs Beta(1,10)\n")
        f.write(tabla12["beta_1_5_1_10"] + "\n")

        f.write("\n% Tabla 12 - Beta(2,2) vs Beta(2,2)\n")
        f.write(tabla12["beta_2_2"] + "\n")

        f.write("\n% Tabla 12 - Beta(0.5,0.5) vs Beta(0.5,0.5)\n")
        f.write(tabla12["beta_0_5"] + "\n")

    print(f"Resultados guardados en: {OUTPUT_FILE}")