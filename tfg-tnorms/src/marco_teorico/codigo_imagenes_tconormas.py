import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

# ruta de salida -------------------------------------------------------

try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# malla (x,y) en [0,1]x[0,1] ----------------------------------------------

N = 401
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# definición de t-normas base ------------------------------------------

def T_min(X, Y):
    return np.minimum(X, Y)

def T_prod(X, Y):
    return X * Y

def T_luk(X, Y):
    return np.maximum(X + Y - 1.0, 0.0)

def T_frank(X, Y, s):
    if s <= 0 or np.isclose(s, 1.0):
        raise ValueError("Frank: s > 0 y s != 1")
    a = np.power(s, X) - 1.0
    b = np.power(s, Y) - 1.0
    num = np.maximum(1.0 + (a * b) / (s - 1.0), 1e-16)
    return np.log(num) / np.log(s)

def T_yager(X, Y, p):
    r = ((1 - X) ** p + (1 - Y) ** p) ** (1.0 / p)
    return 1.0 - np.minimum(1.0, r)

def T_ss(X, Y, lam):
    if np.isclose(lam, 0.0):
        return X * Y
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.maximum(0.0, X ** (-lam) + Y ** (-lam) - 1.0)
    return np.power(inner, -1.0 / lam, where=inner > 0, out=np.zeros_like(inner))

def T_ham(X, Y, g):
    denom = g + (1 - g) * (X + Y - X * Y)
    denom = np.where(denom == 0, 1e-16, denom)
    return (X * Y) / denom

def T_sw(X, Y, theta):
    return np.maximum(0.0, (X + Y - 1.0 + theta * X * Y) / (1.0 + theta))

def T_drastica(X, Y):
    return np.where(
        np.isclose(X, 1.0) | np.isclose(Y, 1.0),
        np.minimum(X, Y),
        0.0,
    )

# t-normas base -------------------------------------------------------------

TNORM_FAMILIES_ALL = {
    "Minimo":                  lambda X, Y: T_min(X, Y),
    "Producto":                lambda X, Y: T_prod(X, Y),
    "Lukasiewicz":             lambda X, Y: T_luk(X, Y),
    "Drastica":                lambda X, Y: T_drastica(X, Y),
    "Frank s=0.15":            lambda X, Y: T_frank(X, Y, s=0.15),
    "Frank s=50":              lambda X, Y: T_frank(X, Y, s=50.0),
    "Yager p=10":              lambda X, Y: T_yager(X, Y, p=10.0),
    "Yager p=1.2":             lambda X, Y: T_yager(X, Y, p=1.2),
    "Schweizer-Sklar lam=+2":  lambda X, Y: T_ss(X, Y, lam=+2.0),
    "Schweizer-Sklar lam=-2":  lambda X, Y: T_ss(X, Y, lam=-2.0),
    "Hamacher g=0.5":          lambda X, Y: T_ham(X, Y, g=0.5),
    "Hamacher g=6":            lambda X, Y: T_ham(X, Y, g=6.0),
    "Sugeno-Weber theta=-0.9": lambda X, Y: T_sw(X, Y, theta=-0.9),
    "Sugeno-Weber theta=12":   lambda X, Y: T_sw(X, Y, theta=12.0),
}

# colores por familia --------------------------------------------------

COLORS_S = {
    "Maximo":                  "#1f77b4",
    "Suma probabilistica":     "#ff7f0e",
    "Lukasiewicz":             "#2ca02c",
    "Drastica":                "#FFFC32",
    "Frank s=0.15":            "#a55194",
    "Frank s=50":              "#d62728",
    "Yager p=10":              "#8c564b",
    "Yager p=1.2":             "#17becf",
    "Schweizer-Sklar lam=+2":  "#7f7f7f",
    "Schweizer-Sklar lam=-2":  "#e377c2",
    "Hamacher g=0.5":          "#bcbd22",
    "Hamacher g=6":            "#9467bd",
    "Sugeno-Weber theta=-0.9": "#1f9e89",
    "Sugeno-Weber theta=12":   "#386cb0",
}

# construcción de t-conormas por dualidad: S(x,y) = 1 - T(1-x, 1-y) --

def dual_from_T(F):
    return lambda X, Y: 1.0 - F(1.0 - X, 1.0 - Y)

# familias de t-conormas -----------------------------------------------

FAMILIES_ALL = {
    "Maximo":                  dual_from_T(TNORM_FAMILIES_ALL["Minimo"]),
    "Suma probabilistica":     dual_from_T(TNORM_FAMILIES_ALL["Producto"]),
    "Lukasiewicz":             dual_from_T(TNORM_FAMILIES_ALL["Lukasiewicz"]),
    "Drastica":                dual_from_T(TNORM_FAMILIES_ALL["Drastica"]),
    "Frank s=0.15":            dual_from_T(TNORM_FAMILIES_ALL["Frank s=0.15"]),
    "Frank s=50":              dual_from_T(TNORM_FAMILIES_ALL["Frank s=50"]),
    "Yager p=10":              dual_from_T(TNORM_FAMILIES_ALL["Yager p=10"]),
    "Yager p=1.2":             dual_from_T(TNORM_FAMILIES_ALL["Yager p=1.2"]),
    "Schweizer-Sklar lam=+2":  dual_from_T(TNORM_FAMILIES_ALL["Schweizer-Sklar lam=+2"]),
    "Schweizer-Sklar lam=-2":  dual_from_T(TNORM_FAMILIES_ALL["Schweizer-Sklar lam=-2"]),
    "Hamacher g=0.5":          dual_from_T(TNORM_FAMILIES_ALL["Hamacher g=0.5"]),
    "Hamacher g=6":            dual_from_T(TNORM_FAMILIES_ALL["Hamacher g=6"]),
    "Sugeno-Weber theta=-0.9": dual_from_T(TNORM_FAMILIES_ALL["Sugeno-Weber theta=-0.9"]),
    "Sugeno-Weber theta=12":   dual_from_T(TNORM_FAMILIES_ALL["Sugeno-Weber theta=12"]),
}

# leyenda --------------------------------------------------------------

def _legend_handles():
    return [Line2D([0], [0], color=COLORS_S[n], lw=2, label=n) for n in FAMILIES_ALL]

# imagen 1: contornos superpuestos -------------------------------------------

def plot_contours_overlay(levels=(0.2, 0.5, 0.8)):
    style_for_level = {levels[0]: "-", levels[1]: "--", levels[2]: ":"}
    plt.figure(figsize=(8.2, 6.6), dpi=180)

    for name, F in FAMILIES_ALL.items():
        Z = F(X, Y)
        for c in levels:
            plt.contour(
                X, Y, Z,
                levels=[c],
                colors=[COLORS_S[name]],
                linestyles=style_for_level[c],
                linewidths=2.0 if name != "Lukasiewicz" else 2.4,
                zorder=5 if name == "Lukasiewicz" else 2,
            )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Contornos superpuestos S(x,y)=c")
    plt.grid(True, alpha=0.25)
    plt.legend(
        handles=_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        framealpha=0.95,
        title="Familias",
        fontsize=8,
    )

    out = os.path.join(BASE_DIR, "snorms_contours_overlay.jpg")
    plt.tight_layout()
    plt.savefig(out, format="jpg", bbox_inches="tight")
    plt.close()
    return out

# imagen 2: cortes 1D con y fijo ---------------------------------------

def plot_1d_slices(y_values=(0.2, 0.5, 0.8)):
    outputs = []
    for y0 in y_values:
        plt.figure(figsize=(8.8, 4.8), dpi=170)

        for name, F in FAMILIES_ALL.items():
            Z = F(x[:, None], np.full((len(x), 1), y0))
            lw = 2.6 if name == "Lukasiewicz" else 1.8
            zo = 5  if name == "Lukasiewicz" else 2
            plt.plot(x, Z.ravel(), linewidth=lw, color=COLORS_S[name], zorder=zo)

        plt.xlabel("x")
        plt.ylabel(f"S(x,{y0})")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"Cortes de t-conormas cuando y = {y0}")
        plt.grid(True, alpha=0.3)
        plt.legend(
            handles=_legend_handles(),
            fontsize=8,
            ncol=1,
            framealpha=0.95,
            title="Familias",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

        out = os.path.join(BASE_DIR, f"snorms_slice_y{str(y0).replace('.', '_')}.jpg")
        plt.savefig(out, format="jpg", bbox_inches="tight")
        plt.close()
        outputs.append(out)

    return outputs

# main -----------------------------------------------------------------

if __name__ == "__main__":
    tareas = [
        "Contornos superpuestos",
        "Cortes 1D",
    ]

    with tqdm(total=len(tareas), desc="Generando imagenes") as pbar:
        plot_contours_overlay()
        pbar.update(1)

        plot_1d_slices()
        pbar.update(1)

    print(f"Imagenes guardadas en: {BASE_DIR}")