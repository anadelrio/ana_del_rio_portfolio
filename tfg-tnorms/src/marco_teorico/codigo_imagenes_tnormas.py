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

# malla (x,y) en [0,1]x[0,1] -----------------------------------------

N = 401
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
TP = X * Y  # t-norma producto, referencia para Delta

# definición de t-normas -----------------------------------------------

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

# familias a visualizar ------------------------------------------------

FAMILIES_ALL = {
    "Minimo":                  lambda X, Y: T_min(X, Y),
    "Producto":                lambda X, Y: T_prod(X, Y),
    "Lukasiewicz":             lambda X, Y: T_luk(X, Y),
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

COLORS = {
    "Minimo":                  "#1f77b4",
    "Producto":                "#ff7f0e",
    "Lukasiewicz":             "#2ca02c",
    "Frank s=0.15":            "#a55194",
    "Frank s=50":              "#d62728",
    "Yager p=10":              "#8c564b",
    "Yager p=1.2":             "#17becf",
    "Schweizer-Sklar lam=+2":  "#7f7f7f",
    "Schweizer-Sklar lam=-2":  "#e377c2",
    "Hamacher g=0.5":          "#969619",
    "Hamacher g=6":            "#9467bd",
    "Sugeno-Weber theta=-0.9": "#1f9e89",
    "Sugeno-Weber theta=12":   "#386cb0",
}

# leyenda --------------------------------------------------------------

def _legend_handles():
    return [Line2D([0], [0], color=COLORS[n], lw=2, label=n) for n in FAMILIES_ALL]

# imagen 1: contornos superpuestos -------------------------------------

def plot_contours_overlay(levels=(0.2, 0.5, 0.8)):
    style_for_level = {levels[0]: "-", levels[1]: "--", levels[2]: ":"}
    plt.figure(figsize=(8.2, 6.6), dpi=180)

    for name, F in FAMILIES_ALL.items():
        Z = F(X, Y)
        for c in levels:
            plt.contour(
                X, Y, Z,
                levels=[c],
                colors=[COLORS[name]],
                linestyles=style_for_level[c],
                linewidths=2.0 if name != "Lukasiewicz" else 2.4,
                zorder=5 if name == "Lukasiewicz" else 2,
            )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Contornos superpuestos T(x,y)=c")
    plt.grid(True, alpha=0.25)
    plt.legend(
        handles=_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        framealpha=0.95,
        title="Familias",
        fontsize=8,
    )

    out = os.path.join(BASE_DIR, "tnorms_contours_overlay.jpg")
    plt.tight_layout()
    plt.savefig(out, format="jpg", bbox_inches="tight")
    plt.close()
    return out

# imagen 2: contornos separados por nivel c ----------------------------

def plot_contours_by_c(c_values=(0.2, 0.5, 0.8)):
    outputs = []
    for c in c_values:
        plt.figure(figsize=(8.2, 6.6), dpi=180)

        for name, F in FAMILIES_ALL.items():
            Z = F(X, Y)
            plt.contour(
                X, Y, Z,
                levels=[c],
                colors=[COLORS[name]],
                linewidths=2.0 if name != "Lukasiewicz" else 2.4,
                zorder=5 if name == "Lukasiewicz" else 2,
            )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"Contornos superpuestos T(x,y) = {c}")
        plt.grid(True, alpha=0.25)
        plt.legend(
            handles=_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            framealpha=0.95,
            title="Familias",
            fontsize=8,
        )

        out = os.path.join(BASE_DIR, f"tnorms_contours_c{str(c).replace('.', '_')}.jpg")
        plt.tight_layout()
        plt.savefig(out, format="jpg", bbox_inches="tight")
        plt.close()
        outputs.append(out)

    return outputs

# imagen 3: grid 3D de Delta = T - x*y --------------------------------

def plot_delta_grid(cols=4):
    names = list(FAMILIES_ALL.keys())
    rows = int(np.ceil(len(names) / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows), dpi=180)
    fig.subplots_adjust(
        left=0.04, right=0.98, top=0.96, bottom=0.04,
        wspace=0.22, hspace=0.55,
    )

    for i, name in enumerate(names, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        Z = FAMILIES_ALL[name](X, Y)
        D = Z - TP
        m = max(float(np.max(np.abs(D))), 1e-6)

        ax.plot_surface(
            X, Y, D,
            cmap="coolwarm", vmin=-m, vmax=m,
            edgecolor="none", antialiased=True, alpha=0.98,
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(-m, m))
        ax.set_xlabel("x", fontsize=8, labelpad=4)
        ax.set_ylabel("y", fontsize=8, labelpad=4)
        ax.set_zlabel("Delta", fontsize=8, labelpad=2)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="z", labelsize=7)
        ax.view_init(elev=32, azim=-135)
        ax.set_title(f"{name}\n(T - xy)", fontsize=9, pad=6)

    out = os.path.join(BASE_DIR, "tnorms_delta_grid.jpg")
    fig.savefig(out, format="jpg")
    plt.close(fig)
    return out

# imagen 4: cortes 1D con y fijo --------------------------------------

def plot_1d_slices(y_values=(0.2, 0.5, 0.8)):
    outputs = []
    for y0 in y_values:
        plt.figure(figsize=(8.8, 4.8), dpi=170)

        for name, F in FAMILIES_ALL.items():
            Z = F(x[:, None], np.full((len(x), 1), y0))
            lw = 2.6 if name == "Lukasiewicz" else 1.8
            zo = 5  if name == "Lukasiewicz" else 2
            plt.plot(x, Z.ravel(), linewidth=lw, color=COLORS[name], zorder=zo)

        plt.xlabel("x")
        plt.ylabel(f"T(x,{y0})")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f"Cortes cuando y = {y0}")
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

        out = os.path.join(BASE_DIR, f"tnorms_slice_y{str(y0).replace('.', '_')}.jpg")
        plt.savefig(out, format="jpg", bbox_inches="tight")
        plt.close()
        outputs.append(out)

    return outputs

# main -----------------------------------------------------------------

if __name__ == "__main__":
    tareas = [
        "Contornos superpuestos",
        "Contornos por nivel",
        "Grid 3D Delta",
        "Cortes 1D",
    ]

    with tqdm(total=len(tareas), desc="Generando imagenes") as pbar:
        plot_contours_overlay()
        pbar.update(1)

        plot_contours_by_c()
        pbar.update(1)

        plot_delta_grid()
        pbar.update(1)

        plot_1d_slices()
        pbar.update(1)

    print(f"Imagenes guardadas en: {BASE_DIR}")