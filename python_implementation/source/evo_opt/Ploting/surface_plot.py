import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_surface(npz_path):
    npz_path = Path(npz_path)
    data     = np.load(npz_path)

    a0s      = data["a0s"]
    a1s      = data["a1s"]
    Z        = data["Z"]
    center   = data["center"]
    grid_min = data["grid_min"]
    shell    = int(data["shell"])
    lbl      = str(data["l"])
    N        = int(data["N"])

    GA, GB = np.meshgrid(a0s, a1s)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    # clip the colour range to [min, median] so the deep basin is resolved and
    # high-energy / exploded points don't wash out the whole colormap
    lo     = float(np.nanmin(Z))
    hi     = float(np.nanmedian(Z))
    levels = np.linspace(lo, hi, 40)
    cf     = ax.contourf(GA, GB, Z, levels=levels, cmap="viridis", extend="max")
    fig.colorbar(cf, ax=ax, label="E (Eh)")

    ax.plot(center[0], center[1], "s", color="#00cd6c", markeredgecolor="black",
            markersize=11.0, label="grid center")
    ax.plot(grid_min[0], grid_min[1], "*", color="#ff3333", markeredgecolor="black",
            markersize=20.0, label=f"min  E={float(np.nanmin(Z)):.6f} Eh")

    ax.set_xlabel("a0  (log-scale param)")
    ax.set_ylabel("a1  (range param)")
    ax.set_title(f"Shell {shell} ({lbl})   N={N}   tempering energy surface")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


# ---- file to plot (looked up next to this script) ----
HERE     = Path(__file__).resolve().parent
NPZ_FILE = "scan_shell0_N17.npz"   # just the filename; keep the .npz beside this script

plot_surface(HERE / NPZ_FILE)
plt.show()
