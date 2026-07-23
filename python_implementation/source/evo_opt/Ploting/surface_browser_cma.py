import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "2d_surface_data"


def load_surfaces(data_dir):
    ds = []
    for p in sorted(data_dir.glob("scan_*.npz")):
        d = np.load(p)
        ds.append({
            "a0s": d["a0s"], "a1s": d["a1s"], "Z": d["Z"],
            "center": d["center"], "grid_min": d["grid_min"],
            "shell": int(d["shell"]), "l": str(d["l"]), "N": int(d["N"]),
        })
    ds.sort(key=lambda r: (r["shell"], r["N"]))
    return ds


def load_trajs(data_dir):
    """CMA mean paths + lowest energy, keyed by internal (shell, N) — filenames may be scrambled."""
    tr = {}
    for p in sorted(data_dir.glob("traj_*.npz")):
        d = np.load(p)
        # e_start: true starting-point energy if the run saved it, else fall back
        # to the gen-0 best (earliest energy available in older trajectory files)
        e_start = float(d["e_start"]) if "e_start" in d.files else float(d["best_energy"][0])
        tr[(int(d["shell"]), int(d["N"]))] = {
            "mean":    d["mean"],            # (G, 2) mean per generation
            "e_start": e_start,              # energy at the starting point
            "e_final": float(d["e_final"]),  # lowest energy CMA found
        }
    return tr


datasets = load_surfaces(DATA_DIR)
trajs    = load_trajs(DATA_DIR)
if not datasets:
    raise SystemExit(f"No scan_*.npz found in {DATA_DIR}")

fig, ax = plt.subplots(figsize=(8.4, 7.6))
fig.subplots_adjust(bottom=0.16, right=0.86)
cax   = fig.add_axes([0.885, 0.16, 0.025, 0.74])
state = {"idx": 0}


def draw():
    d           = datasets[state["idx"]]
    a0s, a1s, Z = d["a0s"], d["a1s"], d["Z"]
    GA, GB      = np.meshgrid(a0s, a1s)

    ax.cla()
    cax.cla()

    lo     = float(np.nanmin(Z))
    hi     = float(np.nanmedian(Z))
    cf     = ax.contourf(GA, GB, Z, levels=np.linspace(lo, hi, 40), cmap="viridis", extend="max")
    fig.colorbar(cf, cax=cax, label="E (Eh)")

    ax.plot(d["center"][0], d["center"][1], "s", color="#00cd6c", markeredgecolor="black",
            markersize=10.0, label="grid center")
    ax.plot(d["grid_min"][0], d["grid_min"][1], "*", color="#ff3333", markeredgecolor="black",
            markersize=19.0, label=f"grid min ({lo:.6f})")

    # CMA mean at every generation (early -> late), overlaid on the surface
    tr = trajs.get((d["shell"], d["N"]))
    if tr is not None and len(tr["mean"]):
        mean    = tr["mean"]
        e_cma   = tr["e_final"]
        e_start = tr["e_start"]
        ax.plot(mean[:, 0], mean[:, 1], "-", color="white", lw=1.4, alpha=0.85, zorder=4)
        ax.scatter(mean[:, 0], mean[:, 1], c=np.arange(len(mean)), cmap="cool",
                   s=30.0, edgecolor="black", linewidths=0.5, zorder=5,
                   label=f"CMA mean ({len(mean)} gens, early→late)")
        ax.plot(mean[0, 0], mean[0, 1], "P", color="yellow", markeredgecolor="black",
                markersize=13.0, zorder=6, label=f"CMA start ({e_start:.6f})")
        ax.plot(mean[-1, 0], mean[-1, 1], "X", color="white", markeredgecolor="black",
                markersize=13.0, zorder=6, label=f"CMA min ({e_cma:.6f})")

    # keep the surface framed; clip any trajectory point that wanders outside
    ax.set_xlim(a0s.min(), a0s.max())
    ax.set_ylim(a1s.min(), a1s.max())
    ax.set_xlabel("a0  (log-scale param)")
    ax.set_ylabel("a1  (range param)")
    ax.set_title(f"[{state['idx'] + 1}/{len(datasets)}]   Shell {d['shell']} ({d['l']})   N={d['N']}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.canvas.draw_idle()


def go_next(event=None):
    state["idx"] = (state["idx"] + 1) % len(datasets)
    draw()


def go_prev(event=None):
    state["idx"] = (state["idx"] - 1) % len(datasets)
    draw()


def on_key(event):
    if event.key in ("right", "n"):
        go_next()
    elif event.key in ("left", "p"):
        go_prev()


b_prev = Button(fig.add_axes([0.30, 0.035, 0.15, 0.06]), "◀ Prev")
b_next = Button(fig.add_axes([0.55, 0.035, 0.15, 0.06]), "Next ▶")
b_prev.on_clicked(go_prev)
b_next.on_clicked(go_next)
fig.canvas.mpl_connect("key_press_event", on_key)

draw()
plt.show()
