import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE / "2d_surface_data"


def load_all(data_dir):
    """Load every .npz surface, sorted by shell (L) then N: all s by N, then p, then d ..."""
    datasets = []
    for p in sorted(data_dir.glob("*.npz")):
        d = np.load(p)
        datasets.append({
            "a0s": d["a0s"], "a1s": d["a1s"], "Z": d["Z"],
            "center": d["center"], "grid_min": d["grid_min"],
            "shell": int(d["shell"]), "l": str(d["l"]), "N": int(d["N"]),
        })
    datasets.sort(key=lambda r: (r["shell"], r["N"]))
    return datasets


datasets = load_all(DATA_DIR)
if not datasets:
    raise SystemExit(f"No .npz files found in {DATA_DIR}")

fig, ax = plt.subplots(figsize=(8.2, 7.4))
fig.subplots_adjust(bottom=0.16, right=0.86)
cax   = fig.add_axes([0.885, 0.16, 0.025, 0.74])   # dedicated colorbar axes
state = {"idx": 0}


def draw():
    d           = datasets[state["idx"]]
    a0s, a1s, Z = d["a0s"], d["a1s"], d["Z"]
    GA, GB      = np.meshgrid(a0s, a1s)

    ax.cla()
    cax.cla()

    # clip the colour range to [min, median] so the deep basin resolves and
    # exploded / high-energy points don't wash out the colormap
    lo     = float(np.nanmin(Z))
    hi     = float(np.nanmedian(Z))
    levels = np.linspace(lo, hi, 40)
    cf     = ax.contourf(GA, GB, Z, levels=levels, cmap="viridis", extend="max")
    fig.colorbar(cf, cax=cax, label="E (Eh)")

    ax.plot(d["center"][0], d["center"][1], "s", color="#00cd6c", markeredgecolor="black",
            markersize=11.0, label="grid center")
    ax.plot(d["grid_min"][0], d["grid_min"][1], "*", color="#ff3333", markeredgecolor="black",
            markersize=20.0, label=f"min  E={lo:.6f} Eh")

    ax.set_xlabel("a0  (log-scale param)")
    ax.set_ylabel("a1  (range param)")
    ax.set_title(f"[{state['idx'] + 1}/{len(datasets)}]   Shell {d['shell']} ({d['l']})   N={d['N']}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
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


# ── second figure: true minimum energy vs N, one curve per shell ──────────────

shell_ids   = sorted({d["shell"] for d in datasets})
shell_label = {d["shell"]: d["l"] for d in datasets}
# datasets are already (shell, N)-sorted, so each shell's points come out N-ordered
min_curves  = {s: [(d["N"], float(np.nanmin(d["Z"]))) for d in datasets if d["shell"] == s]
               for s in shell_ids}

fig_m, ax_m = plt.subplots(figsize=(8.2, 6.4))
fig_m.subplots_adjust(bottom=0.16)
state_m = {"idx": 0}


def draw_min():
    s   = shell_ids[state_m["idx"]]
    pts = min_curves[s]
    Ns  = np.array([p[0] for p in pts], dtype=float)
    Es  = np.array([p[1] for p in pts], dtype=float)

    ax_m.cla()
    ax_m.plot(Ns, Es, "o-", color="#1f77b4", markersize=7.0, label="grid minimum")
    for N, E in pts:
        ax_m.annotate(f"{E:.5f}", (N, E), textcoords="offset points", xytext=(0, 9),
                      fontsize=8, ha="center", color="#333333")

    # simple linear trend across the sizes
    if len(Ns) >= 2:
        slope, intercept = np.polyfit(Ns, Es, 1)
        xf = np.linspace(Ns.min(), Ns.max(), 100)
        ax_m.plot(xf, slope * xf + intercept, "--", color="#ff3333",
                  label=f"trend  {slope:+.3e} Eh / N")

    ax_m.set_xlabel("N (exponents)")
    ax_m.set_ylabel("true minimum E (Eh)")
    ax_m.set_title(f"[{state_m['idx'] + 1}/{len(shell_ids)}]   Shell {s} ({shell_label[s]}) — minimum energy vs N")
    ax_m.legend(loc="upper right", fontsize=9)
    ax_m.grid(True, alpha=0.25)
    fig_m.canvas.draw_idle()


def go_next_m(event=None):
    state_m["idx"] = (state_m["idx"] + 1) % len(shell_ids)
    draw_min()


def go_prev_m(event=None):
    state_m["idx"] = (state_m["idx"] - 1) % len(shell_ids)
    draw_min()


def on_key_m(event):
    if event.key in ("right", "n"):
        go_next_m()
    elif event.key in ("left", "p"):
        go_prev_m()


b_prev_m = Button(fig_m.add_axes([0.30, 0.035, 0.15, 0.06]), "◀ Prev")
b_next_m = Button(fig_m.add_axes([0.55, 0.035, 0.15, 0.06]), "Next ▶")
b_prev_m.on_clicked(go_prev_m)
b_next_m.on_clicked(go_next_m)
fig_m.canvas.mpl_connect("key_press_event", on_key_m)

draw_min()

plt.show()
