from numpy import log10, zeros_like, linspace, exp, concatenate, asarray, float64, pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

L_LABELS = ["s", "p", "d", "f", "g", "h", "i"]


def _kde_log(exponents, grid, bandwidth):
    x = log10(asarray(exponents, dtype=float64))
    d = zeros_like(grid)
    for i in range(len(x)):
        d += exp(-0.5 * ((grid - x[i]) / bandwidth) ** 2)
    return d / (len(x) * bandwidth * sqrt(2.0 * pi))


def plot_shell_densities(
    exp_sets,
    names=None,
    bandwidth=0.35,
    n_grid=400,
    pad=0.6,
    fill=True,
):
    if hasattr(exp_sets, "exponents"):
        exp_sets = [exp_sets]
    if isinstance(names, str):
        names = [names]
    if names is None:
        names = [f"set {i}" for i in range(len(exp_sets))]

    n_shells = min(len(es.exponents) for es in exp_sets)
    if n_shells == 0:
        raise ValueError("Exponent sets have no shells.")

    cycle  = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = [cycle[i % len(cycle)] for i in range(len(exp_sets))] if cycle else [None] * len(exp_sets)

    _state = {"shell": 0}

    fig     = plt.figure(figsize=(10, 5))
    ax      = fig.add_axes([0.09, 0.18, 0.87, 0.74])
    ax_prev = fig.add_axes([0.35, 0.03, 0.13, 0.08])
    ax_next = fig.add_axes([0.52, 0.03, 0.13, 0.08])
    btn_prev = Button(ax_prev, "← Prev")
    btn_next = Button(ax_next, "Next →")

    def _draw():
        ax.clear()
        sh  = _state["shell"]
        lbl = L_LABELS[sh] if sh < len(L_LABELS) else str(sh)

        arrays  = []
        clabels = []
        cols    = []
        for i in range(len(exp_sets)):
            arr = asarray(exp_sets[i].exponents[sh], dtype=float64)
            if len(arr) >= 2 and (arr > 0).all():
                arrays.append(arr)
                clabels.append(names[i])
                cols.append(colors[i])

        ax.set_title(f"Shell {sh}  ({lbl})")

        if not arrays:
            ax.text(0.5, 0.5, f"No valid exponents for {lbl}-shell",
                    ha="center", va="center", transform=ax.transAxes)
            fig.canvas.draw_idle()
            return

        all_log = concatenate([log10(a) for a in arrays])
        lo      = all_log.min() - pad
        hi      = all_log.max() + pad
        grid    = linspace(lo, hi, n_grid)

        for i in range(len(arrays)):
            dens     = _kde_log(arrays[i], grid, bandwidth)
            mean_log = log10(arrays[i]).mean()
            lab      = f"{clabels[i]}  (μ log₁₀ζ = {mean_log:.2f})"
            line,    = ax.plot(grid, dens, lw=2.2, color=cols[i], label=lab)
            if fill:
                ax.fill_between(grid, dens, color=line.get_color(), alpha=0.10)
            ax.scatter(
                log10(arrays[i]),
                zeros_like(arrays[i]),
                color=line.get_color(),
                s=40, zorder=5, clip_on=False,
            )

        ax.set_xlabel("log₁₀(ζ)   diffuse ←  → tight")
        ax.set_ylabel("density")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        fig.canvas.draw_idle()

    def _on_prev(event):
        _state["shell"] = (_state["shell"] - 1) % n_shells
        _draw()

    def _on_next(event):
        _state["shell"] = (_state["shell"] + 1) % n_shells
        _draw()

    btn_prev.on_clicked(_on_prev)
    btn_next.on_clicked(_on_next)

    _draw()
    plt.show()


