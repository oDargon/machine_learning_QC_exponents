import sys
from pathlib import Path
from numpy import array, linspace, meshgrid, column_stack, log10, logspace, float64

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SUBMIT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.newton_6 import Newton_6


def quadratic(P):
    x = P[:, 0]
    y = P[:, 1]
    return 1000.0 * (x - 3.0) ** 2 + 1.0 * (y + 2.0) ** 2


def curved(P):
    x = P[:, 0]
    y = P[:, 1]
    return 50.0 * (x - 1.2) ** 2 + 3.0 * (y + 0.5) ** 2 + 4.0 * (x - 1.2) * (y + 0.5) + 2.0 * (x - 1.2) ** 4


def himmelblau(P):
    x = P[:, 0]
    y = P[:, 1]
    return (x ** 2 + y - 11.0) ** 2 + (x + y ** 2 - 7.0) ** 2


def rosenbrock(P):
    x = P[:, 0]
    y = P[:, 1]
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


himmelblau_minima = array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])

cases = [
    ("Ill-conditioned quadratic", quadratic,   array([3.0, -2.0]),  array([0.0, 0.0]),  (-1.0, 4.0), (-3.5, 1.5), 1.0, 20),
    ("Curved (≈ energy surface)", curved,      array([1.2, -0.5]),  array([0.0, 0.0]),  (-0.6, 2.2), (-1.6, 0.7), 0.3, 25),
    ("Himmelblau (4 minima)",     himmelblau,  himmelblau_minima,   array([0.0, 0.0]),  (-5.0, 5.0), (-5.0, 5.0), 0.3, 40),
    ("Rosenbrock (k=100)",        rosenbrock,  array([1.0, 1.0]),   array([-1.2, 1.0]), (-1.6, 1.6), (-0.6, 1.7), 0.3, 80),
]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))

axes = axes.ravel()

for i in range(len(cases)):
    title, func, true_min, start, xlim, ylim, trust, max_steps = cases[i]
    ax = axes[i]

    opt      = Newton_6(func, start, trust_radius=trust)
    found, _ = opt.minimize(max_steps=max_steps)
    path     = array([p for p, _ in opt.history], dtype=float64)

    xs = linspace(xlim[0], xlim[1], 400)
    ys = linspace(ylim[0], ylim[1], 400)
    gx, gy = meshgrid(xs, ys)
    Z = func(column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    zfloor = max(float(Z.min()), 1e-8)
    levels = logspace(log10(zfloor + 1e-9), log10(float(Z.max())), 30)
    ax.contourf(gx, gy, Z + 1e-9, levels=levels, norm=LogNorm(), cmap="viridis")

    ax.plot(path[:, 0], path[:, 1], "-o", color="white", markeredgecolor="black",
            markeredgewidth=0.8, linewidth=2.0, markersize=6.0, label="accepted path")
    ax.plot(start[0], start[1], "s", color="#00cd6c", markeredgecolor="black",
            markersize=12.0, label="start")
    minima = array(true_min, dtype=float64).reshape(-1, 2)
    ax.plot(minima[:, 0], minima[:, 1], "*", color="#ff3333", markeredgecolor="black",
            markersize=20.0, linestyle="none", label="true minima")
    ax.plot(found[0], found[1], "X", color="black", markeredgecolor="white",
            markersize=13.0, label="found minimum")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"{title}\n{len(opt.history) - 1} accepted, {opt.reject_count} rejected, "
                 f"{opt.refit_count} refits, {opt.eval_count} evals")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig.suptitle("Newton_6 trust-region optimizer — path over function contours", fontsize=14)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

plt.show()
