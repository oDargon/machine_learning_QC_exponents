import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


EXPLOSION_THRESHOLD = 1e5


def _parse_meta_label(meta_line):
    """Extract a short label from the # metadata comment line."""
    parts = {}
    for token in meta_line.lstrip("# ").split():
        if "=" in token:
            k, v = token.split("=", 1)
            parts[k] = v
    gen   = parts.get("generator", "?")
    m     = parts.get("M", "?")
    shell = parts.get("shell", "?")
    contr = parts.get("contraction", "?")
    return f"{gen} M={m} shell={shell} contr={contr}"


def load_cbs_csv(path):
    path = Path(path)
    ns, energies = [], []
    meta_label = path.stem
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            raw = row[0].strip()
            if raw.startswith("#"):
                meta_label = _parse_meta_label(" ".join(row))
                continue
            if raw.lower() == "n":
                continue
            try:
                n = int(raw)
                e = float(row[1])
            except (ValueError, IndexError):
                continue
            if np.isnan(e) or abs(e) > EXPLOSION_THRESHOLD:
                continue
            ns.append(n)
            energies.append(e)
    return np.array(ns, dtype=int), np.array(energies, dtype=float), meta_label


def plot_cbs(csv_paths, labels=None, colors=None):
    datasets = [load_cbs_csv(p) for p in csv_paths]

    if labels is None:
        labels = [meta for _, _, meta in datasets]

    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        colors = [cycle[i % len(cycle)] for i in range(len(csv_paths))] if cycle else [None] * len(csv_paths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for (ns, es, *_), lab, col in zip(datasets, labels, colors):
        if len(ns) == 0:
            print(f"Warning: no valid data in {lab!r}, skipping")
            continue

        ax1.plot(ns, es, marker="o", ms=5, lw=1.8, color=col, label=lab)

        if len(ns) > 1:
            dE    = np.abs(np.diff(es))
            valid = dE > 0
            if valid.any():
                ax2.semilogy(ns[1:][valid], dE[valid], marker="o", ms=5, lw=1.8, color=col, label=lab)

    ax1.set_xlabel("N primitives")
    ax1.set_ylabel("Energy (Eh)")
    ax1.set_title("CBS convergence — linear")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("N primitives")
    ax2.set_ylabel("|ΔE| (Eh)")
    ax2.set_title("CBS convergence — log |ΔE|")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, (ax1, ax2)


# ---- paths to plot ----
CSV_PATHS = [
    r"C:\Users\DzJas\Desktop\Code_Prjcts\PHD_work\machine_learning_QC_exponents\python_implementation\Source\evo_opt\Ploting\full.csv",
    r"C:\Users\DzJas\Desktop\Code_Prjcts\PHD_work\machine_learning_QC_exponents\python_implementation\Source\evo_opt\Ploting\cont.csv",
]

plot_cbs(CSV_PATHS)
plt.show()
