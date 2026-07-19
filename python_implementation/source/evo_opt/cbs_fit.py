from pathlib import Path
from collections.abc import Sequence
from numpy import array, log, float64, polyfit

Point  = tuple[int, float]
Window = tuple[int, int, int, float, float, float]


def fit_decay(n: Sequence[float], magnitude: Sequence[float], power: float = 0.5) -> tuple[float, float, float] | None:
    n_arr = array(n,         dtype=float64)
    mag   = array(magnitude, dtype=float64)
    if len(n_arr) < 2:
        return None

    x    = n_arr ** power
    y    = log(mag)
    b, a = polyfit(x, y, 1)
    c    = -float(b)

    yhat   = a + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return c, float(a), r2


def parse_cbs_csv(path: Path | str) -> dict[int, dict]:
    path   = Path(path)
    shells = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                s   = int(parts[0])
                n   = int(parts[2])
                e_f = float(parts[4])
            except (ValueError, IndexError):
                continue
            lbl = parts[1]
            if s not in shells:
                shells[s] = {"l": lbl, "points": []}
            shells[s]["points"].append((n, e_f))
    for s in shells:
        shells[s]["points"].sort(key=lambda p: p[0])
    return shells


# variational energies can only ever be *too high* (a failed optimisation), never
# too low, so the reliable sequence is the running lower envelope: keep a point
# only if it beats every energy seen so far; anything that doesn't is a failure.
def filter_monotonic(points: list[Point]) -> tuple[list[Point], list[Point]]:
    kept    = []
    dropped = []
    best    = None
    for i in range(len(points)):
        n, e = points[i]
        if best is None or e < best:
            kept.append((n, e))
            best = e
        else:
            dropped.append((n, e))
    return kept, dropped


def build_diffs(kept: list[Point]) -> list[Point]:
    diffs = []
    for i in range(1, len(kept)):
        n_hi, e_hi = kept[i]
        e_lo       = kept[i - 1][1]
        diffs.append((n_hi, e_lo - e_hi))
    return diffs


def analyze_shell(points: list[Point], min_points: int = 3, power: float = 0.5) -> tuple[dict, str]:
    kept, dropped = filter_monotonic(points)

    log_lines = []
    for i in range(len(dropped)):
        n, e = dropped[i]
        log_lines.append(f"    dropped N={n:3d}  E={e:.10f}  (non-monotonic, no improvement)")

    diffs = build_diffs(kept)
    ns    = [diffs[i][0] for i in range(len(diffs))]
    mags  = [diffs[i][1] for i in range(len(diffs))]

    windows = []
    cs      = []
    total   = len(diffs)
    for start in range(0, max(1, total - min_points + 1)):
        fit = fit_decay(ns[start:], mags[start:], power=power)
        if fit is None:
            continue
        c, a, r2 = fit
        cs.append(c)
        windows.append((ns[start], ns[-1], total - start, c, a, r2))

    result = {
        "l":       None,
        "kept":    kept,
        "dropped": dropped,
        "diffs":   diffs,
        "windows": windows,
        "c":       None,
        "c_std":   None,
        "c_min":   None,
        "c_max":   None,
    }
    if len(cs) > 0:
        cs_arr           = array(cs, dtype=float64)
        result["c"]      = float(cs_arr.mean())
        result["c_std"]  = float(cs_arr.std())
        result["c_min"]  = float(cs_arr.min())
        result["c_max"]  = float(cs_arr.max())

    return result, "\n".join(log_lines)


def analyze_csv(path: Path | str, min_points: int = 3, power: float = 0.5) -> tuple[dict[int, dict], str]:
    shells  = parse_cbs_csv(path)
    results = {}
    logs    = []
    for s in sorted(shells.keys()):
        lbl           = shells[s]["l"]
        points        = shells[s]["points"]
        res, drop_log = analyze_shell(points, min_points=min_points, power=power)
        res["l"]      = lbl
        results[s]    = res

        logs.append(f"Shell {s} ({lbl}):  {len(points)} pts -> {len(res['kept'])} kept, {len(res['dropped'])} dropped")
        if drop_log:
            logs.append(drop_log)
        if res["c"] is not None:
            logs.append(
                f"    c = {res['c']:.4f} +/- {res['c_std']:.4f}   "
                f"(range {res['c_min']:.4f}..{res['c_max']:.4f}, {len(res['windows'])} windows)"
            )
        else:
            logs.append("    c = <insufficient points>")
    return results, "\n".join(logs)
