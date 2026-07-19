from pathlib import Path
from numpy import log, sqrt
from evo_opt.cbs_fit import analyze_csv

HERE     = Path(__file__).resolve().parent
CSV_PATH = HERE / "data.csv"
LOG_PATH = HERE / "cbs_fit_test.log"

meta = ""
with open(CSV_PATH) as f:
    for raw in f:
        if raw.startswith("#"):
            meta = raw.strip()
            break

results, _ = analyze_csv(CSV_PATH)

lines = []
lines.append("=" * 70)
lines.append("CBS decay-rate fit report")
lines.append(f"source : {CSV_PATH.name}")
lines.append(f"meta   : {meta}")
lines.append("=" * 70)

for s in sorted(results.keys()):
    res     = results[s]
    kept    = res["kept"]
    dropped = res["dropped"]
    diffs   = res["diffs"]
    windows = res["windows"]

    n_lo = kept[0][0]  if kept else 0
    n_hi = kept[-1][0] if kept else 0

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"Shell {s} ({res['l']})   N={n_lo}..{n_hi}   "
                 f"{len(kept) + len(dropped)} pts   {len(kept)} kept   {len(dropped)} dropped")
    lines.append("-" * 70)

    lines.append("  increments  (N, |dE|, ln|dE|, sqrtN):")
    prev_mag = None
    for i in range(len(diffs)):
        n, mag = diffs[i]
        flag   = ""
        if prev_mag is not None and mag > prev_mag:
            flag = "   <-- increment grew (sub-converged / noise floor?)"
        lines.append(f"     N={n:3d}  |dE|={mag:.4e}  ln={log(mag):+8.4f}  sqrtN={sqrt(n):.4f}{flag}")
        prev_mag = mag

    if dropped:
        lines.append("  dropped (non-monotonic energy):")
        for i in range(len(dropped)):
            n, e = dropped[i]
            lines.append(f"     N={n:3d}  E={e:.10f}")
    else:
        lines.append("  dropped: none")

    lines.append("  tail windows  (drop low-N, refit):")
    best_i  = None
    best_r2 = -1.0
    for i in range(len(windows)):
        n0, n1, npts, c, ln_a, r2 = windows[i]
        flag = "   (low R2)" if r2 < 0.9 else ""
        lines.append(f"     N {n0:2d}..{n1:2d}  npts={npts:2d}  c={c:8.4f}  lnA={ln_a:9.4f}  R2={r2:.4f}{flag}")
        if r2 > best_r2:
            best_r2 = r2
            best_i  = i

    lines.append("  summary:")
    if res["c"] is not None:
        lines.append(f"     c = {res['c']:.4f} +/- {res['c_std']:.4f}   "
                     f"range [{res['c_min']:.4f}, {res['c_max']:.4f}]   over {len(windows)} windows")
        if best_i is not None:
            bw = windows[best_i]
            lines.append(f"     best-R2 window: N {bw[0]}..{bw[1]}  c={bw[3]:.4f}  R2={bw[5]:.4f}")
    else:
        lines.append("     c = <insufficient points>")

report = "\n".join(lines)
print(report)
LOG_PATH.write_text(report + "\n")
print(f"\nlog saved to {LOG_PATH}")
