import csv
import argparse
from pathlib import Path

_arg_parser = argparse.ArgumentParser(description="Jobs-to-target analysis of a thirty_second.py run: how much of each arm's budget bought the last sliver of energy")
_arg_parser.add_argument("--results-dir", type=Path, default=Path.cwd() / "results")
_args = _arg_parser.parse_args()

RESULTS_DIR = _args.results_dir.resolve()
TRACE_CSV   = RESULTS_DIR / "cma_vs_bfgs_trace.csv"
SUMMARY_CSV = RESULTS_DIR / "cma_vs_bfgs_summary.csv"

# fractions of each arm's OWN total gain, so neither is judged against the other's scale
FRACTIONS = [0.50, 0.90, 0.99, 0.999, 1.0]

if not TRACE_CSV.exists():
    raise SystemExit(f"no trace at {TRACE_CSV}")

# ─── load ─────────────────────────────────────────────────────────────────────

trace = {}          # (shell, method) -> list of (jobs, wall, best_E), in order
labels = {}
with open(TRACE_CSV) as f:
    for row in csv.DictReader(f):
        key = (int(row["shell"]), row["method"])
        trace.setdefault(key, []).append(
            (int(row["molcas_jobs"]), float(row["wall_sec"]), float(row["best_E"]))
        )
        labels[int(row["shell"])] = row["l"]

e_start = {}
if SUMMARY_CSV.exists():
    with open(SUMMARY_CSV) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#META") or row[0] == "shell":
                continue
            e_start[int(row[0])] = float(row[4])

shells  = sorted(labels)
methods = ["CMA-ES", "BFGS"]


def first_at_or_below(points, target):
    for i in range(len(points)):
        if points[i][2] <= target:
            return points[i]
    return None


# ─── per shell ────────────────────────────────────────────────────────────────

for shell in shells:
    lbl   = labels[shell]
    have  = [m for m in methods if (shell, m) in trace]
    if not have:
        continue

    finals = {}
    for i in range(len(have)):
        finals[have[i]] = trace[(shell, have[i])][-1]

    start = e_start.get(shell)
    if start is None:
        start = max(trace[(shell, have[i])][0][2] for i in range(len(have)))

    print("=" * 86)
    print(f"shell {shell} ({lbl})   E_start = {start:.10f} Eh")
    print("=" * 86)

    for i in range(len(have)):
        m    = have[i]
        jobs, wall, e = finals[m]
        print(f"  {m:<8} final: E {e:.10f}  gain {e - start:+.3e} Eh  "
              f"{jobs} jobs  {wall:.0f}s")

    # ── the question: when did each arm first reach the OTHER's final energy ──
    print()
    for i in range(len(have)):
        for j in range(len(have)):
            if i == j:
                continue
            mine, theirs = have[i], have[j]
            target       = finals[theirs][2]
            hit          = first_at_or_below(trace[(shell, mine)], target)
            mine_jobs    = finals[mine][0]
            mine_wall    = finals[mine][1]
            if hit is None:
                print(f"  {mine} never reached {theirs}'s final E ({target:.10f})")
            else:
                print(f"  {mine} reached {theirs}'s final E ({target:.10f}) at "
                      f"{hit[0]} jobs / {hit[1]:.0f}s  "
                      f"-> the remaining {mine_jobs - hit[0]} jobs "
                      f"({(mine_jobs - hit[0]) / mine_jobs:.0%} of its budget, "
                      f"{mine_wall - hit[1]:.0f}s) bought {abs(finals[mine][2] - target):.2e} Eh")

    # ── jobs to reach a fraction of each arm's own gain ──
    print()
    print(f"  {'target':<34}" + "".join(f"{m:>24}" for m in have))
    for f_i in range(len(FRACTIONS)):
        frac  = FRACTIONS[f_i]
        cells = ""
        for i in range(len(have)):
            m      = have[i]
            gain   = finals[m][2] - start
            target = start + frac * gain
            hit    = first_at_or_below(trace[(shell, m)], target)
            cells += f"{(str(hit[0]) + ' jobs / ' + format(hit[1], '.0f') + 's') if hit else '—':>24}"
        print(f"  {frac * 100:>6.1f}% of own gain ({'':<10}" + ")" + cells)
    print()

# ─── headline ─────────────────────────────────────────────────────────────────

print("=" * 86)
print("If an arm reached the other's final energy well before spending its budget, the")
print("gap between the arms is its STOPPING RULE, not its search. Compare the 99% and")
print("100% rows: the jobs between them are what the last sliver of energy cost.")
print("=" * 86)
