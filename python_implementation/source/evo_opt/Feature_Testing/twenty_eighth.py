import csv
import argparse
from pathlib import Path

from numpy import array, float64

from evo_opt.common import L_LABELS, CHEMICAL_ACCURACY
from evo_opt.exponent_handler import Exponent_Set
from evo_opt.tempering import from_registry
from evo_opt.cbs_engine import _cbs_fit, _cbs_target_n, _cbs_predict, _extrapolate_start

_arg_parser = argparse.ArgumentParser(description="From a CMA-minima results CSV: fit E_inf per shell and report the minimal N to reach each shell's energy tolerance")
_arg_parser.add_argument("--results", type=Path, required=True,
                         help="cma_minima CSV (ordered or unordered/partial — rows are re-sorted by N per shell)")
_args = _arg_parser.parse_args()

RESULTS_CSV = _args.results.resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

# one energy tolerance (Hartree) PER SHELL — how close to that shell's CBS limit it
# must get. Shells present in the data but not listed here use DEFAULT_TOLERANCE.
SHELL_TOLERANCES = {
    0: CHEMICAL_ACCURACY,   # s
    1: CHEMICAL_ACCURACY,   # p
    2: CHEMICAL_ACCURACY,   # d
    3: CHEMICAL_ACCURACY,   # f
    4: CHEMICAL_ACCURACY,   # g
}
DEFAULT_TOLERANCE = CHEMICAL_ACCURACY
CBS_MIN_POINTS    = 3        # smallest tail window used when gauging the E_inf band

# also report an OPTIMISTIC basis size: relax every shell's tolerance by this factor
# (e.g. 1e-4 -> 1.5e-4) and report the (smaller) N it implies.
OPTIMISTIC        = True
OPTIMISTIC_FACTOR = 1.5

# generate starting .expo files (extrapolated exponents at each shell's minimal N) for
# the downstream thorough optimizer — always the standard one, plus the optimistic one
# if OPTIMISTIC above is on. ATOM sets the element metadata / filename.
GENERATE_EXPO = True
ATOM          = "Si"
M_PARAMS      = 2       # tempering params the run used (must match how a0/a1 were fit)
N_FIT_POINTS  = 4       # optima nearest the target N used for the param extrapolation

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════


def read_results(path):
    """Group (N, E_cma) by shell. Re-sorted by N per shell, so ordered and
    unordered (partial) files both work."""
    shells = {}
    with open(path) as f:
        for r in csv.reader(f):
            if not r or not r[0].strip().lstrip("-").isdigit():
                continue                                     # header / blank / comment
            s = int(r[0]); n = int(r[2]); e = float(r[3]); a0 = float(r[4]); a1 = float(r[5])
            shells.setdefault(s, {"l": r[1], "pts": []})["pts"].append((n, e, a0, a1))
    for s in shells:
        shells[s]["pts"].sort()
    return shells


def minimal_n(pts, e_inf, A, b, dE):
    """Smallest N with E(N) - E_inf <= dE, read off the fitted curve so it interpolates
    between strided sample points. If that N happens to be one we actually evaluated,
    report its measured energy (known for certain); otherwise take it from the fit.
    Source label:
       data-verified     - N is a sampled point (exact measured energy)
       interpolated       - N is between sampled points (inside the range)
       extrapolated down  - N is below the smallest sampled (loose dE)
       extrapolated up    - N is above the largest sampled  (tight dE)
    Returns (n, source, e_at)."""
    sampled = {p[0]: p[1] for p in pts}
    ns      = sorted(sampled)
    n_fit   = _cbs_target_n(A, b, dE)
    if n_fit is None:
        if A > 0.0 and b > 0.0 and A <= dE:
            n_fit = 1                          # tolerance looser than the whole curve
        else:
            return None, "unreachable (bad fit)", None

    if n_fit in sampled:
        return n_fit, "data-verified", sampled[n_fit]           # we evaluated it -> exact
    if   n_fit < ns[0]:  src = "extrapolated down"
    elif n_fit > ns[-1]: src = "extrapolated up"
    else:                src = "interpolated"
    return n_fit, src, _cbs_predict(e_inf, A, b, n_fit)


shells = read_results(RESULTS_CSV)

lines = []
def out(s=""):
    lines.append(s)
    print(s)

out("=" * 72)
out("CBS minimal-N report")
out(f"source: {RESULTS_CSV}")
if OPTIMISTIC:
    out(f"optimistic column: tolerance relaxed x{OPTIMISTIC_FACTOR}")
out("=" * 72)

total_n         = 0   # sum of N (radial functions)
total_n_opt     = 0
total_funcs     = 0   # sum of N*(2l+1) (basis functions, counting the 2l+1 orientations)
total_funcs_opt = 0
specs           = []  # (label, N) per shell -> the sNpN... spec string
specs_opt       = []
target_ns       = {}  # shell -> minimal N (standard), for the .expo
target_ns_opt   = {}  # shell -> minimal N (optimistic)
incomplete      = False

for s in sorted(shells):
    lbl = shells[s]["l"] or (L_LABELS[s] if s < len(L_LABELS) else str(s))
    pts = shells[s]["pts"]
    ns  = [p[0] for p in pts]
    es  = [p[1] for p in pts]
    dE  = SHELL_TOLERANCES.get(s, DEFAULT_TOLERANCE)

    out("")
    if len(ns) < CBS_MIN_POINTS:
        out(f"shell {s} ({lbl}): only {len(ns)} point(s) — need >= {CBS_MIN_POINTS} to fit, skipping")
        incomplete = True
        continue

    e_inf, A, b, r2 = _cbs_fit(ns, es)
    e_infs  = [_cbs_fit(ns[start:], es[start:])[0] for start in range(0, len(ns) - CBS_MIN_POINTS + 1)]
    band_lo = (min(e_infs) - e_inf) * 1e6   # how far below the estimate the tail windows reach
    band_hi = (max(e_infs) - e_inf) * 1e6   # ...and above

    out(f"shell {s} ({lbl}):  E_inf = {e_inf:.8f} Eh   [tail-window band {band_lo:+.1f}..{band_hi:+.1f} uEh]"
        f"   b = {b:.3f}   r2 = {r2:.5f}")
    out(f"   sampled N: {ns[0]}..{ns[-1]} ({len(ns)} pts)   target: within {dE:.1e} Eh of the limit")

    n_needed, source, e_at = minimal_n(pts, e_inf, A, b, dE)
    if n_needed is None:
        out(f"   --> tolerance not reachable from this fit (A <= dE or b <= 0)")
        incomplete = True
        continue

    total_n     += n_needed
    total_funcs += n_needed * (2 * s + 1)
    specs.append((lbl, n_needed))
    target_ns[s] = n_needed
    out(f"   --> minimal N = {n_needed}   [{source}]   (remaining error {(e_at - e_inf) * 1e6:+.1f} uEh)")

    # marginal value of +/- one function, from the fitted curve
    if n_needed - 1 >= 1:
        drop = (_cbs_predict(e_inf, A, b, n_needed - 1) - _cbs_predict(e_inf, A, b, n_needed)) * 1e6
        out(f"       drop  to N={n_needed - 1}:  energy {drop:+.1f} uEh  (higher / worse)")
    raise_ = (_cbs_predict(e_inf, A, b, n_needed) - _cbs_predict(e_inf, A, b, n_needed + 1)) * 1e6
    out(f"       raise to N={n_needed + 1}:  energy {-raise_:+.1f} uEh  (lower / better)")

    if OPTIMISTIC:
        n_opt, src_opt, _ = minimal_n(pts, e_inf, A, b, dE * OPTIMISTIC_FACTOR)
        if n_opt is not None:
            total_n_opt     += n_opt
            total_funcs_opt += n_opt * (2 * s + 1)
            specs_opt.append((lbl, n_opt))
            target_ns_opt[s] = n_opt
            out(f"   --> optimistic N = {n_opt}   [{src_opt}, tol {dE * OPTIMISTIC_FACTOR:.1e}]")
        else:
            out(f"   --> optimistic N: not reachable")
            incomplete = True

out("")
out("=" * 72)
tag  = "  (INCOMPLETE — some shells unresolved)" if incomplete else ""
spec = "".join(f"{lbl}{n}" for lbl, n in specs)
out(f"TARGET      {spec}{tag}")
out(f"            radial (sum N) = {total_n}    basis functions (sum N*(2l+1)) = {total_funcs}")
if OPTIMISTIC:
    spec_opt = "".join(f"{lbl}{n}" for lbl, n in specs_opt)
    out(f"OPT x{OPTIMISTIC_FACTOR}    {spec_opt}{tag}")
    out(f"            radial (sum N) = {total_n_opt}    basis functions (sum N*(2l+1)) = {total_funcs_opt}")
out("=" * 72)


def write_expo(targets, filename):
    """Assemble a starting basis: each shell's exponents at its target N, taken from the
    measured params if that N was sampled, else extrapolated. Saved uncontracted."""
    present = sorted(targets)
    if not present or present != list(range(present[-1] + 1)):
        out(f"  [skip {filename}: shells {present} are not contiguous from 0]")
        return
    exp_list = []
    for l in present:
        pts     = shells[l]["pts"]
        N       = targets[l]
        sampled = {p[0]: (p[2], p[3]) for p in pts}
        if N in sampled:
            params = array(sampled[N], dtype=float64)                     # exact optimized params
        else:
            params = _extrapolate_start([(p[0], p[2], p[3]) for p in pts], N, N_FIT_POINTS)
        exp_list.append(from_registry("polynomial", m=M_PARAMS, n=N).decode(params, N))
    es   = Exponent_Set(atom_name=ATOM, exponents=exp_list)
    path = es.save(RESULTS_CSV.parent, filename, overwrite=True)
    out(f"  saved {path}")


if GENERATE_EXPO:
    out("")
    out("starting .expo (extrapolated exponents at each shell's minimal N):")
    write_expo(target_ns, f"{ATOM}_cbs_start.expo")
    if OPTIMISTIC:
        write_expo(target_ns_opt, f"{ATOM}_cbs_start_opt.expo")

log_path = RESULTS_CSV.parent / "cbs_n_report.log"
log_path.write_text("\n".join(lines) + "\n")
print(f"\nsaved {log_path}")
