import argparse
from pathlib import Path

from evo_opt.common import CHEMICAL_ACCURACY
from evo_opt.pipeline_core.target import Target_Config, run_target

_arg_parser = argparse.ArgumentParser(description="From a CMA-minima results CSV: fit E_inf per shell and report the minimal N to reach each shell's energy tolerance")
_arg_parser.add_argument("--results", type=Path, required=True,
                         help="cma_minima CSV (ordered or unordered/partial — rows are re-sorted by N per shell)")
_args = _arg_parser.parse_args()

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
# if OPTIMISTIC above is on. The atom / generator / M are read from the CSV #META
# header the sweep wrote — no need to restate them here.
GENERATE_EXPO = True
N_FIT_POINTS  = 4       # optima nearest the target N used for the param extrapolation

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

cfg = Target_Config(
    results           = _args.results,
    shell_tolerances  = SHELL_TOLERANCES,
    default_tolerance = DEFAULT_TOLERANCE,
    cbs_min_points    = CBS_MIN_POINTS,
    optimistic        = OPTIMISTIC,
    optimistic_factor = OPTIMISTIC_FACTOR,
    generate_expo     = GENERATE_EXPO,
    n_fit_points      = N_FIT_POINTS,
)
run_target(cfg)
