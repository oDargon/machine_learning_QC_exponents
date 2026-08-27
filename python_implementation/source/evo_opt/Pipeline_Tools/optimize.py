import sys
import argparse
from pathlib import Path

from evo_opt.pipeline_core.optimize import Optimize_Config, run_optimize

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

sys.path.insert(0, str(_args.submit_dir.resolve()))

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

# --- input files (looked up in the submit dir) ---
EXPO_FILE      = "Ru.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # contracted frozen shells
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

# --- per-shell optimization ---
OPTIMIZE_FLAGS         = [1, 1, 1, 1, 1, 1]  # 1 = optimize, 0 = freeze; may be shorter than n_shells
GENERATION_SIZE        = [6, 6, 6, 6, 6, 6]  # int, or one entry per OPTIMIZE_FLAGS entry
THREADS_PER_SHELL      = [6, 6, 6, 6, 6, 6]  # int, or one entry per OPTIMIZE_FLAGS entry
SIGMA                  = 0.1
MAX_GENERATIONS        = 300  # target: run until every shell has reached this many gens
GEN_CEILING_MULTIPLIER = 5    # hard per-shell ceiling = MAX_GENERATIONS * this; a fast shell
                              #   may run ahead up to the ceiling while slower shells catch up
                              #   (assumes no shell is more than this many times slower)
USE_CONTRACTION        = True  # True : freeze + contract the other shells while optimizing one
                               #        (uses TEMPLATE_CONT) — cheaper per eval
                               # False: optimize with everything fully uncontracted
                               #        (uses TEMPLATE_FULL). Global evals are always full.
USE_TEMPERING          = True
N_TEMPERING_PARAMS     = 6

# --- global (fully-uncontracted) evaluations ---
THREADS_GLOBAL           = 2   # max concurrent fully-uncontracted eval jobs in flight
GLOBAL_EVAL_WARMUP_GENS  = 10  # all shells must reach this many gens before the first global eval
GLOBAL_EVAL_SPACING_GENS = 2   # all shells must advance this many gens between global evals

# --- early stopping (on the global evals) ---
EARLY_STOP        = True   # stop the whole run once the global energy has plateaued
EARLY_STOP_WINDOW = 5      # number of most-recent global evals that must agree
EARLY_STOP_TOL    = 1e-5   # max spread (Eh) across that window to count as converged

# --- cross-shell coupling ---
# Feed the converged global contraction back to every shell optimizer as its new
# root, so each shell sees the other shells' improvements instead of optimizing
# against a frozen initial guess.
ENABLE_CROSS_SHELL      = False
CROSS_SHELL_WARMUP_GENS = 20   # all shells must reach this many gens before coupling starts

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

cfg = Optimize_Config(
    submit_dir               = _args.submit_dir,
    work_dir                 = _args.work_dir,
    expo_file                = EXPO_FILE,
    template_cont            = TEMPLATE_CONT,
    template_full            = TEMPLATE_FULL,
    run_script               = RUN_SCRIPT,
    extract_script           = EXTRACT_SCRIPT,
    optimize_flags           = OPTIMIZE_FLAGS,
    generation_size          = GENERATION_SIZE,
    threads_per_shell        = THREADS_PER_SHELL,
    sigma                    = SIGMA,
    max_generations          = MAX_GENERATIONS,
    gen_ceiling_multiplier   = GEN_CEILING_MULTIPLIER,
    use_contraction          = USE_CONTRACTION,
    use_tempering            = USE_TEMPERING,
    n_tempering_params       = N_TEMPERING_PARAMS,
    threads_global           = THREADS_GLOBAL,
    global_eval_warmup_gens  = GLOBAL_EVAL_WARMUP_GENS,
    global_eval_spacing_gens = GLOBAL_EVAL_SPACING_GENS,
    early_stop               = EARLY_STOP,
    early_stop_window        = EARLY_STOP_WINDOW,
    early_stop_tol           = EARLY_STOP_TOL,
    enable_cross_shell       = ENABLE_CROSS_SHELL,
    cross_shell_warmup_gens  = CROSS_SHELL_WARMUP_GENS,
)
run_optimize(cfg)
