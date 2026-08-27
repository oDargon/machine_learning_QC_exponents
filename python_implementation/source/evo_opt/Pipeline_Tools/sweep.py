import argparse
from pathlib import Path

from evo_opt.pipeline_core.sweep import Sweep_Config, run_sweep

_arg_parser = argparse.ArgumentParser(description="CMA-ES 2D tempering convergence, all shells in parallel, over N with a configurable step")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS          = [0, 1, 2, 3, 4]   # all swept independently, in parallel
N_INCREASES     = 5        # per shell: this many points beyond N_start
N_STEP          = 1        # gap between successive N points (1 -> 17,18,19..; 2 -> 17,19,21..)
USE_CONTRACTION = True

M_PARAMS        = 2        # 2D tempering
GENERATOR       = "polynomial"  # tempering generator (recorded in the CSV #META header)
SIGMA           = 0.1      # CMA step-size (CMA adapts it internally from here)
GENERATION_SIZE = 6        # CMA population per generation
MAX_GENERATIONS = 100      # hard cap; the early-stop should end well before this
USE_STOPPING    = True     # last-5-best-energies-within-1e-6 early stop

# core budget: run TOTAL_THREADS // THREADS_PER_SHELL shells concurrently, each CMA
# run using THREADS_PER_SHELL cores. On HPC bump TOTAL_THREADS and set per-shell = 6.
TOTAL_THREADS     = 6
THREADS_PER_SHELL = 3

USE_EXTRAPOLATION = True
N_FIT_POINTS      = 4         # optima nearest the query used for the local extrapolation

# CMA-ES seed handed to every CMA run unchanged. int -> all runs use it; None -> random.
SEED = None

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

cfg = Sweep_Config(
    submit_dir        = _args.submit_dir,
    work_dir          = _args.work_dir,
    expo_file         = EXPO_FILE,
    template_cont     = TEMPLATE_CONT,
    template_full     = TEMPLATE_FULL,
    run_script        = RUN_SCRIPT,
    extract_script    = EXTRACT_SCRIPT,
    shells            = SHELLS,
    n_increases       = N_INCREASES,
    n_step            = N_STEP,
    use_contraction   = USE_CONTRACTION,
    m_params          = M_PARAMS,
    generator         = GENERATOR,
    sigma             = SIGMA,
    generation_size   = GENERATION_SIZE,
    max_generations   = MAX_GENERATIONS,
    use_stopping      = USE_STOPPING,
    total_threads     = TOTAL_THREADS,
    threads_per_shell = THREADS_PER_SHELL,
    use_extrapolation = USE_EXTRAPOLATION,
    n_fit_points      = N_FIT_POINTS,
    seed              = SEED,
)
run_sweep(cfg)
