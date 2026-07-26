import shutil
import argparse
from pathlib import Path

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.cbs_engine import run_cbs

_arg_parser = argparse.ArgumentParser(description="CBS Component 1: per-shell converge-to-CBS (sweep + jump-to-N* + verify)")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "CBS_Sweep").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # frozen shells contracted
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted (bootstrap only)
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS            = [0, 1, 2, 3, 4]   # shell indices to converge (each independent)
INITIAL_STEPS     = 3                 # sweep N_start .. N_start+this (>=2 -> >=3 fit points)
TOL               = 1.0e-4            # target: within this of the per-shell CBS limit (Eh)
MAX_JUMPS         = 6                 # cap on jump-to-N* iterations before giving up
N_STAR_CAP        = 60                # refuse to jump past this N (guards a bad early fit); None = off

GENERATOR         = "polynomial"
M                 = 2                 # tempering params per shell (2 -> geom extrapolation in (a0,lnβ))
SIGMA             = 0.1
GENERATION_SIZE   = 6
MAX_GENERATIONS   = 100               # hard cap per CMA run; early-stop ends it sooner
N_FIT_POINTS      = 4                 # recent optima the geometric warm-start fits over
TOTAL_THREADS     = 6
THREADS_PER_SHELL = 6
USE_CONTRACTION   = True
USE_STOPPING      = True              # last-5-best-energies-within-1e-6 early stop
SEED              = None              # int for reproducible runs; None = random

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

START_DIR = WORK_DIR / "Start"
START_DIR.mkdir(parents=True, exist_ok=True)


def stage(name: str) -> Path:
    dst = START_DIR / name
    shutil.copy(SUBMIT_DIR / name, dst)
    return dst


exp_path      = stage(EXPO_FILE)
template_cont = stage(TEMPLATE_CONT)
run_scr       = stage(RUN_SCRIPT)
extract_scr   = stage(EXTRACT_SCRIPT)
template_full = stage(TEMPLATE_FULL) if USE_CONTRACTION else None

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template_cont, cfg)
full_objective = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── build the frozen backdrop the engine sweeps against: bootstrap the contraction
#    here so `base` arrives at run_cbs fully ready (contraction baked in) ──
if USE_CONTRACTION:
    print("=== bootstrap contraction ===")
    boot = evaluate_initial(exp, full_objective, WORK_DIR, threads=TOTAL_THREADS, subdir_name="bootstrap")
    if boot.resulting_contraction is None:
        raise RuntimeError("bootstrap produced no contraction")
    base = boot.copy(no_energy=True)
    base.change_contraction(boot.resulting_contraction)
    print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh\n")
else:
    base = exp.copy(no_energy=True)

# csv_dir override: put the CSVs in the submit dir so the live one is watchable
# during the run (the engine default is the work dir, which is scratch on HPC).
run_cbs(
    base,
    objective,
    work_dir  = WORK_DIR,
    shells    = SHELLS,
    csv_dir   = SUBMIT_DIR,
    m                      = M,
    initial_steps          = INITIAL_STEPS,
    tol                    = TOL,
    max_jumps              = MAX_JUMPS,
    n_star_cap             = N_STAR_CAP,
    generator              = GENERATOR,
    sigma                  = SIGMA,
    generation_size        = GENERATION_SIZE,
    max_generations        = MAX_GENERATIONS,
    n_fit_points           = N_FIT_POINTS,
    total_threads          = TOTAL_THREADS,
    threads_per_shell      = THREADS_PER_SHELL,
    contract_frozen_shells = USE_CONTRACTION,
    use_stopping           = USE_STOPPING,
    seed                   = SEED,
)
