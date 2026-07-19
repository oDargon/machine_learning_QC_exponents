import shutil
import argparse
from pathlib import Path

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cbs_engine import run_cbs

_arg_parser = argparse.ArgumentParser(description="CBS sweep: optimise shell exponents across a range of N, extract CBS limit")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "CBS_Sweep").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS            = [0, 1, 2, 3, 4]    # shell indices to sweep
DOWN              = 2                   # steps below N_start (same for all shells)
UP                = 2                   # steps above N_start (same for all shells)

GENERATOR         = "polynomial"
M                 = 6    # polynomial params per shell
PHASE1_MAX_GENS   = 10   # max CMA generations in Phase 1 (initial optimisation at N_start)
PHASE2_MAX_GENS   = 10   # max CMA generations at each N in Phase 2 (CBS sweep)
SIGMA             = 0.1
GENERATION_SIZE   = 6
TOTAL_THREADS     = 6
THREADS_PER_SHELL = 3
USE_CONTRACTION   = True
USE_STOPPING      = True   # early-stop a sub-optimisation once its last 5 best energies agree to 1e-6

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

A = [len(exp.exponents[SHELLS[i]]) - DOWN for i in range(len(SHELLS))]
B = [len(exp.exponents[SHELLS[i]]) + UP   for i in range(len(SHELLS))]

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template_cont, cfg)
full_objective = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

CSV_DIR = WORK_DIR / "csvs"

run_cbs(
    exp,
    objective,
    full_objective,
    work_dir  = WORK_DIR,
    csv_dir   = CSV_DIR,
    shells    = SHELLS,
    A         = A,
    B         = B,
    generator             = GENERATOR,
    m                     = M,
    phase1_max_gens       = PHASE1_MAX_GENS,
    phase2_max_gens       = PHASE2_MAX_GENS,
    sigma                 = SIGMA,
    generation_size       = GENERATION_SIZE,
    total_threads         = TOTAL_THREADS,
    threads_per_shell     = THREADS_PER_SHELL,
    contract_frozen_shells = USE_CONTRACTION,
    use_stopping          = USE_STOPPING,
)

shutil.copy(CSV_DIR / "cbs_results.csv", SUBMIT_DIR / "cbs_results.csv")
print(f"Results copied to {SUBMIT_DIR / 'cbs_results.csv'}")
