import sys
from pathlib import Path

WORK_DIR   = Path(sys.argv[1])
SUBMIT_DIR = Path(sys.argv[2])
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.scipy_opt import scipy_fixed_exponent_count


exp_path    = SUBMIT_DIR / "exp.expo"
template    = SUBMIT_DIR / "template.inp"
submit_scr  = SUBMIT_DIR / "run.sh"
extract_scr = SUBMIT_DIR / "extract.sh"

METHOD      = "BFGS"
MAX_ITER    = 50
THREADS     = 4
FD_STEP     = 1e-4

# ── Setup ─────────────────────────────────────────────────────────────────────

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = WORK_DIR / "jobs",
    overwrite_existing   = True,
    custom_poll_interval = 0.1
)

objective = Ground_Energy_Objective(template, cfg)

# ── Optimize ──────────────────────────────────────────────────────────────────

best_exp, best_energy, result = scipy_fixed_exponent_count(
    exp,
    None,
    objective,
    work_dir       = WORK_DIR / "scipy_run",
    method         = METHOD,
    max_iterations = MAX_ITER,
    threads        = THREADS,
    fd_step        = FD_STEP,
    logging        = True,
    active_shells  = [0,1,0,0,0] 
)


# ── Results ───────────────────────────────────────────────────────────────────

print(f"\nMethod        : {METHOD}")
print(f"Converged     : {result.success}")
print(f"Message       : {result.message}")
print(f"Best energy   : {best_energy:.10f}  Hartree")

print(f"\nFinal exponents vs start:")
for l in range(len(exp.exponents)):
    print(f"  Shell {l}:")
    for q, (a0, a1) in enumerate(zip(exp.exponents[l], best_exp.exponents[l])):
        print(f"    q={q}  {float(a0):.6f}  ->  {float(a1):.6f}  ({(float(a1)/float(a0)-1)*100:+.2f}%)")
