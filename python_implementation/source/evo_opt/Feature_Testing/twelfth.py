import sys
from pathlib import Path

WORK_DIR   = Path.cwd()
SUBMIT_DIR = Path.cwd()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import cma_fixed_exponent_count, evaluate_initial

exp_path    = SUBMIT_DIR / "exp.expo"
template    = SUBMIT_DIR / "template.inp"
submit_scr  = SUBMIT_DIR / "run.sh"
extract_scr = SUBMIT_DIR / "extract.sh"

ACTIVE_SHELL    = 1
GENERATION_SIZE = 12
SIGMA           = 0.1
MAX_GENERATIONS = 30
THREADS         = 4

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = WORK_DIR / "jobs",
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)

objective = Ground_Energy_Objective(template, cfg)

start_exp    = evaluate_initial(exp, objective, WORK_DIR / "initial_eval", threads=THREADS)
start_energy = start_exp.energy

print(f"Start energy  : {start_energy:.10f}  Hartree")

best_exp, best_energy, es = cma_fixed_exponent_count(
    start_exp,
    start_energy,
    objective,
    work_dir        = WORK_DIR / "cma_run",
    generation_size = GENERATION_SIZE,
    sigma           = SIGMA,
    max_generations = MAX_GENERATIONS,
    threads         = THREADS,
    active_shell    = ACTIVE_SHELL,
    overwrite       = True,
    logging         = True,
)

print(f"\nBest energy   : {best_energy:.10f}  Hartree")
print(f"Final sigma   : {es.sigma:.4e}")
print(f"\nP-shell exponents (l={ACTIVE_SHELL}):")
for q in range(len(exp.exponents[ACTIVE_SHELL])):
    a0 = float(exp.exponents[ACTIVE_SHELL][q])
    a1 = float(best_exp.exponents[ACTIVE_SHELL][q])
    print(f"  q={q}  {a0:.6f}  ->  {a1:.6f}  ({(a1/a0 - 1)*100:+.2f}%)")
