import sys
from pathlib import Path

WORK_DIR   = Path.cwd() / "Optimization"
SUBMIT_DIR = Path.cwd()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt import cma_fixed_exponent_count, evaluate_initial

exp_path    = SUBMIT_DIR / "Si.expo"
template    = SUBMIT_DIR / "template.inp"
submit_scr  = SUBMIT_DIR / "run.sh"
extract_scr = SUBMIT_DIR / "extract.sh"

GENERATION_SIZE    = 35
SIGMA              = 0.01
MAX_GENERATIONS    = 500
THREADS            = 12
SHELLS_TO_OPTIMIZE = [0, 1, 2, 3]
USE_STOPPING       = True

WORK_DIR.mkdir(parents=True, exist_ok=True)

exp = Exponent_Set.from_file(exp_path)
exp.uncontract_all()

cfg = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = WORK_DIR,
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)
objective = Ground_Energy_Objective(template, cfg)

start_exp    = evaluate_initial(exp, objective, WORK_DIR / "initial_eval", threads=THREADS)
start_energy = start_exp.energy 

n_shells      = len(exp.exponents)
active_shells = [1 if j in SHELLS_TO_OPTIMIZE else 0 for j in range(n_shells)]

best_exp, best_energy, es = cma_fixed_exponent_count(
    start_exp,
    start_energy,
    objective,
    work_dir        = WORK_DIR / "cma_run",
    generation_size = GENERATION_SIZE,
    sigma           = SIGMA,
    max_generations = MAX_GENERATIONS,
    threads         = THREADS,
    overwrite       = True,
    logging         = True,
    use_stopping    = USE_STOPPING,
    active_shells   = active_shells,
)

best_exp.save(WORK_DIR, "best", overwrite=True)
print(f"Best energy: {best_energy:.10f} Eh")
