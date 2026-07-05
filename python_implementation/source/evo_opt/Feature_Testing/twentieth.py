import sys
import shutil
import argparse
from pathlib import Path

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, default=None)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = ((_args.work_dir if _args.work_dir is not None else SUBMIT_DIR) / "Optimization").resolve()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import evaluate_initial, cma_fixed_exponent_count

ACTIVE_SHELL       = 0
GENERATION_SIZE    = 10
SIGMA              = 0.01
MAX_GENERATIONS    = 5
THREADS            = 4
USE_TEMPERING      = False
N_TEMPERING_PARAMS = 6

exp_path      = SUBMIT_DIR / "Si.expo"
template      = SUBMIT_DIR / "template.inp"
template_full = SUBMIT_DIR / "template_full.inp"
run_scr       = SUBMIT_DIR / "run.sh"
extract_scr   = SUBMIT_DIR / "extract.sh"

START_DIR = WORK_DIR / "Start"
WORK_DIR.mkdir(parents=True, exist_ok=True)
START_DIR.mkdir(parents=True, exist_ok=True)

for _src in (exp_path, template, template_full, run_scr, extract_scr):
    shutil.copy(_src, START_DIR / _src.name)

exp_path      = START_DIR / exp_path.name
template      = START_DIR / template.name
template_full = START_DIR / template_full.name
run_scr       = START_DIR / run_scr.name
extract_scr   = START_DIR / extract_scr.name

exp = Exponent_Set.from_file(exp_path)

cfg            = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template,      cfg)
full_objective = Ground_Energy_Objective(template_full, cfg)

init_uncontracted = evaluate_initial(exp, full_objective, WORK_DIR / "initial_uncontracted", threads=THREADS)

if init_uncontracted.resulting_contraction is None:
    raise RuntimeError("Initial uncontracted run produced no contraction.")

L_LABELS = ["s", "p", "d", "f", "g", "h"]
if USE_TEMPERING:
    n_active = exp.lengths[ACTIVE_SHELL]
    print(f"Tempering           : polynomial  M={N_TEMPERING_PARAMS}  N={n_active}  shell={ACTIVE_SHELL}")
else:
    print(f"Tempering           : off")
print(f"Uncontracted energy : {init_uncontracted.energy:.10f} Eh")
print("Contraction sizes   :")
rc = init_uncontracted.resulting_contraction
for i in range(len(rc)):
    lbl = L_LABELS[i] if i < len(L_LABELS) else str(i)
    print(f"  shell {i} ({lbl}): {rc[i].shape[0]} <- {rc[i].shape[1]}")

contracted = init_uncontracted.copy(no_energy=True)
contracted.change_contraction(init_uncontracted.resulting_contraction)
contracted.uncontract_shell(ACTIVE_SHELL)

init_contracted = evaluate_initial(
    contracted, objective, WORK_DIR / "initial_contracted",
    threads=THREADS, contract_frozen_shells=True,
)
print(f"Contracted energy   : {init_contracted.energy:.10f} Eh")

cma_fixed_exponent_count(
    init_contracted,
    float(init_contracted.energy),
    objective,
    work_dir               = WORK_DIR / "cma_run",
    generation_size        = GENERATION_SIZE,
    sigma                  = SIGMA,
    max_generations        = MAX_GENERATIONS,
    threads                = THREADS,
    active_shell           = ACTIVE_SHELL,
    overwrite              = True,
    logging                = True,
    contract_frozen_shells = True,
    out_dir                = SUBMIT_DIR,
    use_tempering          = USE_TEMPERING,
    n_tempering_params     = N_TEMPERING_PARAMS,
)
