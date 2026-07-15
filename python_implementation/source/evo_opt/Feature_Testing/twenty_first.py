import sys
import csv
import shutil
import argparse
from pathlib import Path
from numpy import array, float64

_arg_parser = argparse.ArgumentParser(description="CBS limit sweep: evaluate energy vs N primitives for a fixed tempering parameterization")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, default=None)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = ((_args.work_dir if _args.work_dir is not None else SUBMIT_DIR) / "CBS").resolve()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry

# ---- user config ----
ACTIVE_SHELL    = 0
GENERATOR       = "polynomial"
PARAMS          = [9.5, -1.4, 0.03, 0.0, 0.0, 0.0]   # M tempering params — edit these
A               = 6     # smallest N to evaluate
B               = 30    # largest N to evaluate a
USE_CONTRACTION = True  # True: bootstrap GENANO contractions and keep frozen shells contracted
THREADS         = 4
# ---- end user config ----

exp_path      = SUBMIT_DIR / "Si.expo"
template      = SUBMIT_DIR / "template.inp"
template_full = SUBMIT_DIR / "template_full.inp"
run_scr       = SUBMIT_DIR / "run.sh"
extract_scr   = SUBMIT_DIR / "extract.sh"

START_DIR = WORK_DIR / "Start"
WORK_DIR.mkdir(parents=True, exist_ok=True)
START_DIR.mkdir(parents=True, exist_ok=True)

_srcs = [exp_path, template, run_scr, extract_scr]
if USE_CONTRACTION:
    _srcs.append(template_full)
for _src in _srcs:
    shutil.copy(_src, START_DIR / _src.name)

exp_path      = START_DIR / exp_path.name
template      = START_DIR / template.name
template_full = START_DIR / template_full.name
run_scr       = START_DIR / run_scr.name
extract_scr   = START_DIR / extract_scr.name

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template,      cfg)
full_objective = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

params = array(PARAMS, dtype=float64)

L_LABELS = ["s", "p", "d", "f", "g", "h"]
lbl = L_LABELS[ACTIVE_SHELL] if ACTIVE_SHELL < len(L_LABELS) else str(ACTIVE_SHELL)

if USE_CONTRACTION:
    init_uncontracted = evaluate_initial(exp, full_objective, WORK_DIR / "initial_uncontracted", threads=THREADS)
    if init_uncontracted.resulting_contraction is None:
        raise RuntimeError("Initial uncontracted run produced no contraction.")
    base = init_uncontracted.copy(no_energy=True)
    base.change_contraction(init_uncontracted.resulting_contraction)
    print(f"Bootstrap E (uncontracted) : {init_uncontracted.energy:.10f} Eh")
else:
    base = exp.copy(no_energy=True)

print(f"Generator  : {GENERATOR}  M={len(params)}")
print(f"Shell      : {ACTIVE_SHELL} ({lbl})")
print(f"N range    : {A} .. {B}")
print(f"Contraction: {'on' if USE_CONTRACTION else 'off'}")
print()

CSV_FILE = SUBMIT_DIR / "cbs_log.csv"

ns       = list(range(A, B + 1))
exp_list = []
for n in ns:
    codec    = from_registry(GENERATOR, m=len(params), n=n)
    work_exp = base.copy(no_energy=True)
    work_exp.apply_params(ACTIVE_SHELL, codec, params, n=n)
    if not USE_CONTRACTION:
        work_exp.uncontract_all()
    exp_list.append(work_exp)

print(f"Submitting {len(ns)} jobs ({THREADS} concurrent)...")
results = objective.evaluate_batch(exp_list, work_dir=WORK_DIR / "batch", threads=THREADS)

with open(CSV_FILE, "w", newline="") as csv_f:
    writer = csv.writer(csv_f)
    writer.writerow([f"# generator={GENERATOR}  M={len(params)}  shell={ACTIVE_SHELL}({lbl})  contraction={USE_CONTRACTION}  params={list(params)}"])
    writer.writerow(["N", "energy"])
    for n, result in zip(ns, results):
        e = result.energy
        print(f"N={n:3d} | E = {e:.10f} Eh")
        writer.writerow([n, f"{e:.10f}"])

print()
print(f"CSV : {CSV_FILE}")
