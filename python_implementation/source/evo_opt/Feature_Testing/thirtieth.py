import shutil
import argparse
from pathlib import Path
from numpy import array, float64

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry

_arg_parser = argparse.ArgumentParser(description="Run 4 specific (a0,a1) tempering points for one shell as MOLCAS jobs; keep every job folder so the input files can be inspected")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Four_Points").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "B.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

ACTIVE_SHELL    = 1        # p
USE_CONTRACTION = True
N_OVERRIDE      = None     # None -> use the shell's own exponent count from the .expo
THREADS         = 4        # run all 4 points at once

# the 4 points in (a0, a1), paired index-wise
A0_POINTS = [3.58,   3.83,  4.07,  4.36]
A1_POINTS = [-7.784, -7.28, -6.78, -6.28]

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
energy_objective = Ground_Energy_Objective(template_cont, cfg)
full_objective   = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

# ─── frozen backdrop (same setup as the sweep / surface scans) ─────────────────

if USE_CONTRACTION:
    print("=== bootstrap contraction ===")
    boot = evaluate_initial(exp, full_objective, WORK_DIR, threads=THREADS, subdir_name="bootstrap")
    if boot.resulting_contraction is None:
        raise RuntimeError("bootstrap produced no contraction")
    base = boot.copy(no_energy=True)
    base.change_contraction(boot.resulting_contraction)
    print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh\n")
else:
    base = exp.copy(no_energy=True)

N     = N_OVERRIDE if N_OVERRIDE is not None else len(base.exponents[ACTIVE_SHELL])
codec = from_registry("polynomial", m=2, n=N)

if len(A0_POINTS) != len(A1_POINTS):
    raise ValueError(f"A0_POINTS ({len(A0_POINTS)}) and A1_POINTS ({len(A1_POINTS)}) must have the same length")

# ─── build one basis per point, keep each in its own named job folder ──────────

exp_objects = []
names       = []
for i in range(len(A0_POINTS)):
    params = array([A0_POINTS[i], A1_POINTS[i]], dtype=float64)
    work   = base.copy(no_energy=True)
    work.apply_params(ACTIVE_SHELL, codec, params, n=N)
    if not USE_CONTRACTION:
        work.uncontract_all()
    exp_objects.append(work)
    names.append(f"pt{i}")

print(f"=== running {len(exp_objects)} points for shell {ACTIVE_SHELL} ({L_LABELS[ACTIVE_SHELL]}), N={N} ===")
for i in range(len(A0_POINTS)):
    print(f"  pt{i}: a0={A0_POINTS[i]:+.4f}  a1={A1_POINTS[i]:+.4f}")

JOBS_DIR = WORK_DIR / "jobs"
results  = energy_objective.evaluate_batch(exp_objects, work_dir=JOBS_DIR, threads=THREADS, names=names, overwrite=True)

# ─── report; folders are NOT deleted, so the .input files stay put ─────────────

print("\n=== results (job folders kept for inspection) ===")
for i in range(len(results)):
    e = float(results[i].energy)
    print(f"  pt{i}: a0={A0_POINTS[i]:+.4f} a1={A1_POINTS[i]:+.4f}  ->  E = {e:.10f} Eh")
    print(f"        folder: {JOBS_DIR / names[i]}")

print(f"\nall job folders under: {JOBS_DIR}")
print("(inspect each pt*/ for the generated MOLCAS .input and output)")
