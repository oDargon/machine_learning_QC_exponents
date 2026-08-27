import csv
import shutil
import argparse
from pathlib import Path
from numpy import log, exp, linspace, array, float64, argmin, abs as np_abs

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial

_arg_parser = argparse.ArgumentParser(description="Fine 1-D energy scan of a single-exponent shell across log space")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "ShellScan").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELL_IDX       = 4       # single-exponent shell to scan
USE_CONTRACTION = True    # match the per-shell optimizer: frozen shells contracted
SPAN            = 6.0     # scan ln(exponent) over start_ln ± SPAN
N_GRID          = 121     # grid points (fine)
THREADS         = 6

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

START_DIR   = WORK_DIR / "Start"
RESULTS_DIR = SUBMIT_DIR / "results"
START_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def stage(name: str) -> Path:
    dst = START_DIR / name
    shutil.copy(SUBMIT_DIR / name, dst)
    return dst


exp_path      = stage(EXPO_FILE)
template_cont = stage(TEMPLATE_CONT)
run_scr       = stage(RUN_SCRIPT)
extract_scr   = stage(EXTRACT_SCRIPT)
template_full = stage(TEMPLATE_FULL) if USE_CONTRACTION else None

basis = Exponent_Set.from_file(exp_path)
n_exp = len(basis.exponents[SHELL_IDX])
lbl   = L_LABELS[SHELL_IDX]
print(f"shell {SHELL_IDX} ({lbl}): {n_exp} exponent(s)")
if n_exp != 1:
    print(f"  [note] shell has {n_exp} exponents; scanning the first, holding the rest fixed")

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
energy_objective = Ground_Energy_Objective(template_cont, cfg)
full_objective   = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

# ─── frozen backdrop (mirror multi_shell_opt: contracted frozen shells) ────────

if USE_CONTRACTION:
    print("=== bootstrap contraction ===")
    boot = evaluate_initial(basis.copy(no_energy=True), full_objective, WORK_DIR, threads=THREADS, subdir_name="bootstrap")
    if boot.resulting_contraction is None:
        raise RuntimeError("bootstrap produced no contraction")
    base = boot.copy(no_energy=True)
    base.change_contraction(boot.resulting_contraction)
    print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh\n")
else:
    base = basis.copy(no_energy=True)
    base.uncontract_all()

shell_start = base.copy(no_energy=True)
if USE_CONTRACTION:
    shell_start.uncontract_shell(SHELL_IDX)   # active shell free, frozen shells stay contracted

# ─── log-space grid around the shell's starting exponent ──────────────────────

start_ln = float(log(basis.exponents[SHELL_IDX][0]))
grid_ln  = linspace(start_ln - SPAN, start_ln + SPAN, N_GRID)

print(f"scanning ln(exp) in [{start_ln - SPAN:+.3f}, {start_ln + SPAN:+.3f}] "
      f"around start {start_ln:+.3f}  ({N_GRID} points, Δ={2 * SPAN / (N_GRID - 1):.4f})\n")

exp_objects = []
for i in range(N_GRID):
    point = shell_start.copy(no_energy=True)
    point.set_shell_exponents(SHELL_IDX, exp(array([grid_ln[i]])))
    if not USE_CONTRACTION:
        point.uncontract_all()
    exp_objects.append(point)

results  = energy_objective.evaluate_batch(exp_objects, work_dir=WORK_DIR / "scan", threads=THREADS)
energies = array([r.energy for r in results], dtype=float64)

failed  = int((energies >= 1e5).sum())   # objectives.py stamps 1e6 on a failed job
min_idx = int(argmin(energies))
e_min   = float(energies[min_idx])
start_i = int(np_abs(grid_ln - start_ln).argmin())

# ─── write + report ───────────────────────────────────────────────────────────

csv_path = RESULTS_DIR / f"shell{SHELL_IDX}_scan.csv"
csv_f    = open(csv_path, "w", newline="")
writer   = csv.writer(csv_f)
writer.writerow(["idx", "ln_exp", "exponent", "energy", "delta_e_from_min"])
for i in range(N_GRID):
    writer.writerow([i, f"{grid_ln[i]:.10f}", f"{float(exp(grid_ln[i])):.10e}",
                     f"{energies[i]:.10f}", f"{energies[i] - e_min:.10e}"])
csv_f.close()

print(f"grid minimum : ln(exp) {grid_ln[min_idx]:+.4f}  exp {float(exp(grid_ln[min_idx])):.6e}  "
      f"E {e_min:.10f} Eh  (idx {min_idx}/{N_GRID - 1})")
print(f"start point  : ln(exp) {start_ln:+.4f}  E {energies[start_i]:.10f} Eh  "
      f"(ΔE to min {energies[start_i] - e_min:+.3e})")
if failed:
    print(f"failed jobs  : {failed}/{N_GRID} stamped at 1e6 (exponents MOLCAS could not run)")
print(f"\nsaved {csv_path}")
