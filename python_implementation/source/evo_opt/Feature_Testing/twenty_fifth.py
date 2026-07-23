import shutil
import argparse
from pathlib import Path
from numpy import array, float64, linspace, meshgrid, column_stack, savez

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry

_arg_parser = argparse.ArgumentParser(description="Sweep a shell's 2D tempering energy surface across increasing N")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Surface_Sweep").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS          = [0]      # shells to scan; each is swept independently over N
N_INCREASES     = 5        # per shell: scan N from N_start up to N_start + this many
USE_CONTRACTION = True

M_PARAMS        = 2        # tempering params → 2D grid; do not change (the sweep is 2D)
SCAN_HALFWIDTH  = 1.5      # +/- range around the center in each param direction
GRID            = 21       # points per axis (GRID**2 evals per N)
THREADS         = 6

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

START_DIR   = WORK_DIR / "Start"
BATCH_DIR   = WORK_DIR / "batch"
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

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
energy_objective = Ground_Energy_Objective(template_cont, cfg)
full_objective   = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

# ─── frozen backdrop (contracted or not) ──────────────────────────────────────

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


def scan_objective(params_batch, shell, codec, n):
    """Decode each 2-param vector into n exponents for `shell`, evaluate the batch.
    Job files are overwritten each call so disk use stays at one batch."""
    exp_objects = []
    for i in range(len(params_batch)):
        work = base.copy(no_energy=True)
        work.apply_params(shell, codec, params_batch[i], n=n)
        if not USE_CONTRACTION:
            work.uncontract_all()
        exp_objects.append(work)
    results = energy_objective.evaluate_batch(exp_objects, work_dir=BATCH_DIR, threads=THREADS, overwrite=True)
    return array([float(r.energy) for r in results], dtype=float64)


# ─── sweep: per shell, N_start .. N_start + N_INCREASES ───────────────────────

for shell in SHELLS:
    lbl = L_LABELS[shell]
    n0  = len(base.exponents[shell])
    print(f"=== shell {shell} ({lbl}): N {n0}..{n0 + N_INCREASES} ===")

    center = None   # first N centers on the encoded starting exponents; later N on prev min
    for N in range(n0, n0 + N_INCREASES + 1):
        codec = from_registry("polynomial", m=M_PARAMS, n=N)
        if center is None:
            center = array(codec.encode(base.exponents[shell]), dtype=float64)

        a0s = linspace(center[0] - SCAN_HALFWIDTH, center[0] + SCAN_HALFWIDTH, GRID)
        a1s = linspace(center[1] - SCAN_HALFWIDTH, center[1] + SCAN_HALFWIDTH, GRID)
        GA, GB      = meshgrid(a0s, a1s)
        grid_params = column_stack([GA.ravel(), GB.ravel()])

        print(f"  N={N:3d}: scanning {GRID}x{GRID} around ({center[0]:+.4f}, {center[1]:+.4f})", flush=True)
        Z = scan_objective(grid_params, shell, codec, N).reshape(GA.shape)

        best_flat = int(Z.argmin())
        grid_min  = array([GA.ravel()[best_flat], GB.ravel()[best_flat]], dtype=float64)
        print(f"         min E = {Z.min():.10f} Eh at ({grid_min[0]:+.4f}, {grid_min[1]:+.4f})", flush=True)

        out_path = RESULTS_DIR / f"scan_shell{shell}_N{N:02d}.npz"
        savez(
            out_path,
            shell=shell, l=lbl, N=N, m=M_PARAMS,
            halfwidth=SCAN_HALFWIDTH, grid=GRID,
            a0s=a0s, a1s=a1s, Z=Z,
            center=center, grid_min=grid_min,
        )
        print(f"         saved {out_path.name}", flush=True)

        center = grid_min                                   # next N centers here
        shutil.rmtree(BATCH_DIR, ignore_errors=True)        # free the batch before the next N

print(f"\nAll scans saved to {RESULTS_DIR}")
