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

_arg_parser = argparse.ArgumentParser(description="Scan a shell's 2D tempering energy surface across increasing N over a FIXED (a0,a1) window (no recentering)")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Surface_Fixed").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "B.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS          = [1]      # shells to scan; each is swept independently over N
N_INCREASES     = 5        # per shell: scan N from N_start up to N_start + this many
USE_CONTRACTION = True

M_PARAMS        = 2        # tempering params → 2D grid; do not change (the sweep is 2D)

# fixed (a0, a1) window scanned identically for EVERY N (no recentering on the prev min)
A0_MIN, A0_MAX  = 2.0, 10.0
A1_MIN, A1_MAX  = -16.0, -5.0
GRID            = 21       # points per axis (GRID**2 evals per N)

THREADS         = 6
POLL_INTERVAL   = 0.5      # seconds between job-completion checks in the manager (default is 5.0)

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
    executor_type        = Executor_Type.LOCAL_BASH,
    execution_script     = run_scr,
    extraction_script    = extract_scr,
    overwrite_existing   = True,
    custom_poll_interval = POLL_INTERVAL,
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


# ─── fixed window: same (a0, a1) grid for every N ─────────────────────────────

a0s         = linspace(A0_MIN, A0_MAX, GRID)
a1s         = linspace(A1_MIN, A1_MAX, GRID)
GA, GB      = meshgrid(a0s, a1s)
grid_params = column_stack([GA.ravel(), GB.ravel()])

for shell in SHELLS:
    lbl       = L_LABELS[shell]
    shell_dir = RESULTS_DIR / lbl
    shell_dir.mkdir(parents=True, exist_ok=True)
    n0        = len(base.exponents[shell])
    center    = array(from_registry("polynomial", m=M_PARAMS, n=n0).encode(base.exponents[shell]), dtype=float64)
    print(f"=== shell {shell} ({lbl}): N {n0}..{n0 + N_INCREASES}   fixed a0[{A0_MIN},{A0_MAX}] a1[{A1_MIN},{A1_MAX}] ===")

    for N in range(n0, n0 + N_INCREASES + 1):
        codec = from_registry("polynomial", m=M_PARAMS, n=N)

        print(f"  N={N:3d}: scanning {GRID}x{GRID} over the fixed window", flush=True)
        Z = scan_objective(grid_params, shell, codec, N).reshape(GA.shape)

        best_flat = int(Z.argmin())
        grid_min  = array([GA.ravel()[best_flat], GB.ravel()[best_flat]], dtype=float64)
        print(f"         min E = {Z.min():.10f} Eh at ({grid_min[0]:+.4f}, {grid_min[1]:+.4f})", flush=True)

        out_path = shell_dir / f"scan_shell{shell}_N{N:02d}.npz"
        savez(
            out_path,
            shell=shell, l=lbl, N=N, m=M_PARAMS,
            grid=GRID, a0s=a0s, a1s=a1s, Z=Z,
            center=center, grid_min=grid_min,
        )
        print(f"         saved {lbl}/{out_path.name}", flush=True)

        shutil.rmtree(BATCH_DIR, ignore_errors=True)        # free the batch before the next N

print(f"\nAll scans saved under {RESULTS_DIR}, one folder per shell: {', '.join(L_LABELS[s] for s in SHELLS)}")
