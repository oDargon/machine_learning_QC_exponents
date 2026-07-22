import shutil
import argparse
from pathlib import Path
from numpy import array, float64, linspace, meshgrid, column_stack, median, savez
import matplotlib.pyplot as plt

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry
from evo_opt.newton_6 import Newton_6

_arg_parser = argparse.ArgumentParser(description="Scan one shell's 2D tempering energy surface + overlay Newton_6 path")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Newton6_Scan").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

ACTIVE_SHELL    = 0
USE_CONTRACTION = True

SCAN_HALFWIDTH  = 1.5     # +/- range around x0 in each param direction
GRID            = 21      # points per axis (GRID**2 total evals)
THREADS         = 6

RUN_OPTIMIZER   = True     # overlay a Newton_6 path on the surface
TRUST_RADIUS    = 0.1
MAX_STEPS       = 30

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

n_active = len(base.exponents[ACTIVE_SHELL])
codec    = from_registry("polynomial", m=2, n=n_active)
x0       = codec.encode(base.exponents[ACTIVE_SHELL])


def tempered_objective(params_batch):
    exp_objects = []
    for i in range(len(params_batch)):
        work = base.copy(no_energy=True)
        work.apply_params(ACTIVE_SHELL, codec, params_batch[i], n=n_active)
        if not USE_CONTRACTION:
            work.uncontract_all()
        exp_objects.append(work)
    results = energy_objective.evaluate_batch(exp_objects, work_dir=WORK_DIR / "batch", threads=THREADS, overwrite=True)
    return array([float(r.energy) for r in results], dtype=float64)


# ── scan the surface ──
a0s = linspace(x0[0] - SCAN_HALFWIDTH, x0[0] + SCAN_HALFWIDTH, GRID)
a1s = linspace(x0[1] - SCAN_HALFWIDTH, x0[1] + SCAN_HALFWIDTH, GRID)
GA, GB = meshgrid(a0s, a1s)
grid_params = column_stack([GA.ravel(), GB.ravel()])

print(f"=== scanning {GRID}x{GRID} = {GRID * GRID} points around shell {ACTIVE_SHELL} ({L_LABELS[ACTIVE_SHELL]}) ===")
Z = tempered_objective(grid_params).reshape(GA.shape)

best_flat = int(Z.argmin())
grid_min  = array([GA.ravel()[best_flat], GB.ravel()[best_flat]])
print(f"  grid min E = {Z.min():.10f} Eh  at params {grid_min}")

# ── optional optimizer path ──
path = None
if RUN_OPTIMIZER:
    print("=== Newton_6 ===")
    opt        = Newton_6(tempered_objective, x0, trust_radius=TRUST_RADIUS, trust_radius_max=0.2)
    found, f_e = opt.minimize(max_steps=MAX_STEPS, verbose=True)
    path       = array([p for p, _ in opt.history], dtype=float64)
    print(f"  Newton found E = {f_e:.10f} Eh at {found}  ({len(opt.history) - 1} acc, {opt.reject_count} rej, {opt.eval_count} evals)")

# ── plot ──
fig, ax = plt.subplots(figsize=(8.0, 7.0))
levels  = linspace(float(Z.min()), float(median(Z)), 40)
cf      = ax.contourf(GA, GB, Z, levels=levels, cmap="viridis", extend="max")
fig.colorbar(cf, ax=ax, label="E (Eh)")

ax.plot(x0[0], x0[1], "s", color="#00cd6c", markeredgecolor="black", markersize=12.0, label="start")
ax.plot(grid_min[0], grid_min[1], "*", color="#ff3333", markeredgecolor="black", markersize=20.0, label="grid min")
if path is not None:
    ax.plot(path[:, 0], path[:, 1], "-o", color="white", markeredgecolor="black",
            markeredgewidth=0.8, linewidth=2.0, markersize=6.0, label="Newton path")
    ax.plot(path[-1, 0], path[-1, 1], "X", color="black", markeredgecolor="white", markersize=13.0, label="Newton found")

ax.set_xlabel("a0  (log scale param)")
ax.set_ylabel("a1  (range param)")
ax.set_title(f"Shell {ACTIVE_SHELL} ({L_LABELS[ACTIVE_SHELL]}) 2D tempering energy surface")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
fig.tight_layout()

png_path = SUBMIT_DIR / "scan.png"
npz_path = SUBMIT_DIR / "scan.npz"
fig.savefig(png_path, dpi=130)
save_dict = {"a0s": a0s, "a1s": a1s, "Z": Z, "x0": x0, "grid_min": grid_min}
if path is not None:
    save_dict["path"] = path
savez(npz_path, **save_dict)
print(f"saved {png_path}")
print(f"saved {npz_path}")

plt.show()
