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
from evo_opt.newton_6 import Newton_6, minimize_whitened

_arg_parser = argparse.ArgumentParser(description="Newton_6 feature test: optimise one shell's 2D tempering params via real MOLCAS jobs")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Newton6_Test").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

ACTIVE_SHELL      = 0        # which shell to optimise
USE_CONTRACTION   = True     # bootstrap GENANO contraction, keep frozen shells contracted

TRUST_RADIUS      = 0.3      # initial trust radius in tempering-param space
TRUST_RADIUS_MIN  = 1e-3     # stop once it collapses to here
STENCIL_MIN       = 0.05     # stencil never shrinks below this (keeps model above noise floor)
PURE_NEWTON       = False    # True: skip the trust region, full Newton step every time
WHITEN            = True     # optimise in Hessian-whitened coords (decouples the valley)
PROBE_RADIUS      = 0.15     # stencil for the one-off Hessian probe (whitening only)
STALL_TOL         = 1e-5     # stop when gain over last 5 steps < this Eh (0 = off)
MAX_STEPS         = 30
THREADS           = 6        # 6 parallel jobs = one lump of 6 at once

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

# ── bootstrap contraction on the starting basis (frozen shells stay contracted) ──
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

# ── fit the shell's starting exponents to a 2-param tempering codec ──
n_active = len(base.exponents[ACTIVE_SHELL])
codec    = from_registry("polynomial", m=2, n=n_active)
x0       = codec.encode(base.exponents[ACTIVE_SHELL])

print(f"=== optimise shell {ACTIVE_SHELL} ({L_LABELS[ACTIVE_SHELL]}), N={n_active} exponents ===")
print(f"  start params: {x0}")

# ── objective wrapper: (m,2) params -> (m,) energies, one lump = one batch ──
def tempered_objective(params_batch):
    exp_objects = []
    for i in range(len(params_batch)):
        work = base.copy(no_energy=True)
        work.apply_params(ACTIVE_SHELL, codec, params_batch[i], n=n_active)
        if not USE_CONTRACTION:
            work.uncontract_all()
        exp_objects.append(work)
    results = energy_objective.evaluate_batch(
        exp_objects,
        work_dir  = WORK_DIR / "batch",
        threads   = THREADS,
        overwrite = True,
    )
    return array([float(r.energy) for r in results], dtype=float64)


if WHITEN:
    best, best_e, opt, _ = minimize_whitened(
        tempered_objective, x0,
        probe_radius=PROBE_RADIUS, trust_radius=TRUST_RADIUS, trust_radius_min=TRUST_RADIUS_MIN,
        stencil_min=STENCIL_MIN, pure_newton=PURE_NEWTON,
        max_steps=MAX_STEPS, stall_tol=STALL_TOL, verbose=True,
    )
else:
    opt          = Newton_6(tempered_objective, x0, trust_radius=TRUST_RADIUS, trust_radius_min=TRUST_RADIUS_MIN,
                            stencil_min=STENCIL_MIN, pure_newton=PURE_NEWTON)
    best, best_e = opt.minimize(max_steps=MAX_STEPS, stall_tol=STALL_TOL, verbose=True)

start_e   = opt.history[0][1]
final_exp = codec.decode(best, n_active)

print()
print("=== result ===")
print(f"  start  E : {start_e:.10f} Eh")
print(f"  final  E : {best_e:.10f} Eh")
print(f"  gain     : {(start_e - best_e) * 1e3:.4f} mEh")
print(f"  params   : {best}")
print(f"  exponents: {final_exp}")
print(f"  {len(opt.history) - 1} accepted, {opt.reject_count} rejected, {opt.refit_count} refits, {opt.eval_count} evals")
