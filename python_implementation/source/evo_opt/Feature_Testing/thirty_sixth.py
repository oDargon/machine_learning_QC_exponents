import sys
import csv
import time
import shutil
import argparse
from pathlib import Path
from numpy import array, float64, log, abs as npabs

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS, CHEMICAL_ACCURACY
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.pipeline_core.optimize import Optimize_Config, run_optimize

_arg_parser = argparse.ArgumentParser(description="Full basis optimization, no contraction, under three schedulers: shells uncoupled, shells coupled through the global eval, and cyclic one-at-a-time")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
BASE_WORK  = _args.work_dir.resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # required by Optimize_Config even with contraction off
TEMPLATE_FULL  = "temp_full.inp"    # every job in this test runs through this one
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

# Three FULL basis optimizations. Expensive — size it before launching.
ARMS = ["uncoupled", "coupled", "cyclic"]

OPTIMIZE_FLAGS        = [1, 1, 1]       # 1 = optimize, 0 = freeze
OPT_GEN_SIZE          = [6, 6, 6]       # int, or one entry per flag
OPT_THREADS_PER_SHELL = [6, 6, 6]       # int, or one entry per flag
OPT_SIGMA             = 0.1
USE_TEMPERING         = True
N_TEMPERING_PARAMS    = 6
SEED                  = 42              # the SAME seed for all three arms

# ── joint arms (uncoupled / coupled) ──
OPT_MAX_GENS             = 150   # target gens per shell
GEN_CEILING_MULTIPLIER   = 2     # headroom so a slow shell can catch up. Do NOT set 1:
                                 # the run ends the moment the FASTEST shell hits the
                                 # ceiling, which truncates the slower shells.
THREADS_GLOBAL           = 2
GLOBAL_EVAL_WARMUP_GENS  = 10
GLOBAL_EVAL_SPACING_GENS = 2
CROSS_SHELL_WARMUP_GENS  = 20    # coupled arm only; must be >= GLOBAL_EVAL_WARMUP_GENS

# ── cyclic arm ──
# CYCLIC_CYCLES * CYCLIC_GENS_PER_VISIT is cyclic's gens-per-shell budget. Matching it to
# OPT_MAX_GENS equalises GENERATIONS, not jobs — the schedulers spend different numbers of
# full evals, so compare on jobs using the traces rather than trusting the gen budget.
CYCLIC_CYCLES         = 5
CYCLIC_GENS_PER_VISIT = 30       # 5 * 30 = 150 = OPT_MAX_GENS
CYCLIC_WARM_RESTART   = True
CYCLIC_SHELL_ORDER    = None     # None -> ascending l

EARLY_STOP        = False        # off: an early stop fires at different points per arm and
EARLY_STOP_WINDOW = 5            # would end the arms at incomparable places
EARLY_STOP_TOL    = 1e-5

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

RESULTS_DIR = SUBMIT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_f = open(RESULTS_DIR / "schedulers.log", "w")


def emit(msg):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()


def base_cfg(arm):
    """Identical config for every arm except the three scheduler fields and the output
    paths. Keeping this in one place is what makes the comparison about the scheduler."""
    return dict(
        submit_dir               = SUBMIT_DIR,
        work_dir                 = BASE_WORK / arm,
        expo_file                = EXPO_FILE,
        template_cont            = TEMPLATE_CONT,
        template_full            = TEMPLATE_FULL,
        run_script               = RUN_SCRIPT,
        extract_script           = EXTRACT_SCRIPT,
        optimize_flags           = OPTIMIZE_FLAGS,
        generation_size          = OPT_GEN_SIZE,
        threads_per_shell        = OPT_THREADS_PER_SHELL,
        sigma                    = OPT_SIGMA,
        max_generations          = OPT_MAX_GENS,
        gen_ceiling_multiplier   = GEN_CEILING_MULTIPLIER,
        use_contraction          = False,
        use_tempering            = USE_TEMPERING,
        n_tempering_params       = N_TEMPERING_PARAMS,
        threads_global           = THREADS_GLOBAL,
        global_eval_warmup_gens  = GLOBAL_EVAL_WARMUP_GENS,
        global_eval_spacing_gens = GLOBAL_EVAL_SPACING_GENS,
        early_stop               = EARLY_STOP,
        early_stop_window        = EARLY_STOP_WINDOW,
        early_stop_tol           = EARLY_STOP_TOL,
        seed                     = SEED,
        results_subdir           = arm,
    )


ARM_SPEC = {
    "uncoupled": dict(mode="joint",  enable_cross_shell=False),
    "coupled":   dict(mode="joint",  enable_cross_shell=True,
                      cross_shell_warmup_gens=CROSS_SHELL_WARMUP_GENS),
    "cyclic":    dict(mode="cyclic", enable_cross_shell=False,
                      cyclic_cycles=CYCLIC_CYCLES,
                      cyclic_gens_per_visit=CYCLIC_GENS_PER_VISIT,
                      cyclic_warm_restart=CYCLIC_WARM_RESTART,
                      cyclic_shell_order=CYCLIC_SHELL_ORDER),
}

for a in range(len(ARMS)):
    if ARMS[a] not in ARM_SPEC:
        raise SystemExit(f"unknown arm {ARMS[a]!r}; known arms are {sorted(ARM_SPEC)}")

# ─── does the coupled arm have anything to couple? ────────────────────────────
# Cross-shell coupling only fires when the global eval returns a contraction, because it
# propagates the shared basis via change_contraction. With contraction off the contraction
# itself is stripped by the shells, but the EXPONENTS still propagate — which is the point.
# If template_full produces no contraction the coupling silently never fires and the
# coupled arm becomes identical to the uncoupled one, so check once up front.

emit("=== precheck: will cross-shell coupling fire? ===")
probe_dir = BASE_WORK / "precheck"
probe_dir.mkdir(parents=True, exist_ok=True)
for n in (EXPO_FILE, TEMPLATE_FULL, RUN_SCRIPT, EXTRACT_SCRIPT):
    shutil.copy(SUBMIT_DIR / n, probe_dir / n)
probe_cfg = Job_Manager_Config(
    executor_type        = Executor_Type.LOCAL_BASH,
    execution_script     = probe_dir / RUN_SCRIPT,
    extraction_script    = probe_dir / EXTRACT_SCRIPT,
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)
probe_basis = Exponent_Set.from_file(probe_dir / EXPO_FILE)
probe_basis.uncontract_all()
probe = evaluate_initial(probe_basis, Ground_Energy_Objective(probe_dir / TEMPLATE_FULL, probe_cfg),
                         probe_dir, threads=1, subdir_name="probe", contract_frozen_shells=False)
if probe.resulting_contraction is None:
    emit("  [WARNING] the fully-uncontracted template returns NO contraction, so cross-shell")
    emit("            coupling can never fire and the 'coupled' arm will be identical to")
    emit("            'uncoupled'. Fix the template or drop the coupled arm.")
else:
    emit(f"  ok: template returns a contraction, coupling can fire after "
         f"{CROSS_SHELL_WARMUP_GENS} gens")
emit(f"  E (fully uncontracted start): {float(probe.energy):.10f} Eh")
emit("")

emit(f"arms            : {ARMS}")
emit(f"shells          : {[i for i in range(len(OPTIMIZE_FLAGS)) if OPTIMIZE_FLAGS[i]]}")
emit(f"contraction     : OFF for every arm (all jobs through {TEMPLATE_FULL})")
emit(f"parametrization : {'tempering M=' + str(N_TEMPERING_PARAMS) if USE_TEMPERING else 'raw log-exponents'}")
emit(f"seed            : {SEED} (shared by all arms)")
emit(f"gens per shell  : joint {OPT_MAX_GENS} (ceiling x{GEN_CEILING_MULTIPLIER})   "
     f"cyclic {CYCLIC_CYCLES} x {CYCLIC_GENS_PER_VISIT} = {CYCLIC_CYCLES * CYCLIC_GENS_PER_VISIT}")
emit(f"early stop      : {'on' if EARLY_STOP else 'off (arms must end at comparable places)'}")
emit("")

SUMMARY_HEADER = ["arm", "mode", "cross_shell", "E0", "E_best", "dE",
                  "dE_chem_acc", "wall_s", "results_subdir"]
sum_f = open(RESULTS_DIR / "schedulers.csv", "w", newline="")
sum_w = csv.writer(sum_f)
sum_w.writerow([f"#META atom={probe_basis.atom_name} arms={ARMS} seed={SEED} "
                f"use_contraction=False tempering={USE_TEMPERING} M={N_TEMPERING_PARAMS} "
                f"opt_max_gens={OPT_MAX_GENS} cyclic={CYCLIC_CYCLES}x{CYCLIC_GENS_PER_VISIT} "
                f"chem_acc={CHEMICAL_ACCURACY:.6e}"])
sum_w.writerow(SUMMARY_HEADER)
sum_f.flush()

results = {}

for a in range(len(ARMS)):
    arm  = ARMS[a]
    spec = ARM_SPEC[arm]

    emit("#" * 78)
    emit(f"ARM {a + 1}/{len(ARMS)}: {arm}   (mode={spec['mode']}, "
         f"cross_shell={spec['enable_cross_shell']})")
    emit("#" * 78)

    cfg = Optimize_Config(**base_cfg(arm), **spec)

    t0 = time.time()
    best_exp, best_energy, e0 = run_optimize(cfg)
    wall = time.time() - t0

    best_exp.save(RESULTS_DIR, f"best_{arm}", overwrite=True)
    results[arm] = (best_exp, float(best_energy), float(e0), wall)

    emit("")
    emit(f"  {arm}: E0 {e0:.10f} -> best {best_energy:.10f}  "
         f"(dE {best_energy - e0:+.3e} Eh = {(best_energy - e0) / CHEMICAL_ACCURACY:+.3f} "
         f"x chemical accuracy)  {wall:.1f}s")
    emit("")

    sum_w.writerow([arm, spec["mode"], int(spec["enable_cross_shell"]),
                    f"{e0:.10f}", f"{best_energy:.10f}", f"{best_energy - e0:+.10f}",
                    f"{(best_energy - e0) / CHEMICAL_ACCURACY:+.6f}",
                    f"{wall:.1f}", arm])
    sum_f.flush()

# ─── cross-arm comparison ─────────────────────────────────────────────────────

emit("#" * 78)
emit("COMPARISON")
emit("#" * 78)
emit(f"  {'arm':<12}{'E_best':>18}{'dE (Eh)':>13}{'dE/chem':>10}{'wall (s)':>11}")
for a in range(len(ARMS)):
    arm = ARMS[a]
    _, e_best, e0, wall = results[arm]
    emit(f"  {arm:<12}{e_best:>18.10f}{e_best - e0:>13.3e}{(e_best - e0) / CHEMICAL_ACCURACY:>10.3f}{wall:>11.1f}")

ref_arm   = ARMS[0]
ref_best  = results[ref_arm][1]
emit("")
emit(f"  relative to '{ref_arm}':")
for a in range(1, len(ARMS)):
    arm = ARMS[a]
    e_best = results[arm][1]
    d      = e_best - ref_best
    emit(f"    {arm:<12} {d:+.3e} Eh ({d / CHEMICAL_ACCURACY:+.4f} x chem acc)  "
         f"{'better' if d < 0 else 'worse' if d > 0 else 'identical'}")

# how far apart are the bases themselves, shell by shell
emit("")
emit(f"  max|dln(alpha)| per shell, each arm vs '{ref_arm}':")
ref_exp = results[ref_arm][0]
for l in range(len(ref_exp.exponents)):
    row = f"    shell {l} ({L_LABELS[l]}): "
    for a in range(1, len(ARMS)):
        arm   = ARMS[a]
        other = results[arm][0]
        if len(other.exponents[l]) != len(ref_exp.exponents[l]):
            row += f"{arm}=n/a(size)  "
            continue
        d = float(npabs(log(array(other.exponents[l], dtype=float64))
                        - log(array(ref_exp.exponents[l], dtype=float64))).max())
        row += f"{arm}={d:.3e}  "
    emit(row)

if "uncoupled" in results and "coupled" in results:
    if abs(results["coupled"][1] - results["uncoupled"][1]) < 1e-12:
        emit("")
        emit("  [WARNING] 'coupled' and 'uncoupled' reached bit-identical energies. Coupling")
        emit("            probably never fired — check CROSS_SHELL_WARMUP_GENS against the")
        emit("            generations actually reached, and the precheck above.")

sum_f.close()
emit("")
emit(f"saved {RESULTS_DIR / 'schedulers.csv'}   (one row per arm)")
emit(f"saved {RESULTS_DIR / 'schedulers.log'}")
emit(f"saved best_<arm>.expo in {RESULTS_DIR}")
emit(f"each arm's own trace/log/best live in {RESULTS_DIR}/<arm>/")
emit("")
emit("Compare arms on JOBS, not generations: the three schedulers spend different numbers")
emit("of full evals, so an equal gens-per-shell budget is not an equal job budget. Each")
emit("arm's <arm>/global_trace.csv carries jobs_total against best_energy for that.")
log_f.close()
