import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from threading import Lock, Thread

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.cma_shell_opt import Shell_Optimization

# ─── CLI + directories ────────────────────────────────────────────────────────

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, default=None)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = ((_args.work_dir if _args.work_dir is not None else SUBMIT_DIR) / "Optimization").resolve()
sys.path.insert(0, str(SUBMIT_DIR))

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

# --- input files (looked up in the submit dir) ---
EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # contracted frozen shells
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

# --- per-shell optimization ---
OPTIMIZE_FLAGS         = [1, 1, 1]     # 1 = optimize, 0 = freeze; may be shorter than n_shells
GENERATION_SIZE        = [10, 10, 10]  # int, or one entry per OPTIMIZE_FLAGS entry
THREADS_PER_SHELL      = [2, 2, 2]     # int, or one entry per OPTIMIZE_FLAGS entry
SIGMA                  = 0.01
MAX_GENERATIONS        = 50   # target: run until every shell has reached this many gens
GEN_CEILING_MULTIPLIER = 5    # hard per-shell ceiling = MAX_GENERATIONS * this; a fast shell
                              #   may run ahead up to the ceiling while slower shells catch up
                              #   (assumes no shell is more than this many times slower)
USE_TEMPERING          = False
N_TEMPERING_PARAMS     = 6

# --- global (fully-uncontracted) evaluations ---
THREADS_GLOBAL           = 2   # max concurrent fully-uncontracted eval jobs in flight
GLOBAL_EVAL_WARMUP_GENS  = 10  # all shells must reach this many gens before the first global eval
GLOBAL_EVAL_SPACING_GENS = 2   # all shells must advance this many gens between global evals

# --- cross-shell coupling ---
# Feed the converged global contraction back to every shell optimizer as its new
# root, so each shell sees the other shells' improvements instead of optimizing
# against a frozen initial guess.
ENABLE_CROSS_SHELL      = True
CROSS_SHELL_WARMUP_GENS = 20   # all shells must reach this many gens before coupling starts

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

GEN_CEILING = MAX_GENERATIONS * GEN_CEILING_MULTIPLIER   # hard per-shell generation cap


def per_shell(value, n: int, name: str) -> list:
    """Broadcast an int, or validate a list, to one value per shell."""
    if isinstance(value, int):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} length {len(value)} must match OPTIMIZE_FLAGS length {n}")
    return list(value)


# ─── stage inputs + normalise config ──────────────────────────────────────────

START_DIR = WORK_DIR / "Start"
START_DIR.mkdir(parents=True, exist_ok=True)


def stage(name: str) -> Path:
    """Copy an input (start) file from the submit dir into the run's Start dir."""
    dst = START_DIR / name
    shutil.copy(SUBMIT_DIR / name, dst)
    return dst

exp_path      = stage(EXPO_FILE)
template      = stage(TEMPLATE_CONT)
template_full = stage(TEMPLATE_FULL)
run_scr       = stage(RUN_SCRIPT)
extract_scr   = stage(EXTRACT_SCRIPT)

_n_flags       = len(OPTIMIZE_FLAGS)
_gen_sizes     = per_shell(GENERATION_SIZE,   _n_flags, "GENERATION_SIZE")
_threads_shell = per_shell(THREADS_PER_SHELL, _n_flags, "THREADS_PER_SHELL")

# Cross-shell coupling only ever runs inside a global eval, which cannot start
# before its own warmup. A coupling warmup below the global-eval warmup would be
# silently clamped up to it, so reject that combination rather than mislead.
if ENABLE_CROSS_SHELL and CROSS_SHELL_WARMUP_GENS < GLOBAL_EVAL_WARMUP_GENS:
    raise ValueError(
        f"CROSS_SHELL_WARMUP_GENS ({CROSS_SHELL_WARMUP_GENS}) must be >= GLOBAL_EVAL_WARMUP_GENS "
        f"({GLOBAL_EVAL_WARMUP_GENS}); coupling happens during a global eval and cannot "
        f"fire before global evals begin."
    )

# ─── load basis and validate flags ────────────────────────────────────────────

basis    = Exponent_Set.from_file(exp_path)
n_shells = len(basis.exponents)

if _n_flags > n_shells:
    raise ValueError(f"OPTIMIZE_FLAGS has {_n_flags} entries but basis only has {n_shells} shells")

flags = list(OPTIMIZE_FLAGS) + [0] * (n_shells - _n_flags)

cfg = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template,      cfg)
full_objective = Ground_Energy_Objective(template_full, cfg)

# ─── initial uncontracted run ─────────────────────────────────────────────────

init_source = basis.copy(no_energy=True)
init_source.uncontract_all()   # guarantee a fully uncontracted starting point regardless of input .expo
init_uncontracted = evaluate_initial(init_source, full_objective, WORK_DIR / "initial_uncontracted", threads=1)

if init_uncontracted.resulting_contraction is None:
    raise RuntimeError("Initial uncontracted run produced no contraction.")

print(f"Uncontracted energy : {init_uncontracted.energy:.10f} Eh")
print("Contraction sizes   :")
rc = init_uncontracted.resulting_contraction
for i in range(len(rc)):
    print(f"  shell {i} ({L_LABELS[i]}): {rc[i].shape[0]} <- {rc[i].shape[1]}")

contracted_base = init_uncontracted.copy(no_energy=True)
contracted_base.change_contraction(init_uncontracted.resulting_contraction)

# ─── initialise per-shell optimizers ──────────────────────────────────────────

optimizers: dict[int, Shell_Optimization] = {}

for shell_idx in range(_n_flags):
    if flags[shell_idx] == 0:
        continue

    lbl   = L_LABELS[shell_idx]
    n_exp = len(basis.exponents[shell_idx])

    if n_exp <= 1:
        print(f"  shell {shell_idx} ({lbl}): only {n_exp} exponent(s), skipping")
        continue

    shell_n_tempering = min(N_TEMPERING_PARAMS, n_exp) if USE_TEMPERING else N_TEMPERING_PARAMS

    shell_start = contracted_base.copy(no_energy=True)
    shell_start.uncontract_shell(shell_idx)

    init_result = evaluate_initial(
        shell_start, objective, WORK_DIR / f"initial_shell_{shell_idx}",
        threads=1, contract_frozen_shells=True,   # single job — extra threads would be idle
    )
    print(f"  shell {shell_idx} ({lbl}) initial energy : {init_result.energy:.10f} Eh")

    optimizers[shell_idx] = Shell_Optimization(
        init_result,
        float(init_result.energy),
        objective,
        work_dir               = WORK_DIR / f"cma_shell_{shell_idx}",
        generation_size        = _gen_sizes[shell_idx],
        sigma                  = SIGMA,
        max_generations        = GEN_CEILING,
        active_shell           = shell_idx,
        overwrite              = True,
        logging                = True,
        contract_frozen_shells = True,
        use_tempering          = USE_TEMPERING,
        n_tempering_params     = shell_n_tempering,
    )

if not optimizers:
    raise RuntimeError("No shells to optimize after filtering.")

print(f"\nOptimizing shells : {sorted(optimizers)}")
print(f"Target gens       : {MAX_GENERATIONS} (stop once all shells reach this)")
print(f"Per-shell ceiling : {GEN_CEILING} gens")
print(f"Warmup            : {GLOBAL_EVAL_WARMUP_GENS} gens before first global eval")
print(f"Spacing           : {GLOBAL_EVAL_SPACING_GENS} gens between triggers")
print(f"Max concurrent    : {THREADS_GLOBAL} global evals\n")

# ─── start all optimizers ─────────────────────────────────────────────────────

t0 = time.time()

for shell_idx in sorted(optimizers):
    optimizers[shell_idx].start(threads=_threads_shell[shell_idx])

# ─── global eval infrastructure ───────────────────────────────────────────────

GLOBAL_EVAL_DIR = WORK_DIR / "global_evals"
GLOBAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)

E0         = float(init_uncontracted.energy)
best_lock  = Lock()
best_state = {
    "best_energy": E0,
    "best_exp":    init_uncontracted.copy(no_energy=True),
}
best_state["best_exp"].energy = E0

csv_f = open(SUBMIT_DIR / "global_trace.csv", "w", newline="")
log_f = open(SUBMIT_DIR / "global.log", "w")
csv_w = csv.writer(csv_f)

csv_w.writerow(
    ["eval_idx", "time_sec", "global_energy", "delta_e"]
    + [f"shell_{idx}_gen_at_trigger" for idx in sorted(optimizers)]
)
csv_f.flush()


def collect_combined() -> Exponent_Set:
    """Fully-uncontracted basis with each optimized shell's current best exponents."""
    combined = contracted_base.copy(no_energy=True)
    combined.uncontract_all()
    for idx in sorted(optimizers):
        state = optimizers[idx].get_state()
        if state["best_exp"] is not None:
            combined.set_shell_exponents(idx, state["best_exp"].exponents[idx])
    return combined


def global_eval_worker(eval_idx, snapshot, trigger_gens):
    eval_dir = GLOBAL_EVAL_DIR / f"eval_{eval_idx:04d}"
    results  = full_objective.evaluate_batch([snapshot], work_dir=eval_dir, threads=1)
    energy   = float(results[0].energy)
    elapsed  = time.time() - t0
    delta_e  = energy - E0

    with best_lock:
        if energy < best_state["best_energy"]:
            best_state["best_energy"]     = energy
            best_state["best_exp"]        = results[0].copy(no_energy=True)
            best_state["best_exp"].energy = energy
            best_state["best_exp"].save(SUBMIT_DIR, "best", overwrite=True)
        best_so_far = best_state["best_energy"]

        line = (
            f"[GlobalEval {eval_idx:4d}] T {elapsed:.1f}s | "
            f"E {energy:.10f} | ΔE {delta_e:+.8f} | BestE {best_so_far:.10f}"
        )
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

        csv_w.writerow(
            [eval_idx, elapsed, energy, delta_e]
            + [trigger_gens.get(idx, -1) for idx in sorted(optimizers)]
        )
        csv_f.flush()

    # cross-shell coupling: feed the converged contraction back to every shell
    # optimizer as a new root
    if (
        ENABLE_CROSS_SHELL
        and results[0].resulting_contraction is not None
        and all(optimizers[idx].generation >= CROSS_SHELL_WARMUP_GENS - 1 for idx in optimizers)
    ):
        shared_exp = results[0].copy(no_energy=True)
        shared_exp.change_contraction(results[0].resulting_contraction)
        for idx in optimizers:
            optimizers[idx].update_root_exponent(shared_exp)


# ─── poll and trigger loop ────────────────────────────────────────────────────

def all_reached_target() -> bool:
    return all(optimizers[idx].generation >= MAX_GENERATIONS - 1 for idx in optimizers)

def any_hit_ceiling() -> bool:
    return any(optimizers[idx].generation >= GEN_CEILING - 1 for idx in optimizers)

def abort_if_crashed() -> None:
    """A shell optimizer crashing should be rare; if one does, stop everything and
    re-raise so the whole run fails loudly rather than limping to the ceiling."""
    for idx in optimizers:
        exc = optimizers[idx].exception
        if exc is not None:
            for other in optimizers:
                optimizers[other].stop(wait=False)
            raise RuntimeError(f"Shell {idx} ({L_LABELS[idx]}) optimizer crashed; aborting run.") from exc

active_evals      = []
trigger_idx       = 0
first_triggered   = False
last_trigger_gens = {idx: -1 for idx in optimizers}

# Run until every shell has reached the target. Shells that get there first keep
# optimizing (up to the ceiling) so global evals keep firing for the whole run;
# they are stopped once the slowest shell catches up. If any shell runs all the way
# to the ceiling, the ">5x slower" assumption has broken — bail out immediately.
while any(optimizers[idx].is_running for idx in optimizers) and not all_reached_target() and not any_hit_ceiling():
    abort_if_crashed()
    active_evals[:] = [t for t in active_evals if t.is_alive()]

    current_gens = {idx: optimizers[idx].generation for idx in optimizers}

    slots_free  = len(active_evals) < THREADS_GLOBAL
    all_started = all(current_gens[idx] >= 0 for idx in optimizers)

    if slots_free and all_started:
        if not first_triggered:
            ready = all(current_gens[idx] >= GLOBAL_EVAL_WARMUP_GENS - 1 for idx in optimizers)
        else:
            ready = all(
                current_gens[idx] >= last_trigger_gens[idx] + GLOBAL_EVAL_SPACING_GENS
                for idx in optimizers
            )

        if ready:
            t = Thread(
                target=global_eval_worker,
                args=(trigger_idx, collect_combined(), dict(current_gens)),
                daemon=True,
            )
            t.start()
            active_evals.append(t)

            last_trigger_gens = dict(current_gens)
            first_triggered   = True
            trigger_idx      += 1

    time.sleep(1)

# ─── stop stragglers and drain remaining global evals ─────────────────────────

abort_if_crashed()   # catch a crash that ended the loop (or one where it never ran)

if any_hit_ceiling() and not all_reached_target():
    stalled = [idx for idx in optimizers if optimizers[idx].generation >= GEN_CEILING - 1]
    print(f"\n[WARNING] shell(s) {stalled} hit the {GEN_CEILING}-gen ceiling before all shells "
          f"reached the target ({MAX_GENERATIONS}); exiting early.")

for idx in optimizers:
    optimizers[idx].stop(wait=False)   # target reached (or ceiling hit); halt any shell still running
for idx in optimizers:
    optimizers[idx].wait()
for t in active_evals:
    t.join()

# ─── final global eval ────────────────────────────────────────────────────────

final_gens = {idx: optimizers[idx].get_state()["generation"] for idx in sorted(optimizers)}
global_eval_worker(trigger_idx, collect_combined(), final_gens)

csv_f.close()
log_f.close()

print(f"\nBest global energy : {best_state['best_energy']:.10f} Eh"
      f"  (ΔE = {best_state['best_energy'] - E0:+.10f})")
