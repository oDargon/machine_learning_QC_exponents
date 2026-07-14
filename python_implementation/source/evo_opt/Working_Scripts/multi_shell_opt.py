import sys
import shutil
import argparse
import time
import csv
from pathlib import Path
from threading import Thread, Lock

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
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.cma_shell_opt import Shell_Optimization

# ─── configuration ────────────────────────────────────────────────────────────

OPTIMIZE_FLAGS         = [1, 1, 1]  # 1 = optimize, 0 = freeze; may be shorter than n_shells
GENERATION_SIZE        = [10, 10, 10]  # int or list (one per OPTIMIZE_FLAGS entry)
THREADS_PER_SHELL      = [2,  2,  2]  # int or list (one per OPTIMIZE_FLAGS entry)
SIGMA                  = 0.01
MAX_GENERATIONS        = 50
USE_TEMPERING          = False
N_TEMPERING_PARAMS     = 6

THREADS_GLOBAL            = 2    # max concurrent fully-uncontracted eval jobs in flight
GLOBAL_EVAL_WARMUP_GENS   = 10   # all shells must reach this many gens before first global eval
GLOBAL_EVAL_SPACING_GENS  = 2    # all shells must advance this many gens since last trigger

ENABLE_MIXING             = True
MIXING_WARMUP_GENS        = 20   # all shells must reach this many gens before mixing starts

# ─── normalise per-shell lists ────────────────────────────────────────────────

_n_flags = len(OPTIMIZE_FLAGS)

if isinstance(GENERATION_SIZE, int):
    _gen_sizes = [GENERATION_SIZE] * _n_flags
else:
    if len(GENERATION_SIZE) != _n_flags:
        raise ValueError(f"GENERATION_SIZE list length {len(GENERATION_SIZE)} must match OPTIMIZE_FLAGS length {_n_flags}")
    _gen_sizes = list(GENERATION_SIZE)

if isinstance(THREADS_PER_SHELL, int):
    _threads_shell = [THREADS_PER_SHELL] * _n_flags
else:
    if len(THREADS_PER_SHELL) != _n_flags:
        raise ValueError(f"THREADS_PER_SHELL list length {len(THREADS_PER_SHELL)} must match OPTIMIZE_FLAGS length {_n_flags}")
    _threads_shell = list(THREADS_PER_SHELL)

# ─── paths ────────────────────────────────────────────────────────────────────

exp_path      = SUBMIT_DIR / "Si.expo"
template      = SUBMIT_DIR / "temp_cont.inp"
template_full = SUBMIT_DIR / "temp_full.inp"
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

# ─── load basis and validate flags ────────────────────────────────────────────

exp      = Exponent_Set.from_file(exp_path)
n_shells = len(exp.exponents)

if _n_flags > n_shells:
    raise ValueError(
        f"OPTIMIZE_FLAGS has {_n_flags} entries but basis only has {n_shells} shells"
    )

flags = list(OPTIMIZE_FLAGS) + [0] * (n_shells - _n_flags)

cfg            = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template,      cfg)
full_objective = Ground_Energy_Objective(template_full, cfg)

# ─── initial uncontracted run ─────────────────────────────────────────────────

init_uncontracted = evaluate_initial(exp, full_objective, WORK_DIR / "initial_uncontracted", threads=THREADS_GLOBAL)

if init_uncontracted.resulting_contraction is None:
    raise RuntimeError("Initial uncontracted run produced no contraction.")

L_LABELS = ["s", "p", "d", "f", "g", "h"]
print(f"Uncontracted energy : {init_uncontracted.energy:.10f} Eh")
print("Contraction sizes   :")
rc = init_uncontracted.resulting_contraction
for i in range(len(rc)):
    lbl = L_LABELS[i] if i < len(L_LABELS) else str(i)
    print(f"  shell {i} ({lbl}): {rc[i].shape[0]} <- {rc[i].shape[1]}")

contracted_base = init_uncontracted.copy(no_energy=True)
contracted_base.change_contraction(init_uncontracted.resulting_contraction)

# ─── initialise per-shell optimizers ──────────────────────────────────────────

optimizers: dict[int, Shell_Optimization] = {}

for shell_idx in range(_n_flags):
    if flags[shell_idx] == 0:
        continue

    lbl   = L_LABELS[shell_idx] if shell_idx < len(L_LABELS) else str(shell_idx)
    n_exp = len(exp.exponents[shell_idx])

    if n_exp <= 1:
        print(f"  shell {shell_idx} ({lbl}): only {n_exp} exponent(s), skipping")
        continue

    shell_n_tempering = min(N_TEMPERING_PARAMS, n_exp) if USE_TEMPERING else N_TEMPERING_PARAMS

    shell_start = contracted_base.copy(no_energy=True)
    shell_start.uncontract_shell(shell_idx)

    init_result = evaluate_initial(
        shell_start, objective, WORK_DIR / f"initial_shell_{shell_idx}",
        threads=_threads_shell[shell_idx], contract_frozen_shells=True,
    )
    print(f"  shell {shell_idx} ({lbl}) initial energy : {init_result.energy:.10f} Eh")

    optimizers[shell_idx] = Shell_Optimization(
        init_result,
        float(init_result.energy),
        objective,
        work_dir               = WORK_DIR / f"cma_shell_{shell_idx}",
        generation_size        = _gen_sizes[shell_idx],
        sigma                  = SIGMA,
        max_generations        = MAX_GENERATIONS,
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

_E0      = float(init_uncontracted.energy)
_gs_lock = Lock()
_gs      = {
    "best_energy": _E0,
    "best_exp":    init_uncontracted.copy(no_energy=True),
}
_gs["best_exp"].energy = _E0

csv_path = SUBMIT_DIR / "global_trace.csv"
log_path = SUBMIT_DIR / "global.log"
csv_f    = open(csv_path, "w", newline="")
log_f    = open(log_path, "w")
csv_w    = csv.writer(csv_f)

_header = ["eval_idx", "time_sec", "global_energy", "delta_e"]
for _idx in sorted(optimizers):
    _header.append(f"shell_{_idx}_gen_at_trigger")
csv_w.writerow(_header)
csv_f.flush()


def _global_eval_worker(eval_idx, snapshot, trigger_gens):
    eval_dir = GLOBAL_EVAL_DIR / f"eval_{eval_idx:04d}"
    results  = full_objective.evaluate_batch([snapshot], work_dir=eval_dir, threads=1)
    energy   = float(results[0].energy)
    elapsed  = time.time() - t0
    delta_e  = energy - _E0

    with _gs_lock:
        if energy < _gs["best_energy"]:
            _gs["best_energy"]       = energy
            _gs["best_exp"]          = results[0].copy(no_energy=True)
            _gs["best_exp"].energy   = energy
            _gs["best_exp"].save(SUBMIT_DIR, "best", overwrite=True)
        best_so_far = _gs["best_energy"]
        line = (
            f"[GlobalEval {eval_idx:4d}] T {elapsed:.1f}s | "
            f"E {energy:.10f} | ΔE {delta_e:+.8f} | BestE {best_so_far:.10f}"
        )
        print(line)
        log_f.write(line + "\n")
        log_f.flush()
        row = [eval_idx, elapsed, energy, delta_e]
        for _idx in sorted(optimizers):
            row.append(trigger_gens.get(_idx, -1))
        csv_w.writerow(row)
        csv_f.flush()

    if (
        ENABLE_MIXING
        and results[0].resulting_contraction is not None
        and all(optimizers[idx].generation >= MIXING_WARMUP_GENS - 1 for idx in optimizers)
    ):
        mix_exp = results[0].copy(no_energy=True)
        mix_exp.change_contraction(results[0].resulting_contraction)
        for idx in optimizers:
            optimizers[idx].update_root_exponent(mix_exp)


# ─── poll and trigger loop ────────────────────────────────────────────────────

active_evals      = []
trigger_idx       = 0
first_triggered   = False
last_trigger_gens = {idx: -1 for idx in optimizers}

while any(optimizers[idx].is_running for idx in optimizers):
    active_evals[:] = [t for t in active_evals if t.is_alive()]

    current_gens = {idx: optimizers[idx].generation for idx in optimizers}

    if len(active_evals) < THREADS_GLOBAL and all(current_gens[idx] >= 0 for idx in optimizers):
        if not first_triggered:
            ready = all(current_gens[idx] >= GLOBAL_EVAL_WARMUP_GENS - 1 for idx in optimizers)
        else:
            ready = all(
                current_gens[idx] >= last_trigger_gens[idx] + GLOBAL_EVAL_SPACING_GENS
                for idx in optimizers
            )

        if ready:
            combined = contracted_base.copy(no_energy=True)
            combined.uncontract_all()
            for idx in sorted(optimizers):
                state = optimizers[idx].get_state()
                if state["best_exp"] is not None:
                    combined.exponents[idx] = state["best_exp"].exponents[idx].copy()

            t = Thread(target=_global_eval_worker, args=(trigger_idx, combined, dict(current_gens)), daemon=True)
            t.start()
            active_evals.append(t)

            last_trigger_gens = dict(current_gens)
            first_triggered   = True
            trigger_idx      += 1

    time.sleep(1)

# ─── drain all remaining global evals ────────────────────────────────────────

for idx in optimizers:
    optimizers[idx].wait()
for t in active_evals:
    t.join()

# ─── final global eval ────────────────────────────────────────────────────────

combined   = contracted_base.copy(no_energy=True)
combined.uncontract_all()
final_gens = {}
for idx in sorted(optimizers):
    state = optimizers[idx].get_state()
    if state["best_exp"] is not None:
        combined.exponents[idx] = state["best_exp"].exponents[idx].copy()
    final_gens[idx] = state["generation"]

_global_eval_worker(trigger_idx, combined, final_gens)

csv_f.close()
log_f.close()

print(f"\nBest global energy : {_gs['best_energy']:.10f} Eh"
      f"  (ΔE = {_gs['best_energy'] - float(init_uncontracted.energy):+.10f})")
