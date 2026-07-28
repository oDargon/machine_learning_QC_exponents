import csv
import shutil
import argparse
from pathlib import Path
from threading import Thread, Semaphore, Lock
from numpy import array, float64, savez

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry
from evo_opt.cma_shell_opt import Shell_Optimization
from evo_opt.cbs_engine import _extrapolate_start   # N-aware geom extrapolation (handles N gaps)

_arg_parser = argparse.ArgumentParser(description="CMA-ES 2D tempering convergence, all shells in parallel, over N with a configurable step")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "CMA_Converge").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Si.expo"
TEMPLATE_CONT  = "temp_cont.inp"
TEMPLATE_FULL  = "temp_full.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS          = [0, 1, 2, 3, 4]   # all swept independently, in parallel
N_INCREASES     = 5        # per shell: this many points beyond N_start
N_STEP          = 1        # gap between successive N points (1 -> 17,18,19..; 2 -> 17,19,21..)
USE_CONTRACTION = True

M_PARAMS        = 2        # 2D tempering
SIGMA           = 0.1      # CMA step-size (CMA adapts it internally from here)
GENERATION_SIZE = 6        # CMA population per generation
MAX_GENERATIONS = 100      # hard cap; the early-stop should end well before this
USE_STOPPING    = True     # last-5-best-energies-within-1e-6 early stop

# core budget: run TOTAL_THREADS // THREADS_PER_SHELL shells concurrently, each CMA
# run using THREADS_PER_SHELL cores. On HPC bump TOTAL_THREADS and set per-shell = 6.
TOTAL_THREADS     = 6
THREADS_PER_SHELL = 3

USE_EXTRAPOLATION = True
N_FIT_POINTS      = 4         # optima nearest the query used for the local extrapolation

# CMA-ES seed handed to every CMA run unchanged. int -> all runs use it; None -> random.
SEED = None

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
    boot = evaluate_initial(exp, full_objective, WORK_DIR, threads=TOTAL_THREADS, subdir_name="bootstrap")
    if boot.resulting_contraction is None:
        raise RuntimeError("bootstrap produced no contraction")
    base = boot.copy(no_energy=True)
    base.change_contraction(boot.resulting_contraction)
    print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh\n")
else:
    base = exp.copy(no_energy=True)


def cma_converge(shell, codec, N, start_params, seed, threads):
    """One 2D CMA-ES run at fixed N, warm-started from start_params, using `threads`
    cores. Returns (best_energy, best_params, gens, history, e_start). Dirs removed after."""
    init_dir = WORK_DIR / f"s{shell}_N{N:02d}_init"
    cma_dir  = WORK_DIR / f"cma_s{shell}_N{N:02d}"

    work = base.copy(no_energy=True)
    work.apply_params(shell, codec, start_params, n=N)
    if not USE_CONTRACTION:
        work.uncontract_all()

    init    = evaluate_initial(work, energy_objective, init_dir, threads=threads,
                               subdir_name="init", contract_frozen_shells=USE_CONTRACTION)
    e_start = float(init.energy)

    opt = Shell_Optimization(
        init, float(init.energy), energy_objective,
        work_dir               = cma_dir,
        generation_size        = GENERATION_SIZE,
        sigma                  = SIGMA,
        max_generations        = MAX_GENERATIONS,
        active_shell           = shell,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = USE_CONTRACTION,
        use_tempering          = True,
        n_tempering_params     = M_PARAMS,
        use_stopping           = USE_STOPPING,
        seed                   = seed,
    )
    opt.start(threads=threads)
    opt.wait()

    if opt.exception is not None:
        print(f"  [WARNING] shell {shell} N={N}: CMA crashed ({opt.exception!r}); "
              f"recording the initial energy.", flush=True)

    state  = opt.get_state()
    e_best = float(state["best_energy"]) if state["best_energy"] is not None else float(init.energy)
    if state["best_exp"] is not None:
        best_params = array(codec.encode(state["best_exp"].exponents[shell]), dtype=float64)
    else:
        best_params = array(start_params, dtype=float64)
    gens    = max(state["generation"] + 1, 0)
    history = opt.history                          # in-memory; survives the rmtree below

    shutil.rmtree(init_dir, ignore_errors=True)
    shutil.rmtree(cma_dir,  ignore_errors=True)
    return e_best, best_params, gens, history, e_start


# ─── parallel sweep: one thread per shell, gated by a core-budget semaphore ────

n_slots   = max(1, TOTAL_THREADS // THREADS_PER_SHELL)
_sem      = Semaphore(n_slots)   # at most this many CMA runs at once
_csv_lock = Lock()               # the live CSV / row list are shared across shell threads

HEADER = ["shell", "l", "N", "E_cma", "a0", "a1",
          "start_a0", "start_a1", "start_source", "gens_to_converge"]

# live CSV: flushed per point in completion order (crash-safe). The ordered CSV
# (sorted by shell, N) is written from the collected rows once everything finishes.
live_path = RESULTS_DIR / "cma_minima_live.csv"
live_f    = open(live_path, "w", newline="")
live_w    = csv.writer(live_f)
live_w.writerow(HEADER)
live_f.flush()
all_rows  = []


def shell_worker(shell):
    lbl      = L_LABELS[shell]
    n0       = len(base.exponents[shell])
    n_values = [n0 + i * N_STEP for i in range(N_INCREASES + 1)]
    print(f"=== shell {shell} ({lbl}): N {n_values} ===", flush=True)

    opt_hist = []   # (N, a0, a1) of converged optima for this shell
    for N in n_values:
        codec = from_registry("polynomial", m=M_PARAMS, n=N)

        if USE_EXTRAPOLATION and len(opt_hist) >= 2:
            center, src = _extrapolate_start(opt_hist, N, N_FIT_POINTS), "geom"
        elif opt_hist:
            center, src = array([opt_hist[-1][1], opt_hist[-1][2]], dtype=float64), "prev"
        else:
            center, src = array(codec.encode(base.exponents[shell]), dtype=float64), "encode"

        seed = SEED   # every CMA gets the exact same seed (None -> CMA picks random)
        with _sem:   # hold a core slot only for the actual CMA run
            e_best, best, gens, history, e_start = cma_converge(shell, codec, N, center, seed, THREADS_PER_SHELL)

        row = [shell, lbl, N, f"{e_best:.10f}", f"{best[0]:.10e}", f"{best[1]:.10e}",
               f"{center[0]:.10e}", f"{center[1]:.10e}", src, gens]
        with _csv_lock:
            live_w.writerow(row)
            live_f.flush()
            all_rows.append(row)

        if history:
            savez(
                RESULTS_DIR / f"traj_shell{shell}_N{N:02d}.npz",
                shell=shell, l=lbl, N=N, e_start=e_start, e_final=e_best, best_params=best,
                start_params=array(center, dtype=float64), start_source=src,
                gen                 = array([h["gen"]                 for h in history]),
                mean                = array([h["mean"]                for h in history]),
                sigma               = array([h["sigma"]               for h in history]),
                cov                 = array([h["cov"]                 for h in history]),
                best_energy         = array([h["best_energy"]         for h in history]),
                best_energy_overall = array([h["best_energy_overall"] for h in history]),
            )

        print(f"  shell {shell} ({lbl}) N={N:3d} [{src:>6}]: E_cma={e_best:.10f} "
              f"at ({best[0]:+.3f},{best[1]:+.3f})  ({gens} gens)", flush=True)

        opt_hist.append((N, float(best[0]), float(best[1])))


threads = [Thread(target=shell_worker, args=(s,), daemon=True) for s in SHELLS]
for t in threads:
    t.start()
for t in threads:
    t.join()

live_f.close()

# ordered CSV: sorted by shell then N, written once every shell has finished
ordered_path = RESULTS_DIR / "cma_minima.csv"
with open(ordered_path, "w", newline="") as ord_f:
    ord_w = csv.writer(ord_f)
    ord_w.writerow(HEADER)
    for row in sorted(all_rows, key=lambda r: (r[0], r[2])):
        ord_w.writerow(row)

print(f"\nsaved {ordered_path}       (ordered)")
print(f"saved {live_path}  (live, completion order)")
