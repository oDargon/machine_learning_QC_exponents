import csv
import shutil
import argparse
from pathlib import Path
from numpy import array, float64, savez, polyfit, polyval, diff, median

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type, L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry
from evo_opt.cma_shell_opt import Shell_Optimization

_arg_parser = argparse.ArgumentParser(description="CMA-ES 2D tempering convergence per shell across increasing N")
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

SHELLS          = [0]      # shells to test; each swept independently over N
N_INCREASES     = 5        # per shell: N_start up to N_start + this many
USE_CONTRACTION = True

M_PARAMS        = 2        # 2D tempering (matches the 25th grid scans, for comparison)
SIGMA           = 0.1      # CMA step-size (CMA adapts it internally from here)
GENERATION_SIZE = 6        # CMA population per generation
MAX_GENERATIONS = 100      # hard cap; the early-stop should end well before this
USE_STOPPING    = True     # last-5-best-energies-within-1e-6 early stop
THREADS         = 6

# after >=2 optima, predict the next N's start with a geometric-increment model of
# the converged optima in (a0, lnβ) space (lnβ = a1/(N-1) factors out a1's mechanical
# N-growth), then reconstruct a1 = (N-1)*lnβ. Uses only the last N_FIT_POINTS optima.
USE_EXTRAPOLATION = True
N_FIT_POINTS      = 4         # recent optima used for the local model

# CMA-ES seed. Set an int for reproducible runs (each sub-optimisation is seeded
# deterministically from this base); None = fresh random each run.
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
    boot = evaluate_initial(exp, full_objective, WORK_DIR, threads=THREADS, subdir_name="bootstrap")
    if boot.resulting_contraction is None:
        raise RuntimeError("bootstrap produced no contraction")
    base = boot.copy(no_energy=True)
    base.change_contraction(boot.resulting_contraction)
    print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh\n")
else:
    base = exp.copy(no_energy=True)


def cma_converge(shell, codec, N, start_params, seed):
    """One 2D CMA-ES run at fixed N, warm-started from start_params. Returns
    (best_energy, best_params, gens, history, e_start). Job dirs are removed after."""
    init_dir = WORK_DIR / f"s{shell}_N{N:02d}_init"
    cma_dir  = WORK_DIR / f"cma_s{shell}_N{N:02d}"

    work = base.copy(no_energy=True)
    work.apply_params(shell, codec, start_params, n=N)
    if not USE_CONTRACTION:
        work.uncontract_all()

    init    = evaluate_initial(work, energy_objective, init_dir, threads=THREADS,
                               subdir_name="init", contract_frozen_shells=USE_CONTRACTION)
    e_start = float(init.energy)   # energy at the (warm-started) starting point

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
    opt.start(threads=THREADS)
    opt.wait()

    if opt.exception is not None:
        print(f"  [WARNING] shell {shell} N={N}: CMA crashed ({opt.exception!r}); "
              f"recording the initial energy.", flush=True)

    state  = opt.get_state()
    e_best = float(state["best_energy"]) if state["best_energy"] is not None else float(init.energy)
    # params of the actual best point found (encode is exact for the linear
    # generator), so the next N warm-starts from the found minimum, not the mean
    if state["best_exp"] is not None:
        best_params = array(codec.encode(state["best_exp"].exponents[shell]), dtype=float64)
    else:
        best_params = array(start_params, dtype=float64)
    gens    = max(state["generation"] + 1, 0)
    history = opt.history                          # in-memory; survives the rmtree below

    shutil.rmtree(init_dir, ignore_errors=True)   # free job files between runs
    shutil.rmtree(cma_dir,  ignore_errors=True)
    return e_best, best_params, gens, history, e_start


def _geom_predict(Ns, ys, n_new):
    """Geometric-increment predictor: model each step as a fixed fraction of the
    previous (Δ_{k+1} = r·Δ_k), i.e. an exponential approach to the asymptote.
    r is the median of recent increment-ratios (robust). Falls back to a line
    when there aren't enough increments. Assumes consecutive-N history."""
    if len(ys) < 3:                                   # need >=2 increments for a ratio
        return float(polyval(polyfit(Ns, ys, 1), n_new))
    d    = diff(ys)
    prev = d[:-1]
    mask = abs(prev) > 1e-12
    if not mask.any():
        return float(polyval(polyfit(Ns, ys, 1), n_new))
    ratios = d[1:][mask] / prev[mask]
    r      = max(min(float(median(ratios[-3:])), 1.2), -1.0)   # clamp to a sane band
    y_pred, incr = float(ys[-1]), float(d[-1]) * r
    for _ in range(int(round(n_new - Ns[-1]))):       # usually one step
        y_pred += incr
        incr   *= r
    return y_pred


def extrapolate_start(opt_hist, n_new):
    """Predict start params [a0, a1] for n_new from recent converged optima,
    modelling in (a0, lnβ) space then reconstructing a1 = (N-1)*lnβ."""
    pts = opt_hist[-N_FIT_POINTS:]
    Ns  = array([p[0]                for p in pts], dtype=float64)
    a0  = array([p[1]                for p in pts], dtype=float64)
    lnb = array([p[2] / (p[0] - 1.0) for p in pts], dtype=float64)   # lnβ = a1/(N-1)

    a0_pred  = _geom_predict(Ns, a0,  n_new)
    lnb_pred = _geom_predict(Ns, lnb, n_new)
    return array([a0_pred, lnb_pred * (n_new - 1.0)], dtype=float64)


# ─── sweep: per shell, CMA converge at N_start .. N_start + N_INCREASES ────────

csv_path = RESULTS_DIR / "cma_minima.csv"
csv_f    = open(csv_path, "w", newline="")
writer   = csv.writer(csv_f)
writer.writerow(["shell", "l", "N", "E_cma", "a0", "a1",
                 "start_a0", "start_a1", "start_source", "gens_to_converge"])
csv_f.flush()

for shell in SHELLS:
    lbl = L_LABELS[shell]
    n0  = len(base.exponents[shell])
    print(f"=== shell {shell} ({lbl}): N {n0}..{n0 + N_INCREASES} ===")

    opt_hist = []   # (N, a0, a1) of converged optima for this shell
    for N in range(n0, n0 + N_INCREASES + 1):
        codec = from_registry("polynomial", m=M_PARAMS, n=N)

        # choose the starting guess
        if USE_EXTRAPOLATION and len(opt_hist) >= 2:
            center, src = extrapolate_start(opt_hist, N), "geom"
        elif opt_hist:
            center, src = array([opt_hist[-1][1], opt_hist[-1][2]], dtype=float64), "prev"
        else:
            center, src = array(codec.encode(base.exponents[shell]), dtype=float64), "encode"

        # deterministic per-(shell,N) seed from the base SEED (None -> random)
        seed = None if SEED is None else SEED + shell * 1000 + N
        e_best, best, gens, history, e_start = cma_converge(shell, codec, N, center, seed)
        writer.writerow([shell, lbl, N, f"{e_best:.10f}", f"{best[0]:.10e}", f"{best[1]:.10e}",
                         f"{center[0]:.10e}", f"{center[1]:.10e}", src, gens])
        csv_f.flush()

        # dump the full CMA trajectory (mean / sigma / covariance / best-E per gen),
        # plus the extrapolated guess (start_params) and how it was chosen
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

        print(f"  N={N:3d} [{src:>6} guess ({center[0]:+.3f},{center[1]:+.3f})]: "
              f"E_cma={e_best:.10f} at ({best[0]:+.3f},{best[1]:+.3f})  ({gens} gens)", flush=True)

        opt_hist.append((N, float(best[0]), float(best[1])))

csv_f.close()
print(f"\nsaved {csv_path}")
