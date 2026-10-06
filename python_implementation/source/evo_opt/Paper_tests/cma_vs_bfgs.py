import csv
import time
import shutil
import argparse
from math import ceil
from pathlib import Path
from numpy import array, float64, log, exp, zeros, abs as npabs

from scipy.optimize import minimize

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.common import L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.cma_shell_opt import Shell_Optimization
from evo_opt.tempering import from_registry
from evo_opt.Paper_tests.shared import (Counting_Objective, Log, bootstrap_contraction,
                                        local_job_config, progress, stage_inputs)

_arg_parser = argparse.ArgumentParser(description="Per shell, run CMA-ES and scipy BFGS from the SAME start on the SAME parametrization; record MOLCAS jobs and wall clock for each, written live")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "CMA_vs_BFGS").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # contracted frozen shells
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted (bootstrap contraction only)
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS            = [0, 1]   # compared one at a time; every other shell stays frozen
USE_CONTRACTION   = True
THREADS_PER_SHELL = 6        # concurrent MOLCAS jobs; both arms get exactly this

# Job_Manager poll, applied to BOTH arms. Its DEFAULT IS 5.0s, which quantises every
# batch: _run_all_jobs_serial costs 2*ceil(n/threads)-1 poll cycles, so a batch needing
# two refill rounds pays THREE cycles while a single-round batch pays one. CMA at
# popsize == threads is always one round, every BFGS stencil is two, so the default
# inflates BFGS wall clock by ~1.3x. Keep this well below the MOLCAS job time so the
# measurement reflects compute rather than polling.
JOB_POLL_SEC = 0.1

# Shared parametrization — the comparison is about the optimizer, so both arms use this.
# Off = raw log-exponents (dimension N); on = tempering (dimension M), which also shrinks
# the BFGS gradient from 2N to 2M jobs per step.
USE_TEMPERING      = True
N_TEMPERING_PARAMS = 6

# ── arm 1: CMA-ES (cma_shell_opt.Shell_Optimization) ──
# CMA_STOP_TOL is the spread of the last 5 generation bests. BFGS's default gtol=1e-5
# targets an energy excess of about M*gtol^2/(2*lambda) ~ 1e-11..1e-8 Eh, so 1e-6 would
# stop CMA several orders earlier than BFGS and the gap would be the tolerance, not the
# optimizer. 1e-8 is the practical match: tighter than BFGS's loose-curvature case while
# staying above the printed-energy noise floor.
CMA_GEN_SIZE = 6
CMA_SIGMA    = 0.1
CMA_STOP_TOL = 1e-8
CMA_MAX_GENS = 200           # backstop only; CMA_STOP_TOL should fire first
CMA_POLL_SEC = 0.05         # how often the live trace drains the optimizer's history. This
                            # is an observer, not part of the computation: the optimizer
                            # never waits on it. But each trace row is stamped when the
                            # loop NOTICES the generation, and cma_wall absorbs one final
                            # detection latency, so both are inflated by up to this much.
                            # Keep it small — it only ever makes CMA look slower.
SEED         = 42            # int -> reproducible CMA; None -> random

# ── arm 2: scipy BFGS ──
# Central differences with one global step. Known limitation left in deliberately: the
# generator's columns are k^i on k in [0,1], so a single step probes a_0 strongly and the
# high-order coefficients weakly. Per-column scaling would fix it but tunes the BFGS arm
# in a way the CMA arm gets no equivalent of.
# Note the gradient noise here is roughly eps_f/fd_step ~ 1e-10/1e-4 = 1e-6, within an
# order of gtol, so BFGS will often exit on a line-search failure rather than on gtol.
# That is the honest noise limit of FD-BFGS on this objective, not a bug.
BFGS_FD_STEP  = 1e-4
BFGS_GTOL     = 1e-5         # scipy's default
BFGS_MAX_ITER = 200          # backstop only

# How the energy and the gradient are submitted. scipy asks for them through separate
# callables, and never says in advance that it will want both at the same point:
#   False (split)       - fun is its own 1-job batch, jac its own 2M-job batch. Fewest
#                         JOBS, because the ~25% of calls that only need an energy do not
#                         pay for a stencil. But a 1-job batch leaves all but one core idle.
#   True  (speculative) - one callable returns (f, g), so every call submits centre +
#                         stencil = 2M+1 jobs in ONE batch. More jobs, but one parallel
#                         round, so it is faster in WALL CLOCK once cores >= 2M+1.
# At 6 cores split wins on both counts; the crossover is around 2M+1 cores.
BFGS_COMBINE_FUN_JAC = False

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

START_DIR   = WORK_DIR / "Start"
RESULTS_DIR = SUBMIT_DIR / "results"
START_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


exp_path, template_cont, template_full, run_scr, extract_scr = stage_inputs(
    SUBMIT_DIR, START_DIR, [EXPO_FILE, TEMPLATE_CONT, TEMPLATE_FULL, RUN_SCRIPT, EXTRACT_SCRIPT])

basis = Exponent_Set.from_file(exp_path)

job_cfg        = local_job_config(run_scr, extract_scr, JOB_POLL_SEC)
objective      = Counting_Objective(template_cont, job_cfg)
full_objective = Counting_Objective(template_full, job_cfg) if USE_CONTRACTION else None

# ─── live output: opened before any optimization so a crash keeps what ran ─────

run_log = Log(RESULTS_DIR / "cma_vs_bfgs.log")
emit    = run_log.emit


# every MOLCAS batch, flushed as it happens -> convergence curves for both arms
trace_f = open(RESULTS_DIR / "cma_vs_bfgs_trace.csv", "w", newline="")
trace_w = csv.writer(trace_f)
trace_w.writerow(["shell", "l", "method", "unit", "unit_idx", "molcas_jobs", "wall_sec", "E", "best_E"])
trace_f.flush()

# one row per shell, both arms side by side, flushed as each shell finishes
SUMMARY_HEADER = [
    "shell", "l", "n_exp", "n_params", "E_start",
    "cma_jobs", "cma_wall_s", "cma_E", "cma_dE", "cma_gens", "cma_stop",
    "bfgs_jobs", "bfgs_wall_s", "bfgs_E", "bfgs_dE", "bfgs_fun_calls", "bfgs_jac_calls",
    "bfgs_jobs_per_grad", "bfgs_stop",
    "E_cma_minus_bfgs", "max_abs_dln_alpha", "jobs_ratio_bfgs_over_cma", "wall_ratio_bfgs_over_cma",
]
sum_f = open(RESULTS_DIR / "cma_vs_bfgs_summary.csv", "w", newline="")
sum_w = csv.writer(sum_f)
sum_w.writerow([f"#META parametrization={'tempering_M' + str(N_TEMPERING_PARAMS) if USE_TEMPERING else 'raw_log'} "
                f"threads_per_shell={THREADS_PER_SHELL} use_contraction={USE_CONTRACTION} "
                f"cma_popsize={CMA_GEN_SIZE} cma_sigma={CMA_SIGMA} cma_stop_tol={CMA_STOP_TOL:.1e} "
                f"cma_max_gens={CMA_MAX_GENS} seed={SEED} "
                f"bfgs_fd_step={BFGS_FD_STEP:.1e} bfgs_gtol={BFGS_GTOL:.1e} bfgs_max_iter={BFGS_MAX_ITER}"])
sum_w.writerow(SUMMARY_HEADER)
sum_f.flush()

# ─── frozen backdrop, identical for both arms ─────────────────────────────────

if USE_CONTRACTION:
    emit("=== bootstrap contraction ===")
    base, _, e_boot = bootstrap_contraction(basis, full_objective, WORK_DIR,
                                            threads=THREADS_PER_SHELL)
    emit(f"  bootstrap E (uncontracted): {e_boot:.10f} Eh")
else:
    base = basis.copy(no_energy=True)
    base.uncontract_all()

emit("")
emit(f"shells      : {SHELLS}  ({', '.join(L_LABELS[s] for s in SHELLS)})")
emit(f"parametrize : {'tempering M=' + str(N_TEMPERING_PARAMS) if USE_TEMPERING else 'raw log-exponents'}")
emit(f"threads     : {THREADS_PER_SHELL} concurrent MOLCAS jobs per arm")
emit(f"CMA         : popsize {CMA_GEN_SIZE}, sigma {CMA_SIGMA}, stop_tol {CMA_STOP_TOL:.1e}, "
     f"cap {CMA_MAX_GENS} gens, seed {SEED}")
emit(f"BFGS        : central differences fd_step {BFGS_FD_STEP:.1e}, gtol {BFGS_GTOL:.1e}, "
     f"cap {BFGS_MAX_ITER} iters")
emit("")

for shell in SHELLS:
    lbl   = L_LABELS[shell]
    n_exp = len(base.exponents[shell])

    use_tempering_shell = USE_TEMPERING and n_exp > 1
    m_shell             = min(N_TEMPERING_PARAMS, n_exp) if use_tempering_shell else N_TEMPERING_PARAMS
    codec               = from_registry("polynomial", m=m_shell, n=n_exp) if use_tempering_shell else None
    n_params            = m_shell if use_tempering_shell else n_exp

    shell_dir = WORK_DIR / f"shell_{shell}"

    # identical starting point for both arms
    root = base.copy(no_energy=True)
    if USE_CONTRACTION:
        root.uncontract_shell(shell)

    init    = evaluate_initial(root, objective, shell_dir, threads=1, subdir_name="init",
                               contract_frozen_shells=USE_CONTRACTION)
    e_start = float(init.energy)

    emit("=" * 80)
    emit(f"shell {shell} ({lbl}): N={n_exp}, {n_params} parameter(s), start E = {e_start:.10f} Eh")
    emit("=" * 80)

    # ─── arm 1: CMA-ES ────────────────────────────────────────────────────────

    objective.n_jobs = 0
    t_cma            = time.time()

    opt = Shell_Optimization(
        init, e_start, objective,
        work_dir               = shell_dir / "cma",
        generation_size        = CMA_GEN_SIZE,
        sigma                  = CMA_SIGMA,
        max_generations        = CMA_MAX_GENS,
        active_shell           = shell,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = USE_CONTRACTION,
        use_tempering          = use_tempering_shell,
        n_tempering_params     = m_shell,
        use_stopping           = True,
        stop_tol               = CMA_STOP_TOL,
        seed                   = SEED,
    )
    opt.start(threads=THREADS_PER_SHELL)

    # drain the optimizer's history as it grows so the trace is written live rather
    # than reconstructed at the end
    drained = 0
    while True:
        running = opt.is_running
        history = opt.history
        for i in range(drained, len(history)):
            h = history[i]
            trace_w.writerow([shell, lbl, "CMA-ES", "generation", h["gen"],
                              (h["gen"] + 1) * CMA_GEN_SIZE, f"{time.time() - t_cma:.1f}",
                              f"{h['best_energy']:.10f}", f"{h['best_energy_overall']:.10f}"])
            trace_f.flush()
            progress(f"    [CMA gen {h['gen']:4d}] E {h['best_energy']:.8f} "
                     f"best {h['best_energy_overall']:.8f} sig {h['sigma']:.2e}")
        drained = len(history)
        if not running:
            break
        time.sleep(CMA_POLL_SEC)
    opt.wait()
    print()

    if opt.exception is not None:
        raise RuntimeError(f"shell {shell} ({lbl}): CMA crashed") from opt.exception

    cma_wall  = time.time() - t_cma
    cma_jobs  = objective.n_jobs
    cma_state = opt.get_state()
    cma_gens  = max(cma_state["generation"] + 1, 0)
    cma_best  = float(cma_state["best_energy"]) if cma_state["best_energy"] is not None else e_start
    cma_stop  = "stop_tol" if cma_gens < CMA_MAX_GENS else "max_generations"

    emit(f"  CMA-ES : {cma_gens:4d} gens | {cma_jobs:6d} jobs | {cma_wall:9.1f}s | "
         f"E {cma_best:.10f} (dE {cma_best - e_start:+.3e}) | stopped on {cma_stop}")

    # ─── arm 2: scipy BFGS, same start, same codec ────────────────────────────
    # fun and jac are passed SEPARATELY rather than as one jac=True callable, so a call
    # that only needs an energy costs 1 job instead of a whole stencil. The central
    # difference does not use the centre value, so jac costs 2*n_params, not 2*n_params+1.

    if codec is not None:
        x0 = array(codec.encode(init.exponents[shell]), dtype=float64)
    else:
        x0 = log(array(init.exponents[shell], dtype=float64))

    bfgs_dir = shell_dir / "bfgs"          # all BFGS job dirs under one roof, as CMA has
    bfgs_dir.mkdir(parents=True, exist_ok=True)

    objective.n_jobs = 0
    t_bfgs           = time.time()
    n_fun            = [0]
    n_jac            = [0]
    best             = [e_start, init.copy(no_energy=True)]

    def make_exp(params):
        candidate = root.copy(no_energy=True)
        if codec is not None:
            candidate.apply_params(shell, codec, params, n=n_exp)
        else:
            candidate.set_shell_exponents(shell, exp(params))
        if not USE_CONTRACTION:
            candidate.uncontract_all()
        return candidate

    def run_batch(points, unit, unit_idx):
        exps = [make_exp(points[i]) for i in range(len(points))]
        res  = objective.evaluate_batch(exps, threads=THREADS_PER_SHELL,
                                        work_dir=bfgs_dir / f"{unit}_{unit_idx:05d}")
        out  = zeros(len(res), dtype=float64)
        for i in range(len(res)):
            out[i] = float(res[i].energy)
            if out[i] < best[0]:
                best[0] = out[i]
                best[1] = res[i].copy(no_energy=True)
        trace_w.writerow([shell, lbl, "BFGS", unit, unit_idx, objective.n_jobs,
                          f"{time.time() - t_bfgs:.1f}", f"{float(out.min()):.10f}",
                          f"{best[0]:.10f}"])
        trace_f.flush()
        return out

    def f_only(params):
        n_fun[0] += 1
        e = run_batch([params], "fun", n_fun[0])[0]
        progress(f"    [BFGS f{n_fun[0]:3d} j{n_jac[0]:3d}] E {e:.8f} "
                 f"best {best[0]:.8f} {objective.n_jobs} jobs")
        return float(e)

    def grad_only(params):
        n_jac[0] += 1
        points = []
        for i in range(n_params):
            up = params.copy(); up[i] += BFGS_FD_STEP
            dn = params.copy(); dn[i] -= BFGS_FD_STEP
            points.append(up)
            points.append(dn)

        e = run_batch(points, "jac", n_jac[0])
        g = zeros(n_params, dtype=float64)
        for i in range(n_params):
            g[i] = (e[2 * i] - e[2 * i + 1]) / (2.0 * BFGS_FD_STEP)

        progress(f"    [BFGS f{n_fun[0]:3d} j{n_jac[0]:3d}] |g| {float((g ** 2).sum() ** 0.5):.2e} "
                 f"best {best[0]:.8f} {objective.n_jobs} jobs")
        return g

    def f_and_grad(params):
        n_fun[0] += 1
        n_jac[0] += 1
        points = [params]
        for i in range(n_params):
            up = params.copy(); up[i] += BFGS_FD_STEP
            dn = params.copy(); dn[i] -= BFGS_FD_STEP
            points.append(up)
            points.append(dn)

        e = run_batch(points, "fun_jac", n_fun[0])
        g = zeros(n_params, dtype=float64)
        for i in range(n_params):
            g[i] = (e[2 * i + 1] - e[2 * i + 2]) / (2.0 * BFGS_FD_STEP)

        progress(f"    [BFGS c{n_fun[0]:3d}] E {float(e[0]):.8f} "
                 f"|g| {float((g ** 2).sum() ** 0.5):.2e} best {best[0]:.8f} "
                 f"{objective.n_jobs} jobs")
        return float(e[0]), g

    if BFGS_COMBINE_FUN_JAC:
        result = minimize(f_and_grad, x0, jac=True, method="BFGS",
                          options={"gtol": BFGS_GTOL, "maxiter": BFGS_MAX_ITER, "disp": False})
    else:
        result = minimize(f_only, x0, jac=grad_only, method="BFGS",
                          options={"gtol": BFGS_GTOL, "maxiter": BFGS_MAX_ITER, "disp": False})
    print()

    bfgs_wall = time.time() - t_bfgs
    bfgs_jobs = objective.n_jobs
    bfgs_best = best[0]

    emit(f"  BFGS   : {result.nit:4d} iters | {bfgs_jobs:6d} jobs | {bfgs_wall:9.1f}s | "
         f"E {bfgs_best:.10f} (dE {bfgs_best - e_start:+.3e}) | success {result.success}")
    if BFGS_COMBINE_FUN_JAC:
        emit(f"           {n_fun[0]} combined calls ({2 * n_params + 1} jobs each, 1 batch) "
             f"| {result.message}")
    else:
        emit(f"           fun calls {n_fun[0]} (1 job each), jac calls {n_jac[0]} "
             f"({2 * n_params} jobs each) | {result.message}")

    # ─── did they land on the same point? ─────────────────────────────────────

    if cma_state["best_exp"] is not None:
        ln_cma   = log(array(cma_state["best_exp"].exponents[shell], dtype=float64))
        ln_bfgs  = log(array(best[1].exponents[shell], dtype=float64))
        max_dln  = float(npabs(ln_cma - ln_bfgs).max())
        cma_state["best_exp"].save(RESULTS_DIR, f"best_shell{shell}_CMA", overwrite=True)
    else:
        max_dln = float("nan")
    best[1].save(RESULTS_DIR, f"best_shell{shell}_BFGS", overwrite=True)

    emit(f"  same point? dE(CMA-BFGS) {cma_best - bfgs_best:+.3e} Eh | "
         f"max|dln(alpha)| {max_dln:.3e} ({(exp(max_dln) - 1.0) * 100:.3f}% in alpha)")
    emit(f"  cost ratio  BFGS/CMA: jobs {bfgs_jobs / cma_jobs if cma_jobs else float('nan'):.2f}x | "
         f"wall {bfgs_wall / cma_wall if cma_wall else float('nan'):.2f}x")

    sum_w.writerow([
        shell, lbl, n_exp, n_params, f"{e_start:.10f}",
        cma_jobs, f"{cma_wall:.1f}", f"{cma_best:.10f}", f"{cma_best - e_start:+.10f}", cma_gens, cma_stop,
        bfgs_jobs, f"{bfgs_wall:.1f}", f"{bfgs_best:.10f}", f"{bfgs_best - e_start:+.10f}",
        n_fun[0], n_jac[0], 2 * n_params, f"success={result.success}: {result.message}",
        f"{cma_best - bfgs_best:+.3e}", f"{max_dln:.3e}",
        f"{bfgs_jobs / cma_jobs:.3f}" if cma_jobs else "",
        f"{bfgs_wall / cma_wall:.3f}" if cma_wall else "",
    ])
    sum_f.flush()
    emit("")

trace_f.close()
sum_f.close()
emit(f"saved {RESULTS_DIR / 'cma_vs_bfgs_summary.csv'}   (one row per shell)")
emit(f"saved {RESULTS_DIR / 'cma_vs_bfgs_trace.csv'}     (every MOLCAS batch, both arms)")
emit(f"saved {RESULTS_DIR / 'cma_vs_bfgs.log'}")
emit(f"best .expo per shell per method saved in {RESULTS_DIR}")
run_log.close()
