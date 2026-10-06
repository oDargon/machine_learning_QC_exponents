import csv
import argparse
from pathlib import Path
from numpy import array, float64, zeros, log, abs as npabs

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.common import L_LABELS, CHEMICAL_ACCURACY
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.Paper_tests.shared import (Log, local_job_config, progress,
                                        run_cma_shell, stage_inputs)

_arg_parser = argparse.ArgumentParser(description="How much energy does tempering cost? Optimize one shell at a range of parametrization dimensions M, every other shell frozen and UNCONTRACTED, and compare against full per-exponent freedom")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Tempering_Loss").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_FULL  = "temp_full.inp"    # nothing is ever contracted in this test
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS = [1]        # the shell(s) to optimize; every other shell stays frozen and uncontracted
M_MIN  = 2          # parametrization dimension swept from here ...
M_MAX  = 6          # ... to here, inclusive. Clamped to N above; M > N is overdetermined.
                    # M=1 is degenerate whenever N > 1: the polynomial generator's first
                    # column is all-ones, so ln(alpha_k) = a0 for every k and the whole
                    # shell collapses to N IDENTICAL exponents. It is in range only if you
                    # specifically want that datum. M=2 is the first useful point (it is
                    # the even-tempered series).

# Also run with every exponent free (N parameters, no codec). At M == N tempering spans all
# of log-exponent space, so that run and this one should agree to within optimizer noise —
# which is the floor the M < N losses have to beat to mean anything.
INCLUDE_UNTEMPERED = True

# Repeats. Without them an energy difference between two M values cannot be told apart from
# the optimizer's own seed-to-seed scatter, which on a flat valley is not small.
SEEDS = [1, 2, 3]

CMA_GEN_SIZE = 6
CMA_SIGMA    = 0.1
CMA_STOP_TOL = 1e-6
CMA_MAX_GENS = 200           # backstop; CMA_STOP_TOL should fire first
CMA_POLL_SEC = 0.05

THREADS      = 6
JOB_POLL_SEC = 0.1           # default is 5.0s, which quantises every batch

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

START_DIR   = WORK_DIR / "Start"
RESULTS_DIR = SUBMIT_DIR / "results"
START_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


exp_path, template_full, run_scr, extract_scr = stage_inputs(
    SUBMIT_DIR, START_DIR, [EXPO_FILE, TEMPLATE_FULL, RUN_SCRIPT, EXTRACT_SCRIPT])

basis = Exponent_Set.from_file(exp_path)

job_cfg        = local_job_config(run_scr, extract_scr, JOB_POLL_SEC)
full_objective = Ground_Energy_Objective(template_full, job_cfg)

run_log = Log(RESULTS_DIR / "tempering_loss.log")
emit    = run_log.emit


trace_f = open(RESULTS_DIR / "tempering_loss_trace.csv", "w", newline="")
trace_w = csv.writer(trace_f)
trace_w.writerow(["shell", "l", "n_params", "label", "seed", "gen", "jobs", "wall_s",
                  "E_gen_best", "E_best_overall", "sigma"])
trace_f.flush()

RUNS_HEADER = ["shell", "l", "n_exp", "n_params", "label", "seed",
               "gens", "jobs", "wall_s", "hit_gen_cap", "E_best", "dE_from_start"]
runs_f = open(RESULTS_DIR / "tempering_loss_runs.csv", "w", newline="")
runs_w = csv.writer(runs_f)
runs_w.writerow(RUNS_HEADER)
runs_f.flush()

SUMMARY_HEADER = ["shell", "l", "n_exp", "n_params", "label", "n_seeds",
                  "E_mean", "E_std", "E_best", "loss_vs_ref_Eh", "loss_vs_ref_chem_acc",
                  "gens_mean", "jobs_total", "any_hit_cap"]
sum_f = open(RESULTS_DIR / "tempering_loss.csv", "w", newline="")
sum_w = csv.writer(sum_f)
sum_w.writerow([f"#META atom={basis.atom_name} shells={SHELLS} M={M_MIN}..{M_MAX} "
                f"untempered={INCLUDE_UNTEMPERED} seeds={SEEDS} popsize={CMA_GEN_SIZE} "
                f"sigma={CMA_SIGMA} stop_tol={CMA_STOP_TOL:.1e} max_gens={CMA_MAX_GENS} "
                f"chem_acc={CHEMICAL_ACCURACY:.6e}"])
sum_w.writerow(SUMMARY_HEADER)
sum_f.flush()


def run_one(shell, lbl, n_params, label, seed, start_exp, start_energy, use_temp, m_shell):
    """One CMA run at a fixed parametrization. Every other shell stays frozen and
    uncontracted, so all runs share one objective and their energies compare directly."""

    def on_gen(h, elapsed):
        trace_w.writerow([shell, lbl, n_params, label, seed, h["gen"],
                          (h["gen"] + 1) * CMA_GEN_SIZE, f"{elapsed:.1f}",
                          f"{h['best_energy']:.10f}", f"{h['best_energy_overall']:.10f}",
                          f"{h['sigma']:.6e}"])
        trace_f.flush()
        progress(f"    [{label} seed {seed}] gen {h['gen'] + 1:4d}  "
                 f"best {h['best_energy_overall']:.8f}  sig {h['sigma']:.2e}")

    r = run_cma_shell(
        start_exp, start_energy, full_objective,
        WORK_DIR / f"shell_{shell}" / label / f"seed_{seed}", shell,
        generation_size        = CMA_GEN_SIZE,
        sigma                  = CMA_SIGMA,
        max_generations        = CMA_MAX_GENS,
        contract_frozen_shells = False,
        use_tempering          = use_temp,
        n_tempering_params     = m_shell,
        use_stopping           = True,
        stop_tol               = CMA_STOP_TOL,
        seed                   = seed,
        threads                = THREADS,
        poll_sec               = CMA_POLL_SEC,
        on_generation          = on_gen,
        crash_label            = f"shell {shell} {label} seed {seed}",
    )
    return r["best_exp"], r["best_energy"], r["gens"], r["wall"]


# ─── starting point: fully uncontracted, nothing frozen is contracted ─────────

start = basis.copy(no_energy=True)
start.uncontract_all()
init    = evaluate_initial(start, full_objective, WORK_DIR, threads=1,
                           subdir_name="init", contract_frozen_shells=False)
e_start = float(init.energy)
emit(f"start E (fully uncontracted): {e_start:.10f} Eh")
emit("")

for shell in SHELLS:
    lbl   = L_LABELS[shell]
    n_exp = len(init.exponents[shell])

    m_values = [m for m in range(M_MIN, M_MAX + 1) if m <= n_exp]
    dropped  = [m for m in range(M_MIN, M_MAX + 1) if m > n_exp]

    plan = []
    for i in range(len(m_values)):
        plan.append((m_values[i], f"M{m_values[i]:02d}", True, m_values[i]))
    if INCLUDE_UNTEMPERED:
        plan.append((n_exp, "untempered", False, 0))

    emit("=" * 78)
    emit(f"shell {shell} ({lbl}): N={n_exp} exponents")
    emit("=" * 78)
    if dropped:
        emit(f"  [note] M {dropped} exceed N={n_exp} and were dropped — M>N is degenerate "
             f"(the codec is overdetermined and lstsq returns a minimum-norm fit)")
    if 1 in m_values and n_exp > 1:
        emit(f"  [WARNING] M=1 with N={n_exp} is degenerate: the generator's first column is")
        emit(f"            all-ones, so every exponent collapses to the same value. Expect no")
        emit(f"            improvement over the start. It also routes to the 1-D scalar CSA ES")
        emit(f"            rather than cma, so it is a different optimizer as well.")
    emit(f"  runs: {len(plan)} parametrizations x {len(SEEDS)} seeds = {len(plan) * len(SEEDS)}")
    emit("")

    e_all    = zeros((len(plan), len(SEEDS)), dtype=float64)
    g_all    = zeros((len(plan), len(SEEDS)), dtype=float64)
    j_all    = zeros(len(plan), dtype=float64)
    cap_all  = zeros(len(plan), dtype=float64)

    for p in range(len(plan)):
        n_params, label, use_temp, m_shell = plan[p]
        m_pass = m_shell if use_temp else n_exp

        for s in range(len(SEEDS)):
            seed = SEEDS[s]
            x, e, gens, wall = run_one(shell, lbl, n_params, label, seed,
                                       init, e_start, use_temp, m_pass)
            e_all[p][s]  = e
            g_all[p][s]  = gens
            j_all[p]    += gens * CMA_GEN_SIZE
            if gens >= CMA_MAX_GENS:
                cap_all[p] = 1.0

            if x is not None:
                out = x.copy(no_energy=True)
                out.energy = e
                out.save(RESULTS_DIR, f"temper_shell{shell}_{label}_seed{seed}", overwrite=True)

            runs_w.writerow([shell, lbl, n_exp, n_params, label, seed, gens,
                             int(gens * CMA_GEN_SIZE), f"{wall:.1f}",
                             int(gens >= CMA_MAX_GENS), f"{e:.10f}", f"{e - e_start:+.10f}"])
            runs_f.flush()

        emit(f"  {label:<11} ({n_params:>2} params): E {e_all[p].mean():.10f} "
             f"+/- {e_all[p].std():.2e}   best {e_all[p].min():.10f}   "
             f"dE {e_all[p].min() - e_start:+.3e}   {int(g_all[p].mean())} gens avg"
             + ("   [HIT CAP]" if cap_all[p] else ""))

    # reference = the lowest energy anything reached for this shell. All runs share one
    # objective, so this is a like-for-like comparison with no re-scoring.
    e_ref = float(e_all.min())

    emit("")
    emit(f"  --- tempering loss, shell {shell} ({lbl}), reference = best over all runs "
         f"({e_ref:.10f} Eh) ---")
    emit(f"  {'parametrization':<16}{'params':>7}{'E_best':>18}{'loss (Eh)':>13}{'loss / chem acc':>17}")
    for p in range(len(plan)):
        n_params, label, use_temp, _ = plan[p]
        loss = float(e_all[p].min()) - e_ref
        emit(f"  {label:<16}{n_params:>7}{e_all[p].min():>18.10f}{loss:>13.3e}"
             f"{loss / CHEMICAL_ACCURACY:>17.4f}")

        sum_w.writerow([shell, lbl, n_exp, n_params, label, len(SEEDS),
                        f"{e_all[p].mean():.10f}", f"{e_all[p].std():.10e}",
                        f"{e_all[p].min():.10f}", f"{loss:.10e}",
                        f"{loss / CHEMICAL_ACCURACY:.6f}",
                        f"{g_all[p].mean():.1f}", int(j_all[p]), int(cap_all[p])])
    sum_f.flush()

    seed_std = float(e_all.std(axis=1).mean())
    emit("")
    emit(f"  mean within-parametrization seed std: {seed_std:.3e} Eh "
         f"({seed_std / CHEMICAL_ACCURACY:.4f} x chemical accuracy)")
    emit(f"  any loss below that is not distinguishable from optimizer noise.")
    if INCLUDE_UNTEMPERED and n_exp in [plan[p][0] for p in range(len(plan)) if plan[p][2]]:
        emit(f"  consistency check: M={n_exp} and 'untempered' both span the full space, so")
        emit(f"  their difference is pure optimizer noise — compare them directly.")
    emit("")

runs_f.close()
sum_f.close()
trace_f.close()
emit(f"saved {RESULTS_DIR / 'tempering_loss.csv'}        (one row per parametrization)")
emit(f"saved {RESULTS_DIR / 'tempering_loss_runs.csv'}   (one row per run)")
emit(f"saved {RESULTS_DIR / 'tempering_loss_trace.csv'}  (every generation of every run)")
emit(f"saved {RESULTS_DIR / 'tempering_loss.log'}")
emit(f"saved temper_shell*_M*_seed*.expo in {RESULTS_DIR}  (converged basis per run)")
run_log.close()
