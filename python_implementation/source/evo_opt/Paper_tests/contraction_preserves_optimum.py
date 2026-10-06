import csv
import argparse
from pathlib import Path
from numpy import array, float64, zeros, log, abs as npabs

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.common import L_LABELS
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.Paper_tests.shared import (Log, bootstrap_contraction, local_job_config,
                                        progress, run_cma_shell, stage_inputs)

_arg_parser = argparse.ArgumentParser(description="Does optimizing a shell against a CONTRACTED backdrop reach the same solution as against a fully uncontracted one? Repeated over seeds, with both solutions scored on the same uncontracted objective")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Contraction_Optimum").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # frozen shells contracted, active shell free
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

# This is the most expensive test here: one full CMA run per shell PER SEED PER ARM.
# Budget roughly  len(SHELLS) * len(SEEDS) * 2 * gens * CMA_GEN_SIZE  jobs, and the
# uncontracted arm's jobs are individually dearer. Start with one shell.
SHELLS = [0]
SEEDS  = [1, 2, 3, 4, 5]

# held fixed across both arms — this test varies the BACKDROP, nothing else
USE_TEMPERING      = True
N_TEMPERING_PARAMS = 6

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


exp_path, template_cont, template_full, run_scr, extract_scr = stage_inputs(
    SUBMIT_DIR, START_DIR, [EXPO_FILE, TEMPLATE_CONT, TEMPLATE_FULL, RUN_SCRIPT, EXTRACT_SCRIPT])

basis = Exponent_Set.from_file(exp_path)

job_cfg        = local_job_config(run_scr, extract_scr, JOB_POLL_SEC)
objective      = Ground_Energy_Objective(template_cont, job_cfg)
full_objective = Ground_Energy_Objective(template_full, job_cfg)

run_log = Log(RESULTS_DIR / "contraction_optimum.log")
emit    = run_log.emit


# every generation of every run, flushed live — this is how you check afterwards whether a
# run converged or merely ran out of generations. Opened before run_arm, which writes to it.
trace_f = open(RESULTS_DIR / "contraction_optimum_trace.csv", "w", newline="")
trace_w = csv.writer(trace_f)
trace_w.writerow(["shell", "l", "seed", "arm", "gen", "jobs", "wall_s",
                  "E_gen_best", "E_best_overall", "sigma"])
trace_f.flush()


def run_arm(shell, seed, start_exp, start_energy, obj, contract_frozen, tag, n_exp, m_shell, use_temp):
    """One CMA run, draining its history into the live trace so a run that hits the
    generation cap rather than converging is visible afterwards.
    Returns (best_exp, native_best_energy, gens, jobs, wall)."""

    def on_gen(h, elapsed):
        trace_w.writerow([shell, L_LABELS[shell], seed, tag, h["gen"],
                          (h["gen"] + 1) * CMA_GEN_SIZE, f"{elapsed:.1f}",
                          f"{h['best_energy']:.10f}", f"{h['best_energy_overall']:.10f}",
                          f"{h['sigma']:.6e}"])
        trace_f.flush()
        progress(f"    [{tag} seed {seed}] gen {h['gen'] + 1:4d}  "
                 f"best {h['best_energy_overall']:.8f}  sig {h['sigma']:.2e}")

    r = run_cma_shell(
        start_exp, start_energy, obj,
        WORK_DIR / f"shell_{shell}" / f"seed_{seed}" / tag, shell,
        generation_size        = CMA_GEN_SIZE,
        sigma                  = CMA_SIGMA,
        max_generations        = CMA_MAX_GENS,
        contract_frozen_shells = contract_frozen,
        use_tempering          = use_temp,
        n_tempering_params     = m_shell,
        use_stopping           = True,
        stop_tol               = CMA_STOP_TOL,
        seed                   = seed,
        threads                = THREADS,
        poll_sec               = CMA_POLL_SEC,
        on_generation          = on_gen,
        crash_label            = f"shell {shell} seed {seed} arm {tag}",
    )
    return r["best_exp"], r["best_energy"], r["gens"], r["jobs"], r["wall"]


# ─── the contraction under test, generated once at the starting basis ──────────

emit("=== bootstrap: fully uncontracted run at the starting basis -> GENANO contraction ===")
base_cont, base_full, e_boot = bootstrap_contraction(basis, full_objective, WORK_DIR,
                                                      threads=THREADS)
emit(f"  E (uncontracted, unperturbed): {e_boot:.10f} Eh")

PTS_HEADER = ["shell", "l", "seed", "arm", "gens", "jobs", "wall_s", "hit_gen_cap",
              "E_native", "E_on_full", "max_abs_dln_vs_other_arm"]
pts_f = open(RESULTS_DIR / "contraction_optimum_runs.csv", "w", newline="")
pts_w = csv.writer(pts_f)
pts_w.writerow(PTS_HEADER)
pts_f.flush()


SUMMARY_HEADER = [
    "shell", "l", "n_exp", "n_params", "n_seeds",
    "E_full_cont_mean", "E_full_cont_std", "E_full_full_mean", "E_full_full_std",
    "dE_mean", "dE_std", "dE_worst",
    "dln_between_mean", "dln_between_worst",
    "dln_within_cont", "dln_within_full",
    "jobs_cont", "jobs_full", "verdict",
]
sum_f = open(RESULTS_DIR / "contraction_optimum.csv", "w", newline="")
sum_w = csv.writer(sum_f)
sum_w.writerow([f"#META atom={basis.atom_name} seeds={SEEDS} tempering={USE_TEMPERING} "
                f"M={N_TEMPERING_PARAMS} popsize={CMA_GEN_SIZE} sigma={CMA_SIGMA} "
                f"stop_tol={CMA_STOP_TOL:.1e} max_gens={CMA_MAX_GENS} threads={THREADS}"])
sum_w.writerow(SUMMARY_HEADER)
sum_f.flush()

emit("")
emit(f"shells {SHELLS}, seeds {SEEDS} -> {len(SHELLS) * len(SEEDS) * 2} CMA runs")
emit("both arms start from the same exponents and differ ONLY in the backdrop:")
emit("  cont : this shell free, every other shell frozen at the bootstrap contraction")
emit("  full : everything uncontracted")
emit("both converged solutions are then re-scored on the SAME uncontracted objective,")
emit("because the arms' native energies come from different templates and are not comparable.")
emit("")

for shell in SHELLS:
    lbl   = L_LABELS[shell]
    n_exp = len(base_cont.exponents[shell])

    use_temp = USE_TEMPERING and n_exp > 1
    m_shell  = min(N_TEMPERING_PARAMS, n_exp) if use_temp else N_TEMPERING_PARAMS
    n_params = m_shell if use_temp else n_exp

    emit("=" * 78)
    emit(f"shell {shell} ({lbl}): N={n_exp}, {n_params} parameter(s)")
    emit("=" * 78)

    start_cont = base_cont.copy(no_energy=True)
    start_cont.uncontract_shell(shell)
    init_cont  = evaluate_initial(start_cont, objective, WORK_DIR / f"shell_{shell}",
                                  threads=1, subdir_name="init_cont",
                                  contract_frozen_shells=True)
    init_full  = evaluate_initial(base_full, full_objective, WORK_DIR / f"shell_{shell}",
                                  threads=1, subdir_name="init_full",
                                  contract_frozen_shells=False)
    emit(f"  start E: cont arm {float(init_cont.energy):.10f}   "
         f"full arm {float(init_full.energy):.10f}   "
         f"(offset {float(init_cont.energy) - float(init_full.energy):+.3e} Eh)")

    ln_cont   = zeros((len(SEEDS), n_exp), dtype=float64)
    ln_full   = zeros((len(SEEDS), n_exp), dtype=float64)
    e_on_full = zeros((len(SEEDS), 2), dtype=float64)   # [:,0] cont solution, [:,1] full solution
    e_native  = zeros((len(SEEDS), 2), dtype=float64)
    jobs      = zeros(2, dtype=float64)

    for s in range(len(SEEDS)):
        seed = SEEDS[s]

        x_c, e_c, g_c, j_c, w_c = run_arm(shell, seed, init_cont, float(init_cont.energy),
                                          objective, True, "cont", n_exp, m_shell, use_temp)
        x_f, e_f, g_f, j_f, w_f = run_arm(shell, seed, init_full, float(init_full.energy),
                                          full_objective, False, "full", n_exp, m_shell, use_temp)
        jobs[0] += j_c
        jobs[1] += j_f

        # re-score BOTH on the fully uncontracted objective: the only common scale
        score = []
        for cand in (x_c, x_f):
            c = cand.copy(no_energy=True)
            c.uncontract_all()
            score.append(c)
        rescored = full_objective.evaluate_batch(
            score, threads=THREADS,
            work_dir=WORK_DIR / f"shell_{shell}" / f"seed_{seed}" / "rescore")

        ln_cont[s]   = log(array(x_c.exponents[shell], dtype=float64))
        ln_full[s]   = log(array(x_f.exponents[shell], dtype=float64))
        e_native[s]  = (e_c, e_f)
        e_on_full[s] = (float(rescored[0].energy), float(rescored[1].energy))

        dln = float(npabs(ln_cont[s] - ln_full[s]).max())
        emit(f"  seed {seed:>3}: cont {g_c:>4} gens ({j_c:>5} jobs, {w_c:>7.1f}s)  "
             f"full {g_f:>4} gens ({j_f:>5} jobs, {w_f:>7.1f}s)")
        emit(f"            E on full: cont {e_on_full[s][0]:.10f}  full {e_on_full[s][1]:.10f}  "
             f"dE {e_on_full[s][0] - e_on_full[s][1]:+.3e} Eh")
        emit(f"            max|dln(alpha)| between arms {dln:.3e} "
             f"({(2.718281828459045 ** dln - 1.0) * 100:.3f}% in alpha)")

        pts_w.writerow([shell, lbl, seed, "cont", g_c, j_c, f"{w_c:.1f}",
                        int(g_c >= CMA_MAX_GENS),
                        f"{e_c:.10f}", f"{e_on_full[s][0]:.10f}", f"{dln:.6e}"])
        pts_w.writerow([shell, lbl, seed, "full", g_f, j_f, f"{w_f:.1f}",
                        int(g_f >= CMA_MAX_GENS),
                        f"{e_f:.10f}", f"{e_on_full[s][1]:.10f}", f"{dln:.6e}"])
        pts_f.flush()

        # the converged bases are the object of the whole test — persist both, scored on
        # the common uncontracted objective so the saved ENERGY field is comparable
        for a in range(2):
            out = rescored[a].copy(no_energy=True)
            out.energy = e_on_full[s][a]
            out.save(RESULTS_DIR, f"opt_shell{shell}_seed{seed}_{('cont', 'full')[a]}", overwrite=True)

        if g_c >= CMA_MAX_GENS or g_f >= CMA_MAX_GENS:
            emit(f"            [WARNING] hit the {CMA_MAX_GENS}-gen cap "
                 f"(cont {g_c >= CMA_MAX_GENS}, full {g_f >= CMA_MAX_GENS}); "
                 f"that arm stopped on budget, not convergence")

    # ─── paired comparison, plus the seed scatter it has to beat ──────────────

    dE = e_on_full[:, 0] - e_on_full[:, 1]

    dln_between = zeros(len(SEEDS), dtype=float64)
    for s in range(len(SEEDS)):
        dln_between[s] = float(npabs(ln_cont[s] - ln_full[s]).max())

    # within-arm scatter: per-exponent spread across seeds, worst exponent
    dln_within_cont = float(ln_cont.std(axis=0).max()) if len(SEEDS) > 1 else 0.0
    dln_within_full = float(ln_full.std(axis=0).max()) if len(SEEDS) > 1 else 0.0

    e_std_full = float(e_on_full[:, 1].std())
    decisive   = abs(float(dE.mean())) > max(e_std_full, 1e-12)
    verdict    = ("contraction cost EXCEEDS seed scatter" if decisive
                  else "contraction cost WITHIN seed scatter")

    emit("")
    emit(f"  --- shell {shell} ({lbl}) over {len(SEEDS)} seeds ---")
    emit(f"  E on full objective   cont arm  {e_on_full[:, 0].mean():.10f} +/- {e_on_full[:, 0].std():.3e}")
    emit(f"                        full arm  {e_on_full[:, 1].mean():.10f} +/- {e_on_full[:, 1].std():.3e}")
    emit(f"  dE (cont - full)      mean {dE.mean():+.3e}   std {dE.std():.3e}   "
         f"worst {dE[npabs(dE).argmax()]:+.3e} Eh")
    emit(f"  exponents  between arms  mean {dln_between.mean():.3e}  worst {dln_between.max():.3e}")
    emit(f"             within  cont    {dln_within_cont:.3e}      within full  {dln_within_full:.3e}")
    emit(f"  jobs       cont {int(jobs[0])}   full {int(jobs[1])}")
    emit(f"  VERDICT: {verdict}")
    emit(f"           (|mean dE| {abs(float(dE.mean())):.3e} vs full-arm seed std {e_std_full:.3e})")
    emit("")

    sum_w.writerow([
        shell, lbl, n_exp, n_params, len(SEEDS),
        f"{e_on_full[:, 0].mean():.10f}", f"{e_on_full[:, 0].std():.10e}",
        f"{e_on_full[:, 1].mean():.10f}", f"{e_on_full[:, 1].std():.10e}",
        f"{dE.mean():+.10e}", f"{dE.std():.10e}", f"{dE[npabs(dE).argmax()]:+.10e}",
        f"{dln_between.mean():.6e}", f"{dln_between.max():.6e}",
        f"{dln_within_cont:.6e}", f"{dln_within_full:.6e}",
        int(jobs[0]), int(jobs[1]), verdict,
    ])
    sum_f.flush()

pts_f.close()
sum_f.close()
trace_f.close()
emit(f"saved {RESULTS_DIR / 'contraction_optimum.csv'}        (one row per shell)")
emit(f"saved {RESULTS_DIR / 'contraction_optimum_runs.csv'}   (one row per run)")
emit(f"saved {RESULTS_DIR / 'contraction_optimum_trace.csv'}  (every generation of every run)")
emit(f"saved {RESULTS_DIR / 'contraction_optimum.log'}")
emit(f"saved opt_shell*_seed*_{{cont,full}}.expo in {RESULTS_DIR}  (both converged bases per seed)")
emit("")
emit("Read dE against the full arm's seed std. A contraction cost smaller than the")
emit("optimizer's own seed-to-seed scatter is not a measurable cost. The exponent")
emit("difference is reported separately because a flat valley lets the two arms land far")
emit("apart in alpha while agreeing on energy — that is a degenerate optimum, not a failure.")
run_log.close()
