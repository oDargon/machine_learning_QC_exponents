import csv
import argparse
from pathlib import Path
from numpy import array, float64, zeros, abs as npabs
from numpy.random import default_rng
from scipy.stats import spearmanr

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.common import L_LABELS
from evo_opt.Paper_tests.shared import (Log, bootstrap_contraction, clear_progress,
                                        local_job_config, progress, stage_inputs)

_arg_parser = argparse.ArgumentParser(description="Is freezing+contracting the other shells a faithful surrogate for a fully uncontracted calculation? Randomly perturb ONE shell and compare the two energy landscapes")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = (_args.work_dir / "Contraction_Validity").resolve()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

EXPO_FILE      = "Li.expo"
TEMPLATE_CONT  = "temp_cont.inp"    # frozen shells contracted, active shell free
TEMPLATE_FULL  = "temp_full.inp"    # fully uncontracted — the reference
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

SHELLS    = [0, 1, 2]   # each explored independently; every other shell is frozen
N_SAMPLES = 30          # random perturbations per shell (plus the unperturbed centre)
MAX_FRAC  = 0.05        # each exponent scaled by (1 + U(-MAX_FRAC, +MAX_FRAC))
SEED      = 42

THREADS      = 6
JOB_POLL_SEC = 0.1      # default is 5.0s, which quantises every batch — keep it low

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

run_log = Log(RESULTS_DIR / "contraction_validity.log")
emit    = run_log.emit


# ─── the contraction under test, generated ONCE at the unperturbed geometry ────

emit("=== bootstrap: fully uncontracted run at the starting basis -> GENANO contraction ===")
base, _, e_boot = bootstrap_contraction(basis, full_objective, WORK_DIR, threads=THREADS)
emit(f"  E (uncontracted, unperturbed): {e_boot:.10f} Eh")

n_shells  = len(base.exponents)
n_prim    = [len(base.exponents[l])  for l in range(n_shells)]
n_contr   = [base.n_contracted[l]    for l in range(n_shells)]

emit("")
emit("per-shell primitives -> contracted functions:")
for l in range(n_shells):
    emit(f"  shell {l} ({L_LABELS[l]}): {n_prim[l]:>3} -> {n_contr[l]:>3}")

# ─── per-point CSV, flushed as it goes ────────────────────────────────────────

pts_f = open(RESULTS_DIR / "contraction_points.csv", "w", newline="")
pts_w = csv.writer(pts_f)
# frac_rms/frac_max make each row self-contained: delta can be plotted against how far the
# point sits from the centre where the contraction was generated. exponents are carried so
# a bad point can be reproduced without re-deriving it from the seed.
pts_w.writerow(["shell", "l", "sample", "frac_rms", "frac_max",
                "E_uncontracted", "E_contracted", "delta", "rel_pct", "exponents"])
pts_f.flush()

SUMMARY_HEADER = [
    "shell", "l", "n_samples", "max_frac",
    "size_orig_radial", "size_orig_funcs",
    "size_run_radial", "size_run_funcs",
    "size_allcontr_radial", "size_allcontr_funcs",
    "delta_mean", "delta_std", "delta_worst", "spearman_rho", "rank_agree_frac",
    "rel_mean_pct", "rel_std_pct", "rel_worst_pct",
]
sum_f = open(RESULTS_DIR / "contraction_validity.csv", "w", newline="")
sum_w = csv.writer(sum_f)
sum_w.writerow([f"#META atom={base.atom_name} n_samples={N_SAMPLES} max_frac={MAX_FRAC} seed={SEED} "
                f"threads={THREADS} shells={SHELLS}"])
sum_w.writerow(SUMMARY_HEADER)
sum_f.flush()

rng = default_rng(SEED)

emit("")
emit(f"sampling {N_SAMPLES} random perturbations per shell at +/-{MAX_FRAC * 100:.1f}% per exponent")
emit(f"each sample costs 2 MOLCAS jobs (uncontracted reference + frozen-contracted test)")
emit("")

for shell in SHELLS:
    lbl      = L_LABELS[shell]
    base_exp = array(base.exponents[shell], dtype=float64)
    n_exp    = len(base_exp)

    # sample 0 is the unperturbed centre: the error there is the floor, since that is
    # exactly where the contraction was generated
    fracs = zeros((N_SAMPLES + 1, n_exp), dtype=float64)
    fracs[1:] = rng.uniform(-MAX_FRAC, MAX_FRAC, size=(N_SAMPLES, n_exp))

    uncontracted = []
    contracted   = []
    for i in range(N_SAMPLES + 1):
        vals = base_exp * (1.0 + fracs[i])

        # reference: everything free
        full = base.copy(no_energy=True)
        full.set_shell_exponents(shell, vals)
        full.uncontract_all()
        uncontracted.append(full)

        # test: this shell free, every other shell frozen at the bootstrap contraction.
        # set_shell_exponents resets only THIS shell to identity, so the others keep theirs.
        test = base.copy(no_energy=True)
        test.set_shell_exponents(shell, vals)
        contracted.append(test)

    emit("=" * 78)
    emit(f"shell {shell} ({lbl}): N={n_exp}, {N_SAMPLES + 1} points, {2 * (N_SAMPLES + 1)} jobs")
    emit("=" * 78)

    progress(f"  running {N_SAMPLES + 1} uncontracted...")
    res_u = full_objective.evaluate_batch(uncontracted, threads=THREADS,
                                          work_dir=WORK_DIR / f"shell_{shell}" / "uncontracted")
    progress(f"  running {N_SAMPLES + 1} frozen-contracted...")
    res_c = objective.evaluate_batch(contracted, threads=THREADS,
                                     work_dir=WORK_DIR / f"shell_{shell}" / "contracted")
    clear_progress()

    e_u = zeros(N_SAMPLES + 1, dtype=float64)
    e_c = zeros(N_SAMPLES + 1, dtype=float64)
    for i in range(N_SAMPLES + 1):
        e_u[i] = float(res_u[i].energy)
        e_c[i] = float(res_c[i].energy)

    failed = 0
    for i in range(N_SAMPLES + 1):
        if e_u[i] >= 1e6 or e_c[i] >= 1e6:
            failed += 1
    if failed:
        emit(f"  [WARNING] {failed} point(s) had a failed MOLCAS job (energy 1e6); "
             f"stats below include them and are unreliable")

    delta   = e_c - e_u
    rel_pct = delta / npabs(e_u) * 100.0

    rho, _   = spearmanr(e_u, e_c)
    rank_u   = e_u.argsort().argsort()
    rank_c   = e_c.argsort().argsort()
    agree    = float((rank_u == rank_c).mean())

    for i in range(N_SAMPLES + 1):
        vals = base_exp * (1.0 + fracs[i])
        pts_w.writerow([shell, lbl, i,
                        f"{float((fracs[i] ** 2).mean() ** 0.5):.6f}",
                        f"{float(npabs(fracs[i]).max()):.6f}",
                        f"{e_u[i]:.10f}", f"{e_c[i]:.10f}",
                        f"{delta[i]:+.10f}", f"{rel_pct[i]:+.8f}",
                        ";".join(f"{float(vals[q]):.10e}" for q in range(n_exp))])
    pts_f.flush()

    # keep the two bases worth looking at: the centre (where the contraction is exact)
    # and whichever sample the contraction got most wrong
    i_worst = int(npabs(delta).argmax())
    uncontracted[0].save(RESULTS_DIR, f"point_shell{shell}_centre", overwrite=True)
    uncontracted[i_worst].save(RESULTS_DIR, f"point_shell{shell}_worst_s{i_worst:03d}", overwrite=True)

    # basis sizes. "run" is what the contracted arm actually used: this shell's full
    # primitive set plus the contracted count for every other shell.
    size_orig_rad  = sum(n_prim[l] for l in range(n_shells))
    size_orig_fun  = sum(n_prim[l] * (2 * l + 1) for l in range(n_shells))
    size_run_rad   = n_prim[shell] + sum(n_contr[l] for l in range(n_shells) if l != shell)
    size_run_fun   = n_prim[shell] * (2 * shell + 1) + sum(n_contr[l] * (2 * l + 1)
                                                           for l in range(n_shells) if l != shell)
    size_all_rad   = sum(n_contr[l] for l in range(n_shells))
    size_all_fun   = sum(n_contr[l] * (2 * l + 1) for l in range(n_shells))

    emit(f"  basis size   original (all uncontracted) : {size_orig_rad:>4} radial  {size_orig_fun:>4} functions")
    emit(f"               as run ({lbl} free, rest contracted): {size_run_rad:>4} radial  {size_run_fun:>4} functions")
    emit(f"               fully contracted             : {size_all_rad:>4} radial  {size_all_fun:>4} functions")
    emit(f"  centre point delta (contraction exact here): {delta[0]:+.3e} Eh")
    emit(f"  delta        mean {delta.mean():+.3e}   std {delta.std():.3e}   "
         f"worst {delta[npabs(delta).argmax()]:+.3e} Eh")
    emit(f"  relative     mean {rel_pct.mean():+.6f}%  std {rel_pct.std():.6f}%  "
         f"worst {rel_pct[npabs(rel_pct).argmax()]:+.6f}%")
    emit(f"  ranking      spearman rho {rho:.6f}   exact rank agreement {agree:.1%}")
    emit("")

    sum_w.writerow([
        shell, lbl, N_SAMPLES + 1, MAX_FRAC,
        size_orig_rad, size_orig_fun, size_run_rad, size_run_fun, size_all_rad, size_all_fun,
        f"{delta.mean():+.10f}", f"{delta.std():.10f}", f"{delta[npabs(delta).argmax()]:+.10f}",
        f"{rho:.6f}", f"{agree:.4f}",
        f"{rel_pct.mean():+.8f}", f"{rel_pct.std():.8f}", f"{rel_pct[npabs(rel_pct).argmax()]:+.8f}",
    ])
    sum_f.flush()

pts_f.close()
sum_f.close()
emit(f"saved {RESULTS_DIR / 'contraction_validity.csv'}   (one row per shell)")
emit(f"saved {RESULTS_DIR / 'contraction_points.csv'}     (every sampled point, with its exponents)")
emit(f"saved {RESULTS_DIR / 'contraction_validity.log'}")
emit(f"saved point_shell*_centre.expo and point_shell*_worst_s*.expo in {RESULTS_DIR}")
emit("")
emit("spearman rho is the number that matters for CMA-ES: it only consumes the RANKING of")
emit("a generation, so a large constant energy offset is harmless but reordering is not.")
run_log.close()
