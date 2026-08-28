import sys
import argparse
from pathlib import Path

from evo_opt.common import CHEMICAL_ACCURACY, L_LABELS
from evo_opt.exponent_handler import Exponent_Set
from evo_opt.pipeline_core.sweep    import Sweep_Config,    run_sweep
from evo_opt.pipeline_core.target   import Target_Config,   run_target
from evo_opt.pipeline_core.optimize import Optimize_Config, run_optimize

_arg_parser = argparse.ArgumentParser(description="CBS pipeline: sweep -> target -> thorough multi-shell optimize, chained through the submit dir")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

sys.path.insert(0, str(_args.submit_dir.resolve()))

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

# ── stage control ──
SWEEP_ALREADY_MADE = False   # True: skip stage 1 and reuse an existing sweep CSV instead of running it.
                             #       Looks for cma_minima.csv (then cma_minima_live.csv, then any single
                             #       *.csv) in submit_dir/results/ first, then submit_dir/. target runs on it.

# ── shared across all stages (files looked up in the submit dir) ──
EXPO_FILE       = "Si.expo"         # starting basis the SWEEP explores; OPTIMIZE runs on target's generated .expo
TEMPLATE_CONT   = "temp_cont.inp"   # contracted frozen shells
TEMPLATE_FULL   = "temp_full.inp"   # fully uncontracted
RUN_SCRIPT      = "run.sh"
EXTRACT_SCRIPT  = "extract.sh"
USE_CONTRACTION = True



# ── stage 1: SWEEP (CMA-ES tempering convergence over N, per shell) ──
SHELLS            = [0, 1, 2, 3, 4]   # all swept independently, in parallel
N_INCREASES       = 5        # per shell: this many points beyond N_start
N_STEP            = 1        # gap between successive N points
M_PARAMS          = 2        # tempering polynomial params (2 -> 2D tempering)
GENERATOR         = "polynomial"
SWEEP_SIGMA       = 0.1
SWEEP_GEN_SIZE    = 6
SWEEP_MAX_GENS    = 100
SWEEP_STOPPING    = True     # last-5-best-energies-within-1e-6 early stop
TOTAL_THREADS     = 6        # core budget: run TOTAL_THREADS // THREADS_PER_SHELL shells at once
THREADS_PER_SHELL = 3
USE_EXTRAPOLATION = True
N_FIT_POINTS      = 4
SEED              = None      # int -> every CMA uses it; None -> random



# ── stage 2: TARGET (fit E_inf per shell, pick minimal N, emit start .expo) ──
SHELL_TOLERANCES  = {         # per-shell energy tolerance (Eh) to the CBS limit
    0: CHEMICAL_ACCURACY,     # s
    1: CHEMICAL_ACCURACY,     # p
    2: CHEMICAL_ACCURACY,     # d
    3: CHEMICAL_ACCURACY,     # f
    4: CHEMICAL_ACCURACY,     # g
}
DEFAULT_TOLERANCE = CHEMICAL_ACCURACY
CBS_MIN_POINTS    = 3
OPTIMISTIC        = True
OPTIMISTIC_FACTOR = 1.5
TARGET_N_FIT      = 4         # optima nearest the target N used for the param extrapolation



# ── stage 3: OPTIMIZE (thorough all-shells-in-parallel CMA on target's .expo) ──
# OPTIMIZE_FLAGS must be no longer than the number of shells target produces (= SHELLS above).
OPTIMIZE_FLAGS         = [1, 1, 1, 1, 1]  # 1 = optimize, 0 = freeze; may be shorter than n_shells
OPT_GEN_SIZE           = [6, 6, 6, 6, 6]  # int, or one entry per OPTIMIZE_FLAGS entry
OPT_THREADS_PER_SHELL  = [6, 6, 6, 6, 6]  # int, or one entry per OPTIMIZE_FLAGS entry
OPT_SIGMA              = 0.1
OPT_MAX_GENS           = 300  # target: run until every shell reaches this many gens
GEN_CEILING_MULTIPLIER = 5
USE_TEMPERING          = True
N_TEMPERING_PARAMS     = 6
THREADS_GLOBAL           = 2
GLOBAL_EVAL_WARMUP_GENS  = 10
GLOBAL_EVAL_SPACING_GENS = 2
EARLY_STOP        = True
EARLY_STOP_WINDOW = 5
EARLY_STOP_TOL    = 1e-5
ENABLE_CROSS_SHELL      = False
CROSS_SHELL_WARMUP_GENS = 20

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════


















# ── stage 1: sweep (or reuse an existing CSV) ──
if SWEEP_ALREADY_MADE:
    submit   = Path(_args.submit_dir).resolve()
    csv_path = None
    for name in ("cma_minima.csv", "cma_minima_live.csv"):   # prefer the ordered, then the live CSV
        for d in (submit / "results", submit):               # handoff dir first, then submit root
            cand = d / name
            if cand.exists():
                csv_path = cand
                break
        if csv_path is not None:
            break
    if csv_path is None:                                     # no standard name — fall back to any single *.csv
        for d in (submit / "results", submit):
            csvs = sorted(d.glob("*.csv"))
            if len(csvs) == 1:
                csv_path = csvs[0]
                break
            if len(csvs) > 1:
                raise SystemExit(f"SWEEP_ALREADY_MADE: multiple CSVs in {d} "
                                 f"({[c.name for c in csvs]}); rename the one to use to cma_minima.csv.")
    if csv_path is None:
        raise SystemExit(f"SWEEP_ALREADY_MADE is on but no CSV found in {submit / 'results'} or {submit}.")
    print("\n########## STAGE 1/3 : SWEEP — skipped (SWEEP_ALREADY_MADE) ##########")
    print(f"reusing existing sweep CSV: {csv_path}\n")
    e_initial = None   # initial-basis energy isn't computed when the sweep is skipped
else:
    sweep_cfg = Sweep_Config(
        submit_dir        = _args.submit_dir,
        work_dir          = _args.work_dir,
        expo_file         = EXPO_FILE,
        template_cont     = TEMPLATE_CONT,
        template_full     = TEMPLATE_FULL,
        run_script        = RUN_SCRIPT,
        extract_script    = EXTRACT_SCRIPT,
        shells            = SHELLS,
        n_increases       = N_INCREASES,
        n_step            = N_STEP,
        use_contraction   = USE_CONTRACTION,
        m_params          = M_PARAMS,
        generator         = GENERATOR,
        sigma             = SWEEP_SIGMA,
        generation_size   = SWEEP_GEN_SIZE,
        max_generations   = SWEEP_MAX_GENS,
        use_stopping      = SWEEP_STOPPING,
        total_threads     = TOTAL_THREADS,
        threads_per_shell = THREADS_PER_SHELL,
        use_extrapolation = USE_EXTRAPOLATION,
        n_fit_points      = N_FIT_POINTS,
        seed              = SEED,
    )
    print("\n########## STAGE 1/3 : SWEEP ##########\n")
    csv_path, e_initial = run_sweep(sweep_cfg)

# ── stage 2: target ──
target_cfg = Target_Config(
    results           = csv_path,
    shell_tolerances  = SHELL_TOLERANCES,
    default_tolerance = DEFAULT_TOLERANCE,
    cbs_min_points    = CBS_MIN_POINTS,
    optimistic        = OPTIMISTIC,
    optimistic_factor = OPTIMISTIC_FACTOR,
    generate_expo     = True,
    n_fit_points      = TARGET_N_FIT,
)
print("\n########## STAGE 2/3 : TARGET ##########\n")
start_expo = run_target(target_cfg)
if start_expo is None:
    raise SystemExit("target produced no starting .expo (shells not contiguous from 0, or generate_expo off); "
                     "cannot proceed to optimize.")

# ── stage 3: optimize (on target's generated .expo) ──
optimize_cfg = Optimize_Config(
    submit_dir               = _args.submit_dir,
    work_dir                 = _args.work_dir,
    expo_file                = str(start_expo),   # absolute path handed over from target
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
    use_contraction          = USE_CONTRACTION,
    use_tempering            = USE_TEMPERING,
    n_tempering_params       = N_TEMPERING_PARAMS,
    threads_global           = THREADS_GLOBAL,
    global_eval_warmup_gens  = GLOBAL_EVAL_WARMUP_GENS,
    global_eval_spacing_gens = GLOBAL_EVAL_SPACING_GENS,
    early_stop               = EARLY_STOP,
    early_stop_window        = EARLY_STOP_WINDOW,
    early_stop_tol           = EARLY_STOP_TOL,
    enable_cross_shell       = ENABLE_CROSS_SHELL,
    cross_shell_warmup_gens  = CROSS_SHELL_WARMUP_GENS,
)
print("\n########## STAGE 3/3 : OPTIMIZE ##########\n")
best_exp, best_energy, e_target_unopt = run_optimize(optimize_cfg)

# ── pipeline report: energies + basis-size reduction ──
report_lines = []
def rep(s=""):
    report_lines.append(s)
    print(s)

def basis_spec_counts(exp_set):
    """(spec string, radial count = sum N, function count = sum N*(2l+1)) for a basis."""
    n_rad  = 0
    n_func = 0
    spec   = ""
    for l in range(len(exp_set.exponents)):
        Nl      = len(exp_set.exponents[l])
        lbl     = L_LABELS[l] if l < len(L_LABELS) else str(l)
        n_rad  += Nl
        n_func += Nl * (2 * l + 1)
        spec   += f"{lbl}{Nl}"
    return spec, n_rad, n_func

submit_root  = Path(_args.submit_dir).resolve()
initial_path = submit_root / EXPO_FILE   # the input .expo we started from (structure only, no MOLCAS)
initial_basis = Exponent_Set.from_file(initial_path) if initial_path.exists() else None

spec_fin, n_rad_fin, n_func_fin = basis_spec_counts(best_exp)

rep("")
rep("#" * 64)
rep("PIPELINE REPORT")
rep("#" * 64)
rep("basis size:")
if initial_basis is not None:
    spec_init, n_rad_init, n_func_init = basis_spec_counts(initial_basis)
    d_rad  = n_rad_init  - n_rad_fin
    d_func = n_func_init - n_func_fin
    pct    = (100.0 * d_func / n_func_init) if n_func_init else 0.0
    rep(f"  initial (input)  : {spec_init}   radial={n_rad_init}   functions={n_func_init}")
    rep(f"  final (optimized): {spec_fin}   radial={n_rad_fin}   functions={n_func_fin}")
    rep(f"  reduction        : radial -{d_rad}   functions -{d_func}   ({pct:.1f}% fewer functions)")
else:
    rep(f"  initial (input)  : {EXPO_FILE} not found in submit dir — counts n/a")
    rep(f"  final (optimized): {spec_fin}   radial={n_rad_fin}   functions={n_func_fin}")

rep("")
rep("energies (Eh):")
e_init_str = f"{e_initial:.10f}" if e_initial is not None else "n/a (sweep skipped or contraction off)"
rep(f"  initial uncontracted (input basis) : {e_init_str}")
rep(f"  target basis, unoptimized          : {e_target_unopt:.10f}")
rep(f"  final optimized                    : {best_energy:.10f}")

rep("")
rep("differences (Eh):")
if e_initial is not None:
    rep(f"  reduction cost    (target_unopt - initial) : {e_target_unopt - e_initial:+.10f}")
rep(f"  optimization gain (final - target_unopt)   : {best_energy - e_target_unopt:+.10f}")
if e_initial is not None:
    rep(f"  net               (final - initial)        : {best_energy - e_initial:+.10f}")
rep("#" * 64)

report_path = submit_root / "results" / "report.txt"
report_path.write_text("\n".join(report_lines) + "\n")
print(f"\nsaved {report_path}")
print(f"best.expo saved in {submit_root / 'results'}")
