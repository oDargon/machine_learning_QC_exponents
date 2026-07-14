import sys
import csv
import shutil
import argparse
from pathlib import Path

_arg_parser = argparse.ArgumentParser(description="Per-shell CBS sweep: optimize M polynomial params at each N, warm-starting from the previous N")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, default=None)
_args = _arg_parser.parse_args()

SUBMIT_DIR = _args.submit_dir.resolve()
WORK_DIR   = ((_args.work_dir if _args.work_dir is not None else SUBMIT_DIR) / "CBS_Shells").resolve()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import evaluate_initial
from evo_opt.tempering import from_registry
from evo_opt.cma_shell_opt import Shell_Optimization

# ─── user config ──────────────────────────────────────────────────────────────

GENERATOR       = "polynomial"  # tempering generator; passed to from_registry
N_POLY_PARAMS   = 6             # M: number of polynomial params per shell
OPTIMIZE_GENS   = 10            # CMA generations to run at each N point
SIGMA           = 0.1
GENERATION_SIZE = 10
THREADS         = 6
USE_CONTRACTION = True          # True: bootstrap GENANO contractions, keep frozen shells contracted
A               = [4,  3,  2]   # lower bound on N primitives, one per shell
B               = [20, 15, 10]  # upper bound on N primitives, one per shell

# ─── end user config ──────────────────────────────────────────────────────────

L_LABELS = ["s", "p", "d", "f", "g", "h"]

exp_path      = SUBMIT_DIR / "Si.expo"
template      = SUBMIT_DIR / "temp_cont.inp"
template_full = SUBMIT_DIR / "temp_full.inp"
run_scr       = SUBMIT_DIR / "run.sh"
extract_scr   = SUBMIT_DIR / "extract.sh"

START_DIR = WORK_DIR / "Start"
WORK_DIR.mkdir(parents=True, exist_ok=True)
START_DIR.mkdir(parents=True, exist_ok=True)

_srcs = [exp_path, template, run_scr, extract_scr]
if USE_CONTRACTION:
    _srcs.append(template_full)
for _src in _srcs:
    shutil.copy(_src, START_DIR / _src.name)

exp_path      = START_DIR / exp_path.name
template      = START_DIR / template.name
template_full = START_DIR / template_full.name
run_scr       = START_DIR / run_scr.name
extract_scr   = START_DIR / extract_scr.name

exp      = Exponent_Set.from_file(exp_path)
n_shells = len(exp.exponents)

if len(A) != n_shells:
    raise ValueError(f"A must have {n_shells} entries (one per shell), got {len(A)}")
if len(B) != n_shells:
    raise ValueError(f"B must have {n_shells} entries (one per shell), got {len(B)}")

cfg            = Job_Manager_Config(
    executor_type      = Executor_Type.LOCAL_BASH,
    execution_script   = run_scr,
    extraction_script  = extract_scr,
    overwrite_existing = True,
)
objective      = Ground_Energy_Objective(template, cfg)
full_objective = Ground_Energy_Objective(template_full, cfg) if USE_CONTRACTION else None

if USE_CONTRACTION:
    init_uncontracted = evaluate_initial(exp, full_objective, WORK_DIR / "initial_uncontracted", threads=THREADS)
    if init_uncontracted.resulting_contraction is None:
        raise RuntimeError("Initial uncontracted run produced no contraction.")
    contracted_base = init_uncontracted.copy(no_energy=True)
    contracted_base.change_contraction(init_uncontracted.resulting_contraction)
    print(f"Bootstrap E (uncontracted) : {init_uncontracted.energy:.10f} Eh")
    print("Contraction sizes          :")
    rc = init_uncontracted.resulting_contraction
    for i in range(len(rc)):
        lbl = L_LABELS[i] if i < len(L_LABELS) else str(i)
        print(f"  shell {i} ({lbl}): {rc[i].shape[0]} <- {rc[i].shape[1]}")
else:
    contracted_base = exp.copy(no_energy=True)
    print("Contraction : off")
print()

CSV_DIR = SUBMIT_DIR / "csvs"
CSV_DIR.mkdir(exist_ok=True)


def _one_n(shell_idx, n, m_actual, prev_m_params, shell_dir):
    codec_n    = from_registry(GENERATOR, m=m_actual, n=n)
    work_start = contracted_base.copy(no_energy=True)
    if prev_m_params is not None:
        work_start.apply_params(shell_idx, codec_n, prev_m_params, n=n)
    if USE_CONTRACTION:
        work_start.uncontract_shell(shell_idx)
    else:
        work_start.uncontract_all()
    init_exps = work_start.exponents[shell_idx].copy()

    init_result  = evaluate_initial(
        work_start, objective, shell_dir / "init_run",
        threads=THREADS, contract_frozen_shells=USE_CONTRACTION,
    )
    e_initial = float(init_result.energy)

    opt = Shell_Optimization(
        init_result,
        e_initial,
        objective,
        work_dir               = shell_dir / "opt_run",
        generation_size        = GENERATION_SIZE,
        sigma                  = SIGMA,
        max_generations        = OPTIMIZE_GENS,
        active_shell           = shell_idx,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = USE_CONTRACTION,
        use_tempering          = True,
        n_tempering_params     = m_actual,
    )
    opt.start(threads=THREADS)
    opt.wait()

    state     = opt.get_state()
    e_final   = float(state["best_energy"]) if state["best_energy"] is not None else e_initial
    sigma_fin = state["sigma"]
    m_out     = opt.mean
    if m_out is None:
        m_out = codec_n.encode(init_exps)

    final_exps = None
    pct_change = None
    if state["best_exp"] is not None:
        final_exps = state["best_exp"].exponents[shell_idx].copy()
        if len(final_exps) == len(init_exps):
            pct_change = float((abs(final_exps - init_exps) / abs(init_exps)).mean() * 100.0)

    return e_initial, e_final, m_out, state["generation"], sigma_fin, pct_change, final_exps


# ─── per-shell CBS sweep ──────────────────────────────────────────────────────

for shell_idx in range(n_shells):
    lbl      = L_LABELS[shell_idx] if shell_idx < len(L_LABELS) else str(shell_idx)
    n_start  = len(exp.exponents[shell_idx])
    a_s      = A[shell_idx]
    b_s      = B[shell_idx]
    m_actual = min(N_POLY_PARAMS, n_start)

    print(f"=== Shell {shell_idx} ({lbl})  N_start={n_start}  A={a_s}  B={b_s}  M={m_actual} ===")

    if n_start <= 1:
        print(f"  Only {n_start} exponent(s), skipping.\n")
        continue

    if not (a_s <= n_start <= b_s):
        print(f"  Warning: N_start={n_start} is outside [{a_s}, {b_s}] — warm-start chain may be incomplete.")

    shell_dir = WORK_DIR / f"shell_{shell_idx}"
    shell_dir.mkdir(exist_ok=True)

    a_eff = max(a_s, m_actual)
    if a_eff != a_s:
        print(f"  Note: A={a_s} < M={m_actual}; effective lower bound raised to {a_eff}")

    csv_path = CSV_DIR / f"shell_{shell_idx}_{lbl}.csv"
    m_params_n_start = None

    def _write_row(writer, csv_f, n, e_i, e_f, sig, pct, fexps):
        row = [
            n,
            f"{e_i:.10f}",
            f"{e_f:.10f}",
            f"{e_f - e_i:.6e}",
            f"{sig:.6e}" if sig is not None else "",
            f"{pct:.4f}"  if pct is not None else "",
        ]
        if fexps is not None:
            for v in fexps:
                row.append(f"{float(v):.10e}")
        writer.writerow(row)
        csv_f.flush()

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([f"# shell={shell_idx}({lbl})  generator={GENERATOR}  M={m_actual}  N_start={n_start}  gens_per_N={OPTIMIZE_GENS}"])
        writer.writerow(["N", "E_initial", "E_final", "delta_E", "sigma_final", "mean_exp_pct_change", "exponents..."])
        csv_f.flush()

        print(f"  Downward  N={n_start} → {a_eff}")
        prev = None
        for n in range(n_start, a_eff - 1, -1):
            e_i, e_f, prev, gen, sig, pct, fexps = _one_n(shell_idx, n, m_actual, prev, shell_dir)
            _write_row(writer, csv_f, n, e_i, e_f, sig, pct, fexps)
            if n == n_start:
                m_params_n_start = prev
            marker = "  <-- N_start" if n == n_start else ""
            print(f"    N={n:3d} | E_i={e_i:.8f}  E_f={e_f:.8f}  ΔE={e_f-e_i:+.2e}  (gen {gen}){marker}", flush=True)

        if b_s > n_start:
            print(f"  Upward    N={n_start + 1} → {b_s}", flush=True)
            prev = m_params_n_start
            for n in range(n_start + 1, b_s + 1):
                e_i, e_f, prev, gen, sig, pct, fexps = _one_n(shell_idx, n, m_actual, prev, shell_dir)
                _write_row(writer, csv_f, n, e_i, e_f, sig, pct, fexps)
                print(f"    N={n:3d} | E_i={e_i:.8f}  E_f={e_f:.8f}  ΔE={e_f-e_i:+.2e}  (gen {gen})", flush=True)

    print(f"  CSV: {csv_path}\n")

print("All shells done.")
