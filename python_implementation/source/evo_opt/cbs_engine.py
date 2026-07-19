import csv
import time
from pathlib import Path
from threading import Thread, Semaphore, Lock
from numpy import abs

from .exponent_handler import Exponent_Set
from .objectives import Objective
from .cma_opt_2 import evaluate_initial
from .tempering import from_registry
from .cma_shell_opt import Shell_Optimization
from .common import L_LABELS


def _one_n(
    shell_idx:             int,
    n:                     int,
    contracted_base:       Exponent_Set,
    m:                     int,
    objective:             Objective,
    work_dir:              Path,
    generator:             str,
    sigma:                 float,
    generation_size:       int,
    n_gens:                int,
    threads_per_shell:     int,
    contract_frozen_shells: bool,
    prev_params            = None,
    *,
    use_stopping: bool     = False,
) -> tuple:
    codec      = from_registry(generator, m=m, n=n)
    work_start = contracted_base.copy(no_energy=True)
    if prev_params is not None:
        work_start.apply_params(shell_idx, codec, prev_params, n=n)
    else:
        work_start.apply_params(shell_idx, codec, codec.encode(contracted_base.exponents[shell_idx]), n=n)
    if not contract_frozen_shells:
        work_start.uncontract_all()
    init_exps = work_start.exponents[shell_idx].copy()

    init_result = evaluate_initial(
        work_start, objective, work_dir,
        threads=threads_per_shell,
        subdir_name=f"s{shell_idx}",
        contract_frozen_shells=contract_frozen_shells,
    )
    e_initial = float(init_result.energy)

    opt = Shell_Optimization(
        init_result,
        e_initial,
        objective,
        work_dir               = work_dir / f"s{shell_idx}" / "opt",
        generation_size        = generation_size,
        sigma                  = sigma,
        max_generations        = n_gens,
        active_shell           = shell_idx,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = contract_frozen_shells,
        use_tempering          = True,
        n_tempering_params     = m,
        use_stopping           = use_stopping,   # early-stop once last 5 best energies agree to 1e-6
    )
    opt.start(threads=threads_per_shell)
    opt.wait()

    if opt.exception is not None:
        print(
            f"  [WARNING] shell {shell_idx} N={n}: sub-optimization crashed "
            f"({opt.exception!r}); falling back to initial energy for this point.",
            flush=True,
        )

    state     = opt.get_state()
    e_final   = float(state["best_energy"]) if state["best_energy"] is not None else e_initial
    sigma_fin = state["sigma"]
    gen_done  = state["generation"]

    mean_out   = opt.mean if opt.mean is not None else codec.encode(init_exps)

    final_exps = None
    pct_change = None
    if state["best_exp"] is not None:
        final_exps = state["best_exp"].exponents[shell_idx].copy()
        if len(final_exps) == len(init_exps):
            pct_change = float((abs(final_exps - init_exps) / abs(init_exps)).mean() * 100.0)

    return e_initial, e_final, gen_done, sigma_fin, pct_change, final_exps, mean_out


def _write_csv_row(writer, shell_idx, lbl, n, e_i, e_f, n_delta, sig, pct, gens, fexps, params, m, max_n):
    row = [
        shell_idx,
        lbl,
        n,
        f"{e_i:.10f}",
        f"{e_f:.10f}",
        f"{e_f - e_i:.6e}",
        f"{n_delta:.6e}" if n_delta is not None else "",
        f"{sig:.6e}" if sig is not None else "",
        f"{pct:.4f}"  if pct is not None else "",
        gens,
    ]
    # exponents: variable count (= N), padded with blanks to max_n so the columns
    # stay aligned with the exp_1..exp_max header
    evals = list(fexps) if fexps is not None else []
    for i in range(max_n):
        row.append(f"{float(evals[i]):.10e}" if i < len(evals) else "")
    # params: fixed width M, so they land in stable param_1..param_M columns
    pvals = list(params) if params is not None else []
    for i in range(m):
        row.append(f"{float(pvals[i]):.10e}" if i < len(pvals) else "")
    writer.writerow(row)


def _build_header(m: int, max_n: int) -> list:
    header = ["shell", "l", "N", "E_initial", "E_final", "delta_E", "delta_N", "sigma_final", "mean_exp_pct_change", "gens_to_converge"]
    header += [f"exp_{i + 1}"   for i in range(max_n)]
    header += [f"param_{i + 1}" for i in range(m)]
    return header


def run_cbs(
    exp:              Exponent_Set,
    objective:        Objective,
    full_objective:   Objective | None,
    work_dir:         Path,
    csv_dir:          Path,
    shells:           list,
    A:                list,
    B:                list,
    *,
    generator:             str   = "polynomial",
    m:                     int   = 6,
    phase1_max_gens:       int   = 10,
    phase2_max_gens:       int   = 5,
    sigma:                 float = 0.1,
    generation_size:       int   = 6,
    total_threads:         int   = 1,
    threads_per_shell:     int   = 1,
    contract_frozen_shells: bool = True,
    use_stopping:          bool  = False,
) -> None:
    if len(A) != len(shells):
        raise ValueError(f"A must have {len(shells)} entries, got {len(A)}")
    if len(B) != len(shells):
        raise ValueError(f"B must have {len(shells)} entries, got {len(B)}")
    for i in range(len(shells)):
        if shells[i] >= len(L_LABELS):
            raise ValueError(f"Shell index {shells[i]} exceeds maximum supported by MOLCAS ({len(L_LABELS) - 1})")

    n_slots = max(1, total_threads // threads_per_shell)
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    t_start      = time.time()
    total_fevals = [0]
    n_done       = [0]

    total_n_points = 0
    max_n_global   = 0   # widest exponent row across all shells, for CSV column padding
    for i in range(len(shells)):
        a_eff_i = max(A[i], 1)
        b_eff_i = B[i] + max(0, a_eff_i - A[i])
        total_n_points += b_eff_i - a_eff_i + 1
        max_n_global    = max(max_n_global, b_eff_i)

    _sem   = Semaphore(n_slots)
    _lock  = Lock()

    # ── Phase 1 bootstrap ────────────────────────────────────────────────────
    if full_objective is not None:
        print("=== Phase 1: bootstrap contraction ===")
        p1_boot = evaluate_initial(
            exp, full_objective, work_dir,
            threads=total_threads,
            subdir_name="p1_bootstrap",
        )
        if p1_boot.resulting_contraction is None:
            raise RuntimeError("Phase 1 bootstrap produced no contraction.")
        p1_base = p1_boot.copy(no_energy=True)
        p1_base.change_contraction(p1_boot.resulting_contraction)
        total_fevals[0] += 1
        print(f"  Bootstrap E (uncontracted): {p1_boot.energy:.10f} Eh\n")
    else:
        p1_base = exp.copy(no_energy=True)
        print("=== Phase 1: contraction off ===\n")

    # ── Phase 1 concurrent shell optimisation ────────────────────────────────
    print("=== Phase 1: optimise all shells ===")
    phase1_results = {}   # shell_idx -> (e_i, e_f, gen, sig, pct, fexps, mean_out)

    def _phase1_worker(shell_idx):
        n_start = len(exp.exponents[shell_idx])
        with _sem:
            e_i, e_f, gen, sig, pct, fexps, mean_out = _one_n(
                shell_idx, n_start, p1_base, m,
                objective, work_dir / "phase1", generator,
                sigma, generation_size, phase1_max_gens,
                threads_per_shell, contract_frozen_shells,
                use_stopping=use_stopping,
            )
        with _lock:
            phase1_results[shell_idx]  = (e_i, e_f, gen, sig, pct, fexps, mean_out)
            total_fevals[0]           += 1 + generation_size * max(0, gen + 1)
        print(
            f"  Phase1 shell {shell_idx} ({L_LABELS[shell_idx]}): "
            f"E_i={e_i:.10f}  E_f={e_f:.10f}  (gen {gen})",
            flush=True,
        )

    p1_threads = [Thread(target=_phase1_worker, args=(shells[i],), daemon=True) for i in range(len(shells))]
    for t in p1_threads:
        t.start()
    for t in p1_threads:
        t.join()
    print()

    # ── Phase 1 → Phase 2 handoff ────────────────────────────────────────────
    assembled = exp.copy(no_energy=True)
    for i in range(len(shells)):
        s     = shells[i]
        fexps = phase1_results[s][5]
        if fexps is not None:
            assembled.set_shell_exponents(s, fexps)

    if full_objective is not None:
        print("=== Phase 1→2: re-bootstrap contraction ===")
        p2_boot = evaluate_initial(
            assembled, full_objective, work_dir,
            threads=total_threads,
            subdir_name="p2_bootstrap",
        )
        if p2_boot.resulting_contraction is None:
            raise RuntimeError("Phase 2 bootstrap produced no contraction.")
        p2_base = p2_boot.copy(no_energy=True)
        p2_base.change_contraction(p2_boot.resulting_contraction)
        total_fevals[0] += 1
        print(f"  Bootstrap E (uncontracted): {p2_boot.energy:.10f} Eh\n")
    else:
        p2_base = assembled

    # ── Phase 2 concurrent CBS sweeps ────────────────────────────────────────
    print("=== Phase 2: per-shell CBS sweep ===")

    shell_rows = {}
    for i in range(len(shells)):
        shell_rows[shells[i]] = []

    # ── live CSV: each point flushed the moment it completes, so a crash/kill
    #    mid-sweep still leaves everything computed so far. Rows land in sweep
    #    order (not sorted) and delta_N is blank — the sorted human-readable CSV
    #    written at the end fills that in. ─────────────────────────────────────
    live_path = csv_dir / "cbs_results_live.csv"
    live_f    = open(live_path, "w", newline="")
    live_w    = csv.writer(live_f)
    live_w.writerow([f"# LIVE (crash-safe, unsorted)  generator={generator}  M={m}  phase1_max_gens={phase1_max_gens}  phase2_max_gens={phase2_max_gens}  use_stopping={use_stopping}"])
    live_w.writerow(_build_header(m, max_n_global))
    live_f.flush()
    _csv_lock = Lock()

    def _write_live(shell_idx, lbl, n, e_i, e_f, sig, pct, gen, fexps, params):
        with _csv_lock:
            _write_csv_row(live_w, shell_idx, lbl, n, e_i, e_f, None, sig, pct, max(gen + 1, 0), fexps, params, m, max_n_global)
            live_f.flush()

    def _sweep_worker(shell_idx, a_bound, b_bound, phase1_mean):
        n_start = len(exp.exponents[shell_idx])
        a_eff   = max(a_bound, 1)
        cut     = max(0, a_eff - a_bound)
        b_eff   = b_bound + cut
        lbl     = L_LABELS[shell_idx]
        rows    = shell_rows[shell_idx]

        if cut > 0:
            print(f"  Shell {shell_idx} ({lbl}): lower bound raised {a_bound}→{a_eff}; upper extended {b_bound}→{b_eff}", flush=True)

        print(f"  Shell {shell_idx} ({lbl}): N_start={n_start}  [{a_eff}..{b_eff}]", flush=True)

        prev_params    = phase1_mean
        params_n_start = phase1_mean

        for n in range(n_start, a_eff - 1, -1):
            with _sem:
                e_i, e_f, gen, sig, pct, fexps, prev_params = _one_n(
                    shell_idx, n, p2_base, m,
                    objective, work_dir / "phase2", generator,
                    sigma, generation_size, phase2_max_gens,
                    threads_per_shell, contract_frozen_shells,
                    prev_params,
                    use_stopping=use_stopping,
                )
            if n == n_start:
                params_n_start = prev_params
            rows.append((n, e_i, e_f, sig, pct, fexps, prev_params, gen))
            _write_live(shell_idx, lbl, n, e_i, e_f, sig, pct, gen, fexps, prev_params)
            with _lock:
                total_fevals[0] += 1 + generation_size * max(0, gen + 1)
                n_done[0] += 1
                done = n_done[0]
            marker = "  <-- N_start" if n == n_start else ""
            print(
                f"    [{done}/{total_n_points}] Shell {shell_idx} ({lbl}) N={n:3d} | "
                f"E_i={e_i:.8f}  E_f={e_f:.8f}  ΔE={e_f-e_i:+.2e}  (gen {gen}){marker}",
                flush=True,
            )

        prev_params = params_n_start
        for n in range(max(n_start + 1, a_eff), b_eff + 1):
            with _sem:
                e_i, e_f, gen, sig, pct, fexps, prev_params = _one_n(
                    shell_idx, n, p2_base, m,
                    objective, work_dir / "phase2", generator,
                    sigma, generation_size, phase2_max_gens,
                    threads_per_shell, contract_frozen_shells,
                    prev_params,
                    use_stopping=use_stopping,
                )
            rows.append((n, e_i, e_f, sig, pct, fexps, prev_params, gen))
            _write_live(shell_idx, lbl, n, e_i, e_f, sig, pct, gen, fexps, prev_params)
            with _lock:
                total_fevals[0] += 1 + generation_size * max(0, gen + 1)
                n_done[0] += 1
                done = n_done[0]
            print(
                f"    [{done}/{total_n_points}] Shell {shell_idx} ({lbl}) N={n:3d} | "
                f"E_i={e_i:.8f}  E_f={e_f:.8f}  ΔE={e_f-e_i:+.2e}  (gen {gen})",
                flush=True,
            )

    p2_threads = [
        Thread(
            target=_sweep_worker,
            args=(shells[i], A[i], B[i], phase1_results[shells[i]][6]),
            daemon=True,
        )
        for i in range(len(shells))
    ]
    for t in p2_threads:
        t.start()
    for t in p2_threads:
        t.join()

    live_f.close()

    # ── write the human-readable CSV: one block per shell, N sorted low→high.
    #    Only reached on a successful run — the live CSV is the crash record. ──
    csv_path = csv_dir / "cbs_results.csv"
    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        elapsed = time.time() - t_start
        writer.writerow([f"# total_time={elapsed:.1f}s  total_fevals={total_fevals[0]}  generator={generator}  M={m}  phase1_max_gens={phase1_max_gens}  phase2_max_gens={phase2_max_gens}  use_stopping={use_stopping}"])
        writer.writerow(_build_header(m, max_n_global))
        for i in range(len(shells)):
            s    = shells[i]
            lbl  = L_LABELS[s]
            rows = sorted(shell_rows[s], key=lambda r: r[0])
            for j in range(len(rows)):
                n, e_i, e_f, sig, pct, fexps, params, gen = rows[j]
                n_delta = None if j == 0 else e_f - rows[j - 1][2]
                iters   = max(gen + 1, 0)   # generations actually run (gen is 0-based, -1 if none)
                _write_csv_row(writer, s, lbl, n, e_i, e_f, n_delta, sig, pct, iters, fexps, params, m, max_n_global)

    print(f"\nCSV (sorted): {csv_path}")
    print(f"CSV (live)  : {live_path}")
    print("All shells done.")
