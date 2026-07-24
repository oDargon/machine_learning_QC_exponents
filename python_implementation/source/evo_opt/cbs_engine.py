import csv
import time
import shutil
from pathlib import Path
from threading import Thread, Semaphore, Lock
from numpy import abs, array, float64, polyfit, polyval, linspace, logspace, ones, column_stack, argsort, sort, exp, sqrt, log, ceil
from numpy.linalg import lstsq

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


# ══════════════════════════════════════════════════════════════════════════════
# Stage-2 rework: per-shell CBS driver. Up-only, geom-warm-started, N >= m.
# Independent of run_cbs above; a combiner will run it across shells once reviewed.
# STAGE 1 (this commit): just the warm-started sweep. CBS fit / N* / verify come next.
# ══════════════════════════════════════════════════════════════════════════════

def _geom_predict(ns, ys, n_new, k=4):
    # fit y(N) = y_inf + A·r^N (geometric approach to an asymptote) through the k points
    # NEAREST n_new — the low end when extrapolating down, the high end when up, the
    # surrounding points when interpolating — so the local fit straddles the query. r is
    # the only nonlinearity, so grid-profile it and solve (y_inf, A) closed-form per r,
    # keeping the best. N is shifted to n_min for conditioning. Linear fallback < 3 pts.
    ns = array(ns, dtype=float64)
    ys = array(ys, dtype=float64)
    if k is not None and len(ns) > k:
        idx    = sort(argsort(abs(ns - float(n_new)))[:k])
        ns, ys = ns[idx], ys[idx]
    if len(ns) < 3:
        return float(polyval(polyfit(ns, ys, 1), n_new))

    t     = ns - ns.min()
    t_new = float(n_new) - float(ns.min())
    best  = None
    for r in linspace(0.02, 0.99, 256):
        design   = column_stack([ones(len(t)), r ** t])
        coef, *_ = lstsq(design, ys, rcond=None)
        sse      = float(((ys - design @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(r))
    _, y_inf, A, r = best
    return float(y_inf + A * r ** t_new)


def _extrapolate_start(opt_hist, n_new, n_fit_points=4):
    # predict [a0, a1] for n_new in (a0, lnβ) space, where lnβ = a1/(N-1) is N-stable
    # (it strips a1's mechanical N-growth), then reconstruct a1 = (N-1)·lnβ. The fit uses
    # the n_fit_points optima nearest n_new (locality is handled inside _geom_predict).
    ns  = array([p[0]                for p in opt_hist], dtype=float64)
    a0  = array([p[1]                for p in opt_hist], dtype=float64)
    lnb = array([p[2] / (p[0] - 1.0) for p in opt_hist], dtype=float64)
    a0_pred  = _geom_predict(ns, a0,  n_new, k=n_fit_points)
    lnb_pred = _geom_predict(ns, lnb, n_new, k=n_fit_points)
    return array([a0_pred, lnb_pred * (n_new - 1.0)], dtype=float64)


def _cma_one_n(shell, codec, n, base, start_params, objective, work_dir, sigma,
               generation_size, max_generations, threads, contract_frozen_shells,
               use_stopping, seed):
    # one CMA-ES run at fixed N, warm-started from start_params. Scratch dirs are
    # removed after. Returns (e_best, best_params, gens, e_start).
    init_dir = work_dir / f"s{shell}_N{n:02d}_init"
    cma_dir  = work_dir / f"cma_s{shell}_N{n:02d}"

    work = base.copy(no_energy=True)
    work.apply_params(shell, codec, start_params, n=n)
    if not contract_frozen_shells:
        work.uncontract_all()

    init    = evaluate_initial(work, objective, init_dir, threads=threads,
                               subdir_name="init", contract_frozen_shells=contract_frozen_shells)
    e_start = float(init.energy)

    opt = Shell_Optimization(
        init, float(init.energy), objective,
        work_dir               = cma_dir,
        generation_size        = generation_size,
        sigma                  = sigma,
        max_generations        = max_generations,
        active_shell           = shell,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = contract_frozen_shells,
        use_tempering          = True,
        n_tempering_params     = codec.m,
        use_stopping           = use_stopping,
        seed                   = seed,
    )
    opt.start(threads=threads)
    opt.wait()

    if opt.exception is not None:
        print(f"  [WARNING] shell {shell} N={n}: CMA crashed ({opt.exception!r}); "
              f"recording the initial energy.", flush=True)

    state  = opt.get_state()
    e_best = float(state["best_energy"]) if state["best_energy"] is not None else e_start
    if state["best_exp"] is not None:
        best_params = array(codec.encode(state["best_exp"].exponents[shell]), dtype=float64)
    else:
        best_params = array(start_params, dtype=float64)
    gens = max(state["generation"] + 1, 0)

    shutil.rmtree(init_dir, ignore_errors=True)
    shutil.rmtree(cma_dir,  ignore_errors=True)
    return e_best, best_params, gens, e_start


def optimize_shell_cbs(
    shell:                  int,
    base:                   Exponent_Set,
    objective:              Objective,
    work_dir:               Path,
    *,
    m:                      int        = 2,
    initial_steps:          int        = 3,
    generator:              str        = "polynomial",
    sigma:                  float      = 0.1,
    generation_size:        int        = 6,
    max_generations:        int        = 100,
    threads:                int        = 6,
    contract_frozen_shells: bool       = True,
    use_stopping:           bool       = True,
    n_fit_points:           int        = 4,
    seed:                   int | None = None,
) -> dict:
    # STAGE 1: the warm-started sweep only. Optimises N_start .. N_start+initial_steps
    # (initial_steps+1 points, >= 3), each a CMA run geom-warm-started from the previous
    # optima in (a0, lnβ) space. Returns the E(N) points; CBS fit/verify come next.
    n0  = len(base.exponents[shell])
    lbl = L_LABELS[shell]
    print(f"=== shell {shell} ({lbl}): sweep N {n0}..{n0 + initial_steps} ===", flush=True)

    opt_hist = []   # (N, a0, a1) of converged optima, for the extrapolator
    points   = []
    for n in range(n0, n0 + initial_steps + 1):
        codec = from_registry(generator, m=m, n=n)

        if len(opt_hist) >= 2:
            center, src = _extrapolate_start(opt_hist, n, n_fit_points), "geom"
        elif opt_hist:
            center, src = array([opt_hist[-1][1], opt_hist[-1][2]], dtype=float64), "prev"
        else:
            center, src = array(codec.encode(base.exponents[shell]), dtype=float64), "encode"

        e_best, best, gens, e_start = _cma_one_n(
            shell, codec, n, base, center, objective, work_dir, sigma,
            generation_size, max_generations, threads, contract_frozen_shells,
            use_stopping, seed,
        )
        points.append({"N": n, "E": e_best, "a0": float(best[0]), "a1": float(best[1]),
                       "gens": gens, "E_start": e_start, "src": src})
        opt_hist.append((n, float(best[0]), float(best[1])))
        print(f"  N={n:3d} [{src:>6}]: E={e_best:.10f}  ({gens} gens)", flush=True)

    return {"shell": shell, "l": lbl, "n_start": n0, "points": points}


# ══════════════════════════════════════════════════════════════════════════════
# Stage-2 CBS fit + N* + verify (A+C). Pure functions on (N, E) data — no CMA, no
# objective. NOT yet wired into optimize_shell_cbs; built standalone for verification.
# ══════════════════════════════════════════════════════════════════════════════

def _cbs_fit(ns, es):
    # fit E(N) = E_inf + A·exp(-b·sqrt(N)) over the given points. b is the only
    # nonlinearity, so grid-profile it and solve (E_inf, A) in closed form per b,
    # keeping the best. Returns (E_inf, A, b, r2). Needs >= 3 points.
    ns = array(ns, dtype=float64)
    es = array(es, dtype=float64)
    if len(ns) < 3:
        raise ValueError(f"_cbs_fit needs >= 3 points, got {len(ns)}")

    root = sqrt(ns)
    best = None
    for b in logspace(-1.0, 1.3, 400):        # b in ~[0.1, 20]
        design   = column_stack([ones(len(ns)), exp(-b * root)])
        coef, *_ = lstsq(design, es, rcond=None)
        sse      = float(((es - design @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(coef[1]), float(b))
    sse, e_inf, A, b = best
    ss_tot = float(((es - es.mean()) ** 2).sum())
    r2     = 1.0 - sse / ss_tot if ss_tot > 0.0 else 0.0
    return e_inf, A, b, r2


def _cbs_predict(e_inf, A, b, n):
    # the fitted ansatz evaluated at N = n
    return float(e_inf + A * exp(-b * sqrt(float(n))))


def _cbs_target_n(A, b, tol):
    # smallest N whose modeled remaining error A·exp(-b·sqrt(N)) < tol.
    # Invert: sqrt(N) > ln(A/tol)/b  ->  N = ceil((ln(A/tol)/b)^2). None if unreachable.
    if not (A > 0.0 and b > 0.0 and tol > 0.0 and A > tol):
        return None
    return int(ceil((log(A / tol) / b) ** 2))


def _cbs_uncertainty(ns, es, min_points=3):
    # refit over tail windows (drop the lowest-N point progressively, keeping the
    # asymptotic end); the spread of E_inf across windows is the error bar.
    # Returns (e_inf_lo, e_inf_hi, spread, [e_inf per window]).
    ns = list(ns)
    es = list(es)
    e_infs = []
    for start in range(0, len(ns) - min_points + 1):
        e_inf, _, _, _ = _cbs_fit(ns[start:], es[start:])
        e_infs.append(e_inf)
    lo, hi = min(e_infs), max(e_infs)
    return lo, hi, hi - lo, e_infs


def cbs_estimate(ns, es, tol, min_points=3):
    # full CBS estimate from the swept points: fit -> E_inf/A/b, tail-window error bar,
    # and the target N* where the modeled remaining error drops below tol.
    e_inf, A, b, r2        = _cbs_fit(ns, es)
    lo, hi, spread, e_infs = _cbs_uncertainty(ns, es, min_points=min_points)
    n_star                 = _cbs_target_n(A, b, tol)
    return {
        "e_inf":        e_inf,
        "A":            A,
        "b":            b,
        "r2":           r2,
        "e_inf_lo":     lo,
        "e_inf_hi":     hi,
        "e_inf_spread": spread,
        "e_inf_window": e_infs,
        "n_star":       n_star,
        "tol":          tol,
    }


def cbs_verify(estimate, ns, es, n_star, e_actual, tol):
    # A+C acceptance after jumping to n_star and optimising (e_actual).
    #   C (out-of-sample): did the PRE-jump fit predict E(n_star) to within tol?
    #   A (limit stability): does re-fitting WITH the new point leave E_inf < tol away?
    predicted = _cbs_predict(estimate["e_inf"], estimate["A"], estimate["b"], n_star)
    resid_C   = abs(e_actual - predicted)

    e_inf_new, A_new, b_new, r2_new = _cbs_fit(list(ns) + [n_star], list(es) + [e_actual])
    shift_A = abs(e_inf_new - estimate["e_inf"])

    accepted = bool(resid_C < tol and shift_A < tol)
    return {
        "accepted":  accepted,
        "resid_C":   float(resid_C),
        "shift_A":   float(shift_A),
        "predicted": predicted,
        "e_inf_new": e_inf_new,
        "A_new":     A_new,
        "b_new":     b_new,
        "r2_new":    r2_new,
    }
