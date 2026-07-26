import csv
import time
import shutil
from pathlib import Path
from threading import Thread, Semaphore, Lock
from numpy import (abs, array, float64, polyfit, polyval, linspace, logspace,
                   ones, column_stack, argsort, sort, exp, sqrt, log, ceil)
from numpy.linalg import lstsq

from .exponent_handler import Exponent_Set
from .objectives import Objective
from .cma_opt_2 import evaluate_initial
from .tempering import from_registry
from .cma_shell_opt import Shell_Optimization
from .common import L_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# Parameter extrapolation — warm-start the next N's CMA from the trend of the
# converged optima. Fit y(N) = y_inf + A·r^N (geometric approach to an asymptote).
# ══════════════════════════════════════════════════════════════════════════════

def _geom_predict(ns, ys, n_new, k=4):
    # fit y(N) = y_inf + A·r^N through the k points NEAREST n_new (low end when
    # extrapolating down, high end when up, surrounding when interpolating) so the
    # local fit straddles the query. r is the only nonlinearity — grid-profile it
    # and solve (y_inf, A) closed-form per r, keeping the best. N shifted to n_min
    # for conditioning. Linear fallback below 3 points.
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
    # (it strips a1's mechanical N-growth), then reconstruct a1 = (N-1)·lnβ.
    ns  = array([p[0]                for p in opt_hist], dtype=float64)
    a0  = array([p[1]                for p in opt_hist], dtype=float64)
    lnb = array([p[2] / (p[0] - 1.0) for p in opt_hist], dtype=float64)
    a0_pred  = _geom_predict(ns, a0,  n_new, k=n_fit_points)
    lnb_pred = _geom_predict(ns, lnb, n_new, k=n_fit_points)
    return array([a0_pred, lnb_pred * (n_new - 1.0)], dtype=float64)


# ══════════════════════════════════════════════════════════════════════════════
# One CMA-ES run at fixed N (warm-started). Scratch dirs are removed after.
# ══════════════════════════════════════════════════════════════════════════════

def _cma_one_n(shell, codec, n, base, start_params, objective, work_dir, sigma,
               generation_size, max_generations, threads, contract_frozen_shells,
               use_stopping, seed):
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


# ══════════════════════════════════════════════════════════════════════════════
# CBS energy extrapolation: fit E(N) = E_inf + A·exp(-b·sqrt(N)), predict the
# target N*, and verify. Pure functions on (N, E) data — no CMA, no objective.
# ══════════════════════════════════════════════════════════════════════════════

def _cbs_fit(ns, es):
    # fit E(N) = E_inf + A·exp(-b·sqrt(N)). b is the only nonlinearity: grid-profile
    # it and solve (E_inf, A) closed-form per b, keep the best. Returns (E_inf,A,b,r2).
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
    return float(e_inf + A * exp(-b * sqrt(float(n))))


def _cbs_target_n(A, b, tol):
    # smallest N whose modeled remaining error A·exp(-b·sqrt(N)) < tol.
    # sqrt(N) > ln(A/tol)/b  ->  N = ceil((ln(A/tol)/b)^2). None if unreachable.
    if not (A > 0.0 and b > 0.0 and tol > 0.0 and A > tol):
        return None
    return int(ceil((log(A / tol) / b) ** 2))


def _cbs_uncertainty(ns, es, min_points=3):
    # refit over tail windows (drop the lowest-N point progressively, keep the
    # asymptotic end); the spread of E_inf across windows is the error bar.
    ns = list(ns)
    es = list(es)
    e_infs = []
    for start in range(0, len(ns) - min_points + 1):
        e_inf, _, _, _ = _cbs_fit(ns[start:], es[start:])
        e_infs.append(e_inf)
    lo, hi = min(e_infs), max(e_infs)
    return lo, hi, hi - lo, e_infs


def cbs_estimate(ns, es, tol, min_points=3):
    # full CBS estimate: fit -> E_inf/A/b, tail-window error bar, and N* for tol.
    e_inf, A, b, r2        = _cbs_fit(ns, es)
    lo, hi, spread, e_infs = _cbs_uncertainty(ns, es, min_points=min_points)
    n_star                 = _cbs_target_n(A, b, tol)
    return {
        "e_inf": e_inf, "A": A, "b": b, "r2": r2,
        "e_inf_lo": lo, "e_inf_hi": hi, "e_inf_spread": spread, "e_inf_window": e_infs,
        "n_star": n_star, "tol": tol,
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
        "accepted": accepted, "resid_C": float(resid_C), "shift_A": float(shift_A),
        "predicted": predicted, "e_inf_new": e_inf_new, "A_new": A_new,
        "b_new": b_new, "r2_new": r2_new,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Component 1: per-shell converge-to-CBS engine.
#   1. sweep N_start .. N_start+initial_steps (geom-warm-started CMA)
#   2. fit E(N) -> E_inf, N*;  jump to N*, optimise, verify (A+C).
#      If rejected, the jumped point becomes data and we refit + jump again.
#   3. record each shell's converged CBS estimate.
# Writes a live CSV (every point, crash-safe) and a final compiled CSV (sorted per
# shell, plus a per-shell CBS summary line for Component 2).
# ══════════════════════════════════════════════════════════════════════════════

def run_cbs(
    base:             Exponent_Set,
    objective:        Objective,
    work_dir:         Path,
    shells:           list,
    *,
    csv_dir:                Path | None = None,   # defaults to work_dir; override to redirect
    m:                      int        = 2,
    initial_steps:          int        = 3,
    tol:                    float      = 1e-5,
    max_jumps:              int        = 6,
    n_star_cap:             int | None = None,
    generator:              str        = "polynomial",
    sigma:                  float      = 0.1,
    generation_size:        int        = 6,
    max_generations:        int        = 100,
    total_threads:          int        = 1,
    threads_per_shell:      int        = 1,
    contract_frozen_shells: bool       = True,
    use_stopping:           bool       = True,
    n_fit_points:           int        = 4,
    seed:                   int | None = None,
) -> None:
    for s in shells:
        if s >= len(L_LABELS):
            raise ValueError(f"Shell index {s} exceeds MOLCAS max ({len(L_LABELS) - 1})")
    if initial_steps < 2:
        raise ValueError("initial_steps must be >= 2 (the CBS fit needs >= 3 points)")

    # `base` arrives ready: contraction (if any) already baked in, dirs already
    # created — staging/bootstrapping is the caller's job (see cbs_sweep.py).
    csv_dir = csv_dir if csv_dir is not None else work_dir   # CSVs default to the work dir
    n_slots = max(1, total_threads // threads_per_shell)
    _sem    = Semaphore(n_slots)
    _lock   = Lock()   # protects the live writer and the results dict
    t_start = time.time()

    # ── live CSV: flushed per point, crash-safe, sweep order ──
    live_path = csv_dir / "cbs_live.csv"
    live_f    = open(live_path, "w", newline="")
    live_w    = csv.writer(live_f)
    live_w.writerow(["kind", "shell", "l", "N", "E_final", "E_start", "gens", "src", "a0", "a1"])
    live_f.flush()

    def _write_live(kind, shell, lbl, pt):
        with _lock:
            live_w.writerow([kind, shell, lbl, pt["N"], f'{pt["E"]:.10f}', f'{pt["E_start"]:.10f}',
                             pt["gens"], pt["src"], f'{pt["a0"]:.10e}', f'{pt["a1"]:.10e}'])
            live_f.flush()

    results = {}

    def _converge_shell(shell):
        n0  = len(base.exponents[shell])
        lbl = L_LABELS[shell]
        opt_hist = []   # (N, a0, a1) of converged optima, for the extrapolator
        points   = []   # dicts {N, E, a0, a1, gens, E_start, src}
        print(f"=== shell {shell} ({lbl}): N_start={n0}, sweep +{initial_steps}, tol={tol:.1e} ===", flush=True)

        def _do_point(n, kind):
            codec = from_registry(generator, m=m, n=n)
            if len(opt_hist) >= 2:
                center, src = _extrapolate_start(opt_hist, n, n_fit_points), "geom"
            elif opt_hist:
                center, src = array([opt_hist[-1][1], opt_hist[-1][2]], dtype=float64), "prev"
            else:
                center, src = array(codec.encode(base.exponents[shell]), dtype=float64), "encode"
            with _sem:
                e_best, best, gens, e_start = _cma_one_n(
                    shell, codec, n, base, center, objective, work_dir / kind, sigma,
                    generation_size, max_generations, threads_per_shell,
                    contract_frozen_shells, use_stopping, seed,
                )
            pt = {"N": n, "E": e_best, "a0": float(best[0]), "a1": float(best[1]),
                  "gens": gens, "E_start": e_start, "src": src}
            opt_hist.append((n, pt["a0"], pt["a1"]))
            points.append(pt)
            _write_live(kind, shell, lbl, pt)
            return pt

        # 1. initial sweep
        for n in range(n0, n0 + initial_steps + 1):
            pt = _do_point(n, "sweep")
            print(f"  [sweep] shell {shell} ({lbl}) N={n:3d} [{pt['src']:>6}]: E={pt['E']:.10f} ({pt['gens']} gens)", flush=True)

        # 2. jump-verify loop
        converged = False
        n_final   = points[-1]["N"]
        last_ver  = None
        for _ in range(max_jumps):
            ns  = [p["N"] for p in points]
            es  = [p["E"] for p in points]
            est = cbs_estimate(ns, es, tol)
            n_star = est["n_star"]

            if n_star is None or n_star <= max(ns):
                converged = True
                n_final   = int(n_star) if n_star is not None else int(min(ns))
                print(f"  [done ] shell {shell} ({lbl}): within tol at N*={n_final}  E_inf={est['e_inf']:.10f}", flush=True)
                break
            if n_star_cap is not None and n_star > n_star_cap:
                print(f"  [STOP ] shell {shell} ({lbl}): predicted N*={n_star} exceeds cap {n_star_cap}; stopping unconverged.", flush=True)
                break

            pt       = _do_point(n_star, "jump")
            last_ver = cbs_verify(est, ns, es, n_star, pt["E"], tol)
            tag      = "OK" if last_ver["accepted"] else "reject"
            print(f"  [jump ] shell {shell} ({lbl}) N*={n_star:3d} [{pt['src']:>6}]: E={pt['E']:.10f}  "
                  f"C={last_ver['resid_C']:.2e} A={last_ver['shift_A']:.2e} -> {tag} ({pt['gens']} gens)", flush=True)
            if last_ver["accepted"]:
                converged = True
                n_final   = n_star
                break

        ns = [p["N"] for p in points]
        es = [p["E"] for p in points]
        final_est = cbs_estimate(ns, es, tol) if len(ns) >= 3 else None
        with _lock:
            results[shell] = {
                "lbl": lbl,
                "points": sorted(points, key=lambda p: p["N"]),
                "converged": converged, "n_final": n_final,
                "estimate": final_est,
            }

    threads = [Thread(target=_converge_shell, args=(s,), daemon=True) for s in shells]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    live_f.close()

    # ── final compiled CSV: per shell in order, N-sorted points + a CBS summary
    #    line (E_inf/A/b/spread/N*) that Component 2 consumes ──
    final_path = csv_dir / "cbs_results.csv"
    with open(final_path, "w", newline="") as f:
        w       = csv.writer(f)
        elapsed = time.time() - t_start
        w.writerow([f"# CBS Component-1  total_time={elapsed:.1f}s  generator={generator}  M={m}  "
                    f"tol={tol:.1e}  initial_steps={initial_steps}  use_stopping={use_stopping}"])
        w.writerow(["kind", "shell", "l", "N", "E_final", "E_start", "gens", "src",
                    "cbs_E_inf", "cbs_A", "cbs_b", "cbs_spread", "cbs_converged", "cbs_N_star"])
        for s in shells:
            if s not in results:
                continue
            r = results[s]
            for p in r["points"]:
                w.writerow(["pt", s, r["lbl"], p["N"], f'{p["E"]:.10f}', f'{p["E_start"]:.10f}',
                            p["gens"], p["src"], "", "", "", "", "", ""])
            est = r["estimate"]
            if est is not None:
                w.writerow(["cbs", s, r["lbl"], r["n_final"], "", "", "", "",
                            f'{est["e_inf"]:.10f}', f'{est["A"]:.6e}', f'{est["b"]:.6e}',
                            f'{est["e_inf_spread"]:.3e}', int(bool(r["converged"])), r["n_final"]])

    # ── console summary ──
    print(f"\nCSV (live)  : {live_path}")
    print(f"CSV (final) : {final_path}")
    for s in shells:
        if s in results and results[s]["estimate"] is not None:
            r = results[s]
            print(f"  shell {s} ({r['lbl']}): {'CONVERGED' if r['converged'] else 'NOT converged'}  "
                  f"N*={r['n_final']}  E_inf={r['estimate']['e_inf']:.10f}  spread={r['estimate']['e_inf_spread']:.1e}")
    print("Component 1 done.")
