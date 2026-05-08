from pathlib import Path
from numpy import array, eye, exp, log, float64
from scipy.optimize import minimize, OptimizeResult
import time
from datetime import datetime

from .exponent_handler import Exponent_Set
from .objectives import Objective


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _params_to_exp(
    log_params: array,
    start_exp: Exponent_Set,
    active_mask: list[bool],
    contract_frozen_shells: bool = False,
) -> Exponent_Set:
    new_exp = start_exp.copy(no_energy=True)
    idx = 0
    for l, active in enumerate(active_mask):
        if active:
            n = new_exp.lengths[l]
            new_exp.exponents[l] = array(exp(log_params[idx : idx + n]), dtype=float64)
            idx += n
            if contract_frozen_shells:
                new_exp.contractions[l]      = eye(n, dtype=float64)
                new_exp.contracted_shells[l] = False
    if contract_frozen_shells:
        new_exp.contracted = any(new_exp.contracted_shells)
    return new_exp


def scipy_fixed_exponent_count(
    start_exp: Exponent_Set,
    start_energy: float64 | None,
    objective: Objective,
    work_dir: Path | str,
    method: str = "BFGS",
    max_iterations: int = 200,
    threads: int = 1,
    fd_step: float = 1e-4,
    *,
    active_shells: list[int] | None = None,
    contract_frozen_shells: bool = False,
    logging: bool = False,
) -> tuple[Exponent_Set, float64, OptimizeResult]:

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    n_shells = len(start_exp.exponents)

    if active_shells is not None:
        if len(active_shells) != n_shells:
            raise ValueError(
                f"active_shells length ({len(active_shells)}) must match number of shells ({n_shells})"
            )
        active_mask = [bool(v) for v in active_shells]
    else:
        active_mask = [True] * n_shells

    if all(not a for a in active_mask):
        raise ValueError("All shells are frozen — nothing to optimise.")

    frozen_mask = [not a for a in active_mask]

    if contract_frozen_shells and any(frozen_mask):
        needs_contr_run = any(
            start_exp._is_identity(start_exp.contractions[l])
            for l, frozen in enumerate(frozen_mask) if frozen
        )
        if needs_contr_run:
            if start_exp.resulting_contraction is None:
                start_exp = objective.evaluate_batch(
                    [start_exp.copy(no_energy=True)],
                    work_dir = work_dir / "initial_contraction_eval",
                    threads  = threads,
                )[0]
            if start_exp.resulting_contraction is not None:
                for l, frozen in enumerate(frozen_mask):
                    if frozen:
                        start_exp.contractions[l]      = start_exp.resulting_contraction[l].copy()
                        start_exp.contracted_shells[l] = True
                start_exp.contracted = any(start_exp.contracted_shells)
            else:
                print(
                    "[Warning] contract_frozen_shells=True but the initial run "
                    "produced no ANO contraction; frozen shells will remain uncontracted."
                )

    x0 = array([
        float(log(v))
        for l, active in enumerate(active_mask) if active
        for v in start_exp.exponents[l]
    ], dtype=float64)

    _GRADIENT_FREE = {"nelder-mead", "powell", "cobyla"}
    use_gradient   = method.lower() not in _GRADIENT_FREE

    n_params    = len(x0)
    step_count  = [0]
    eval_count  = [0]
    # --- log file ---
    log_file = work_dir / "scipy.log"
    log_f    = open(log_file, "a")
    W        = 72

    def _log(line: str):
        log_f.write(line + "\n")
        log_f.flush()
        if logging:
            print(line)

    if start_energy is None:
        if logging:
            print("  Computing start energy...")
        init_results = objective.evaluate_batch(
            [start_exp.copy(no_energy=True)],
            work_dir = work_dir / "initial_eval",
            threads  = threads,
        )
        start_energy = float(init_results[0].energy)

    ref_energy  = [float(start_energy)]
    best_exp    = [start_exp.copy(no_energy=True)]
    best_energy = [float(start_energy)]

    # header
    _log(f"\n{'=' * W}")
    _log(f"  SCIPY OPT START  {datetime.now().isoformat(timespec='seconds')}")
    _log(f"{'=' * W}")
    _log(f"  Work dir   : {work_dir}")
    _log(f"  Atom       : {start_exp.atom_name}")
    _log(f"  Method     : {method}")
    _log(f"  Parameters : {n_params}")
    _log(f"  fd_step    : {fd_step}")
    _log(f"  Max iters  : {max_iterations}")
    _log(f"  Threads    : {threads}")
    _log(f"  Start E    : {start_energy:.10f}  Hartree")
    _log(f"{'=' * W}\n")

    t0 = time.time()

    def _track_best(result_exp: Exponent_Set):
        e = float(result_exp.energy)
        if e < best_energy[0]:
            best_energy[0] = e
            best_exp[0]    = result_exp.copy(no_energy=True)

    def _log_step(f0: float, elapsed: float, current_exp: Exponent_Set, grad_norm: float | None = None):
        delta_e = f0 - ref_energy[0]

        log_diffs = array([
            d
            for l, active in enumerate(active_mask) if active
            for d in log(current_exp.exponents[l] / start_exp.exponents[l])
        ], dtype=float64)

        if len(log_diffs):
            x_change     = float(exp((log_diffs ** 2).mean() ** 0.5))
            max_x_change = float(exp(abs(log_diffs).max()))
        else:
            x_change = max_x_change = 1.0

        parts = [
            f"[Step {step_count[0]:4d}]",
            f"Evals {eval_count[0]:6d}",
            f"E {f0:14.8f}",
            f"ΔE {delta_e: .8f}",
        ]
        if grad_norm is not None:
            parts.append(f"|∇| {grad_norm:.3e}")
        parts += [
            f"Δx {x_change:.4f}",
            f"maxΔx {max_x_change:.4f}",
            f"T {_hms(elapsed)}",
        ]
        _log(" | ".join(parts))

    def _objective(log_params):
        step_count[0] += 1
        eval_count[0] += 1
        results = objective.evaluate_batch(
            [_params_to_exp(log_params, start_exp, active_mask, contract_frozen_shells)],
            work_dir=work_dir / f"step_{step_count[0]:05d}",
            threads=threads,
        )
        _track_best(results[0])
        f0 = float(results[0].energy)
        _log_step(f0, time.time() - t0, results[0])
        return f0

    def _objective_and_gradient(log_params):
        step_count[0] += 1
        eval_count[0] += 2 * n_params + 1

        points = [log_params]
        for i in range(n_params):
            pos    = log_params.copy(); pos[i] += fd_step
            neg    = log_params.copy(); neg[i] -= fd_step
            points.append(pos)
            points.append(neg)

        exps    = [_params_to_exp(p, start_exp, active_mask, contract_frozen_shells) for p in points]
        results = objective.evaluate_batch(exps, work_dir=work_dir / f"step_{step_count[0]:05d}", threads=threads)

        _track_best(results[0])
        f0 = float(results[0].energy)

        grad = array([
            (float(results[2 * i + 1].energy) - float(results[2 * i + 2].energy)) / (2.0 * fd_step)
            for i in range(n_params)
        ], dtype=float64)

        _log_step(f0, time.time() - t0, results[0], grad_norm=float((grad ** 2).sum() ** 0.5))

        return f0, grad

    if use_gradient:
        result = minimize(
            _objective_and_gradient,
            x0,
            method=method,
            jac=True,
            options={"maxiter": max_iterations, "disp": False},
        )
    else:
        result = minimize(
            _objective,
            x0,
            method=method,
            options={"maxiter": max_iterations, "disp": False},
        )

    _log(f"\n{'=' * W}")
    _log(f"  Converged  : {result.success}")
    _log(f"  Message    : {result.message}")
    _log(f"  Steps      : {step_count[0]}   Evals: {eval_count[0]}")
    _log(f"  Best E     : {best_energy[0]:.10f}  Hartree")
    _log(f"  Time       : {_hms(time.time() - t0)}")
    _log(f"{'=' * W}\n")

    log_f.close()

    return best_exp[0], float64(best_energy[0]), result
