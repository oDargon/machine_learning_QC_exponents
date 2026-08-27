import csv
import shutil
import time
from pathlib import Path
from dataclasses import dataclass, field
from threading import Lock, Thread

from ..exponent_handler import Exponent_Set
from ..objectives import Ground_Energy_Objective
from ..job_manager import Job_Manager_Config
from ..common import Executor_Type, L_LABELS
from ..cma_opt_2 import evaluate_initial
from ..cma_shell_opt import Shell_Optimization


@dataclass
class Optimize_Config:
    submit_dir: Path
    work_dir:   Path
    expo_file:  str            # basis to optimize — a name in the submit dir, or an absolute path (e.g. target's handoff)

    template_cont:  str = "temp_cont.inp"    # contracted frozen shells
    template_full:  str = "temp_full.inp"    # fully uncontracted
    run_script:     str = "run.sh"
    extract_script: str = "extract.sh"

    # per-shell optimization
    optimize_flags:    list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])  # 1 = optimize, 0 = freeze; may be shorter than n_shells
    generation_size:   list | int = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])  # int, or one entry per optimize_flags entry
    threads_per_shell: list | int = field(default_factory=lambda: [6, 6, 6, 6, 6, 6])  # int, or one entry per optimize_flags entry
    sigma:                  float = 0.1
    max_generations:        int   = 300  # target: run until every shell has reached this many gens
    gen_ceiling_multiplier: int   = 5    # hard per-shell ceiling = max_generations * this; a fast shell
                                         #   may run ahead up to the ceiling while slower shells catch up
    use_contraction: bool = True   # True : freeze + contract the other shells while optimizing one (uses template_cont)
                                   # False: optimize with everything fully uncontracted (uses template_full)
    use_tempering:      bool = True
    n_tempering_params: int  = 6

    # global (fully-uncontracted) evaluations
    threads_global:           int = 2   # max concurrent fully-uncontracted eval jobs in flight
    global_eval_warmup_gens:  int = 10  # all shells must reach this many gens before the first global eval
    global_eval_spacing_gens: int = 2   # all shells must advance this many gens between global evals

    # early stopping (on the global evals)
    early_stop:        bool  = True   # stop the whole run once the global energy has plateaued
    early_stop_window: int   = 5      # number of most-recent global evals that must agree
    early_stop_tol:    float = 1e-5   # max spread (Eh) across that window to count as converged

    # cross-shell coupling: feed the converged global contraction back to every shell optimizer as its new root
    enable_cross_shell:      bool = False
    cross_shell_warmup_gens: int  = 20   # all shells must reach this many gens before coupling starts


def per_shell(value, n: int, name: str) -> list:
    """Broadcast an int, or validate a list, to one value per shell."""
    if isinstance(value, int):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} length {len(value)} must match OPTIMIZE_FLAGS length {n}")
    return list(value)


def run_optimize(cfg: Optimize_Config) -> tuple[Exponent_Set, float]:
    SUBMIT_DIR  = Path(cfg.submit_dir).resolve()
    WORK_DIR    = (Path(cfg.work_dir) / "optimize").resolve()
    RESULTS_DIR = SUBMIT_DIR / "results"

    START_DIR = WORK_DIR / "Start"
    START_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def stage(name: str) -> Path:
        """Copy an input (start) file from the submit dir into the run's Start dir."""
        dst = START_DIR / name
        shutil.copy(SUBMIT_DIR / name, dst)
        return dst

    # basis: a name in the submit dir, or an absolute/relative path handed in from a prior stage
    expo_src = Path(cfg.expo_file)
    if not expo_src.is_absolute() and not expo_src.exists():
        expo_src = SUBMIT_DIR / cfg.expo_file
    exp_path = START_DIR / expo_src.name
    shutil.copy(expo_src, exp_path)

    template      = stage(cfg.template_cont)
    template_full = stage(cfg.template_full)
    run_scr       = stage(cfg.run_script)
    extract_scr   = stage(cfg.extract_script)

    _n_flags       = len(cfg.optimize_flags)
    _gen_sizes     = per_shell(cfg.generation_size,   _n_flags, "GENERATION_SIZE")
    _threads_shell = per_shell(cfg.threads_per_shell, _n_flags, "THREADS_PER_SHELL")

    # Cross-shell coupling only ever runs inside a global eval, which cannot start
    # before its own warmup. A coupling warmup below the global-eval warmup would be
    # silently clamped up to it, so reject that combination rather than mislead.
    if cfg.enable_cross_shell and cfg.cross_shell_warmup_gens < cfg.global_eval_warmup_gens:
        raise ValueError(
            f"cross_shell_warmup_gens ({cfg.cross_shell_warmup_gens}) must be >= global_eval_warmup_gens "
            f"({cfg.global_eval_warmup_gens}); coupling happens during a global eval and cannot "
            f"fire before global evals begin."
        )

    GEN_CEILING = cfg.max_generations * cfg.gen_ceiling_multiplier   # hard per-shell generation cap

    # ─── load basis and validate flags ────────────────────────────────────────

    basis    = Exponent_Set.from_file(exp_path)
    n_shells = len(basis.exponents)

    if _n_flags > n_shells:
        raise ValueError(f"OPTIMIZE_FLAGS has {_n_flags} entries but basis only has {n_shells} shells")

    flags = list(cfg.optimize_flags) + [0] * (n_shells - _n_flags)

    job_cfg = Job_Manager_Config(
        executor_type      = Executor_Type.LOCAL_BASH,
        execution_script   = run_scr,
        extraction_script  = extract_scr,
        overwrite_existing = True,
    )
    objective      = Ground_Energy_Objective(template,      job_cfg)
    full_objective = Ground_Energy_Objective(template_full, job_cfg)

    # ─── initial uncontracted run ─────────────────────────────────────────────

    init_source = basis.copy(no_energy=True)
    init_source.uncontract_all()   # guarantee a fully uncontracted starting point regardless of input .expo
    init_uncontracted = evaluate_initial(init_source, full_objective, WORK_DIR / "initial_uncontracted", threads=1)

    print(f"Uncontracted energy : {init_uncontracted.energy:.10f} Eh")

    # `base` is the frozen backdrop each shell optimizes against, and the source for
    # the global evals. With contraction it carries the GENANO contraction on the
    # frozen shells; without, it stays fully uncontracted. Either way the per-shell
    # optimizers use the same template_cont objective — only contract_frozen_shells
    # (and whether base is contracted) differs.
    if cfg.use_contraction:
        if init_uncontracted.resulting_contraction is None:
            raise RuntimeError("Initial uncontracted run produced no contraction.")
        print("Contraction sizes   :")
        rc = init_uncontracted.resulting_contraction
        for i in range(len(rc)):
            print(f"  shell {i} ({L_LABELS[i]}): {rc[i].shape[0]} <- {rc[i].shape[1]}")
        base = init_uncontracted.copy(no_energy=True)
        base.change_contraction(init_uncontracted.resulting_contraction)
        contract_frozen = True
    else:
        print("Contraction         : off (shells optimized fully uncontracted)")
        base = init_uncontracted.copy(no_energy=True)
        base.uncontract_all()
        contract_frozen = False

    # ─── initialise per-shell optimizers ──────────────────────────────────────

    optimizers: dict[int, Shell_Optimization] = {}

    for shell_idx in range(_n_flags):
        if flags[shell_idx] == 0:
            continue

        lbl   = L_LABELS[shell_idx]
        n_exp = len(basis.exponents[shell_idx])

        if n_exp < 1:
            print(f"  shell {shell_idx} ({lbl}): no exponents, skipping")
            continue

        # Tempering describes a shell with an m-term polynomial; a single exponent has
        # nothing to temper, so drop it and optimize the log-exponent directly (which
        # also routes the shell into the fast scalar line search).
        use_tempering_shell = cfg.use_tempering and n_exp > 1
        shell_n_tempering   = min(cfg.n_tempering_params, n_exp) if use_tempering_shell else cfg.n_tempering_params

        shell_start = base.copy(no_energy=True)
        if cfg.use_contraction:
            shell_start.uncontract_shell(shell_idx)   # active shell free, frozen shells stay contracted
        # else: base is already fully uncontracted

        init_result = evaluate_initial(
            shell_start, objective, WORK_DIR / f"initial_shell_{shell_idx}",
            threads=1, contract_frozen_shells=contract_frozen,   # single job — extra threads would be idle
        )
        print(f"  shell {shell_idx} ({lbl}) initial energy : {init_result.energy:.10f} Eh")

        optimizers[shell_idx] = Shell_Optimization(
            init_result,
            float(init_result.energy),
            objective,
            work_dir               = WORK_DIR / f"cma_shell_{shell_idx}",
            generation_size        = _gen_sizes[shell_idx],
            sigma                  = cfg.sigma,
            max_generations        = GEN_CEILING,
            active_shell           = shell_idx,
            overwrite              = True,
            logging                = True,
            contract_frozen_shells = contract_frozen,
            use_tempering          = use_tempering_shell,
            n_tempering_params     = shell_n_tempering,
        )

    if not optimizers:
        raise RuntimeError("No shells to optimize after filtering.")

    print(f"\nOptimizing shells : {sorted(optimizers)}")
    print(f"Contraction       : {'on (frozen shells contracted)' if cfg.use_contraction else 'off (fully uncontracted)'}")
    print(f"Target gens       : {cfg.max_generations} (stop once all shells reach this)")
    print(f"Per-shell ceiling : {GEN_CEILING} gens")
    print(f"Warmup            : {cfg.global_eval_warmup_gens} gens before first global eval")
    print(f"Spacing           : {cfg.global_eval_spacing_gens} gens between triggers")
    print(f"Max concurrent    : {cfg.threads_global} global evals")
    print(f"Early stop        : {'on' if cfg.early_stop else 'off'}"
          + (f" (last {cfg.early_stop_window} globals within {cfg.early_stop_tol:.1e} Eh)" if cfg.early_stop else "") + "\n")

    # ─── start all optimizers ─────────────────────────────────────────────────

    t0 = time.time()

    for shell_idx in sorted(optimizers):
        optimizers[shell_idx].start(threads=_threads_shell[shell_idx])

    # ─── global eval infrastructure ───────────────────────────────────────────

    GLOBAL_EVAL_DIR = WORK_DIR / "global_evals"
    GLOBAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    E0         = float(init_uncontracted.energy)
    best_lock  = Lock()
    best_state = {
        "best_energy": E0,
        "best_exp":    init_uncontracted.copy(no_energy=True),
    }
    best_state["best_exp"].energy = E0

    # early-stop tracking: every completed global-eval energy (completion order), and a
    # flag the poll loop watches once the recent window has plateaued (guarded by best_lock).
    global_energies: list[float] = []
    early_stop_flag = {"stop": False}

    csv_f = open(RESULTS_DIR / "global_trace.csv", "w", newline="")
    log_f = open(RESULTS_DIR / "global.log", "w")
    csv_w = csv.writer(csv_f)

    csv_w.writerow(
        ["eval_idx", "time_sec", "total_molcas_jobs", "global_energy", "delta_e"]
        + [f"shell_{idx}_gen_at_trigger" for idx in sorted(optimizers)]
    )
    csv_f.flush()

    def collect_combined() -> Exponent_Set:
        """Fully-uncontracted basis with each optimized shell's current best exponents."""
        combined = base.copy(no_energy=True)
        combined.uncontract_all()
        for idx in sorted(optimizers):
            state = optimizers[idx].get_state()
            if state["best_exp"] is not None:
                combined.set_shell_exponents(idx, state["best_exp"].exponents[idx])
        return combined

    def total_molcas_jobs(gens: dict, n_full_evals: int) -> int:
        """Freeze-frame count of MOLCAS jobs completed:
          - shell optimizers: for each optimized shell, completed generations
            (gen index + 1) times its population size
          - full (global) evals: one job each
        Excludes the few fixed single-job startup evals (bootstrap + per-shell init)."""
        shell_jobs = sum(max(gens.get(idx, -1) + 1, 0) * _gen_sizes[idx] for idx in optimizers)
        return shell_jobs + n_full_evals

    def global_eval_worker(eval_idx, snapshot, trigger_gens):
        eval_dir = GLOBAL_EVAL_DIR / f"eval_{eval_idx:04d}"
        results  = full_objective.evaluate_batch([snapshot], work_dir=eval_dir, threads=1)
        energy   = float(results[0].energy)
        elapsed  = time.time() - t0
        delta_e  = energy - E0
        jobs_done = total_molcas_jobs(trigger_gens, eval_idx + 1)   # +1: this full eval just finished

        with best_lock:
            if energy < best_state["best_energy"]:
                best_state["best_energy"]     = energy
                best_state["best_exp"]        = results[0].copy(no_energy=True)
                best_state["best_exp"].energy = energy
                best_state["best_exp"].save(RESULTS_DIR, "best", overwrite=True)
            best_so_far = best_state["best_energy"]

            line = (
                f"[GlobalEval {eval_idx:4d}] T {elapsed:.1f}s | Jobs {jobs_done} | "
                f"E {energy:.10f} | ΔE {delta_e:+.8f} | BestE {best_so_far:.10f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

            csv_w.writerow(
                [eval_idx, elapsed, jobs_done, energy, delta_e]
                + [trigger_gens.get(idx, -1) for idx in sorted(optimizers)]
            )
            csv_f.flush()

            # early stop: once the most-recent early_stop_window global energies all sit
            # within early_stop_tol of each other, the global energy has plateaued.
            global_energies.append(energy)
            if cfg.early_stop and len(global_energies) >= cfg.early_stop_window:
                window = global_energies[-cfg.early_stop_window:]
                spread = max(window) - min(window)
                if spread < cfg.early_stop_tol:
                    early_stop_flag["stop"] = True
                    msg = (f"[EarlyStop] last {cfg.early_stop_window} global evals within "
                           f"{cfg.early_stop_tol:.1e} Eh (spread {spread:.2e}); requesting stop.")
                    print(msg)
                    log_f.write(msg + "\n")
                    log_f.flush()

        # cross-shell coupling: feed the converged contraction back to every shell
        # optimizer as a new root
        if (
            cfg.enable_cross_shell
            and results[0].resulting_contraction is not None
            and all(optimizers[idx].generation >= cfg.cross_shell_warmup_gens - 1 for idx in optimizers)
        ):
            shared_exp = results[0].copy(no_energy=True)
            shared_exp.change_contraction(results[0].resulting_contraction)
            for idx in optimizers:
                optimizers[idx].update_root_exponent(shared_exp)

    # ─── poll and trigger loop ────────────────────────────────────────────────

    def all_reached_target() -> bool:
        return all(optimizers[idx].generation >= cfg.max_generations - 1 for idx in optimizers)

    def any_hit_ceiling() -> bool:
        return any(optimizers[idx].generation >= GEN_CEILING - 1 for idx in optimizers)

    def abort_if_crashed() -> None:
        """A shell optimizer crashing should be rare; if one does, stop everything and
        re-raise so the whole run fails loudly rather than limping to the ceiling."""
        for idx in optimizers:
            exc = optimizers[idx].exception
            if exc is not None:
                for other in optimizers:
                    optimizers[other].stop(wait=False)
                raise RuntimeError(f"Shell {idx} ({L_LABELS[idx]}) optimizer crashed; aborting run.") from exc

    active_evals      = []
    trigger_idx       = 0
    first_triggered   = False
    last_trigger_gens = {idx: -1 for idx in optimizers}

    # Run until every shell has reached the target. Shells that get there first keep
    # optimizing (up to the ceiling) so global evals keep firing for the whole run;
    # they are stopped once the slowest shell catches up. If any shell runs all the way
    # to the ceiling, the ">5x slower" assumption has broken — bail out immediately.
    while (any(optimizers[idx].is_running for idx in optimizers)
           and not all_reached_target() and not any_hit_ceiling() and not early_stop_flag["stop"]):
        abort_if_crashed()
        active_evals[:] = [t for t in active_evals if t.is_alive()]

        current_gens = {idx: optimizers[idx].generation for idx in optimizers}

        slots_free  = len(active_evals) < cfg.threads_global
        all_started = all(current_gens[idx] >= 0 for idx in optimizers)

        if slots_free and all_started:
            if not first_triggered:
                ready = all(current_gens[idx] >= cfg.global_eval_warmup_gens - 1 for idx in optimizers)
            else:
                ready = all(
                    current_gens[idx] >= last_trigger_gens[idx] + cfg.global_eval_spacing_gens
                    for idx in optimizers
                )

            if ready:
                t = Thread(
                    target=global_eval_worker,
                    args=(trigger_idx, collect_combined(), dict(current_gens)),
                    daemon=True,
                )
                t.start()
                active_evals.append(t)

                last_trigger_gens = dict(current_gens)
                first_triggered   = True
                trigger_idx      += 1

        time.sleep(1)

    # ─── stop stragglers and drain remaining global evals ─────────────────────

    abort_if_crashed()   # catch a crash that ended the loop (or one where it never ran)

    if any_hit_ceiling() and not all_reached_target():
        stalled = [idx for idx in optimizers if optimizers[idx].generation >= GEN_CEILING - 1]
        print(f"\n[WARNING] shell(s) {stalled} hit the {GEN_CEILING}-gen ceiling before all shells "
              f"reached the target ({cfg.max_generations}); exiting early.")

    if early_stop_flag["stop"]:
        print(f"\n[EarlyStop] global energy plateaued (last {cfg.early_stop_window} evals within "
              f"{cfg.early_stop_tol:.1e} Eh); stopping all shells.")

    for idx in optimizers:
        optimizers[idx].stop(wait=False)   # target reached (or ceiling hit); halt any shell still running
    for idx in optimizers:
        optimizers[idx].wait()
    for t in active_evals:
        t.join()

    # ─── final global eval ────────────────────────────────────────────────────

    final_gens = {idx: optimizers[idx].get_state()["generation"] for idx in sorted(optimizers)}
    global_eval_worker(trigger_idx, collect_combined(), final_gens)

    summary = (
        f"[Summary] total walltime {time.time() - t0:.1f}s | "
        f"total MOLCAS jobs {total_molcas_jobs(final_gens, trigger_idx + 1)} | "
        f"best E {best_state['best_energy']:.10f} (ΔE {best_state['best_energy'] - E0:+.10f})"
    )
    print(summary)
    log_f.write(summary + "\n")

    csv_f.close()
    log_f.close()

    return best_state["best_exp"], best_state["best_energy"]
