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

    # scheduling strategy. Everything else — staging, contraction bootstrap, the objectives,
    # Shell_Optimization, the global evals, the reporting — is shared, so a joint/cyclic
    # comparison isolates the scheduler and nothing else.
    #   "joint"  : every optimized shell runs concurrently against the backdrop built ONCE
    #              before the run; shells never see each other mid-run (unless
    #              enable_cross_shell) and only meet in the periodic global evals.
    #   "cyclic" : block Gauss-Seidel. One shell at a time, its best committed into the
    #              shared basis before the next shell starts, so each shell optimizes
    #              against every earlier commit. No parallelism across shells.
    mode: str = "joint"

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

    # ── cyclic mode only (ignored when mode == "joint") ──
    cyclic_cycles:         int   = 10     # full passes over the optimized shells
    cyclic_gens_per_visit: int   = 100     # generation ceiling for one shell visit
    cyclic_use_stopping:   bool  = True   # commit early if that shell's last 5 gen bests agree to cyclic_stop_tol
    cyclic_stop_tol:       float = 1e-6
    cyclic_warm_restart:   bool  = True   # continue the shell's CMA from its previous visit (mean, sigma, C and
                                          # both evolution paths); False restarts at `sigma` every visit, which
                                          # re-pays step-size adaptation each time and handicaps cyclic
    cyclic_shell_order:    list | None = None   # visit order, e.g. [4, 3, 2, 1, 0]; None -> ascending l.
                                                # Coordinate-descent results depend on it, so it is a knob.

    # CMA-ES seed handed to every per-shell optimizer unchanged. int -> reproducible per-shell
    # trajectories; None -> random. In joint mode global-eval timing stays wall-clock dependent,
    # so the sequence of global evals is NOT reproducible even with a seed set; cyclic mode is
    # fully deterministic under a seed (one shell at a time, fixed order).
    seed: int | None = None

    # Subdirectory of submit_dir/results to write this run's outputs into. Empty means
    # results/ itself, which is the historical behaviour. Set it when several runs share a
    # submit dir — otherwise each run overwrites the previous one's trace, log and best.expo.
    results_subdir: str = ""


MODES = ("joint", "cyclic")


def per_shell(value, n: int, name: str) -> list:
    """Broadcast an int, or validate a list, to one value per shell."""
    if isinstance(value, int):
        return [value] * n
    if len(value) != n:
        raise ValueError(f"{name} length {len(value)} must match OPTIMIZE_FLAGS length {n}")
    return list(value)


def job_counts(n_gen_jobs: int, n_init_jobs: int, n_full_jobs: int, use_contraction: bool) -> dict:
    """MOLCAS job counts, split by which template the job ran. A contracted job
    (template_cont, only the active shell free) and a fully uncontracted one
    (template_full) are very different amounts of compute, so a bare job total
    understates whichever mode leans on the expensive kind.

      n_gen_jobs  - per-shell CMA generation jobs   (template_cont)
      n_init_jobs - per-shell starting-point evals  (template_cont)
      n_full_jobs - global / commit evals           (template_full)

    The one bootstrap eval every run pays is included and is always uncontracted.
    With contraction off every job is uncontracted and 'contracted' is 0.

    Both schedulers report through this, so the two modes' numbers mean the same thing."""
    shell_jobs   = n_gen_jobs + n_init_jobs
    uncontracted = n_full_jobs + 1                  # +1: the shared bootstrap eval
    contracted   = shell_jobs if use_contraction else 0
    if not use_contraction:
        uncontracted += shell_jobs
    return {
        "total":        shell_jobs + n_full_jobs + 1,
        "contracted":   contracted,
        "uncontracted": uncontracted,
        "gens":         n_gen_jobs,
        "init":         n_init_jobs,
        "full":         n_full_jobs,
    }


TRACE_HEADER = ["eval_idx", "t_launch", "t_done", "jobs_total", "jobs_contracted",
                "jobs_uncontracted", "global_energy", "delta_e", "best_energy"]


def run_optimize(cfg: Optimize_Config) -> tuple[Exponent_Set, float, float]:
    SUBMIT_DIR  = Path(cfg.submit_dir).resolve()
    WORK_DIR    = (Path(cfg.work_dir) / "optimize").resolve()
    RESULTS_DIR = SUBMIT_DIR / "results" / cfg.results_subdir if cfg.results_subdir else SUBMIT_DIR / "results"

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

    if cfg.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")

    # results_subdir adds exactly ONE level under results/ and nothing else. Without this,
    # pathlib would let an absolute value replace the whole path and a ".." escape it.
    if cfg.results_subdir:
        _sub = Path(cfg.results_subdir)
        if _sub.is_absolute() or len(_sub.parts) != 1 or cfg.results_subdir in (".", ".."):
            raise ValueError(f"results_subdir must be a single plain directory name, got "
                             f"{cfg.results_subdir!r}")

    # Cross-shell coupling is a joint-mode device: it exists to un-stale the backdrop that
    # joint shells share. Cyclic refreshes the backdrop at every commit by construction, so
    # the flag would be silently ignored — say so instead.
    if cfg.mode == "cyclic" and cfg.enable_cross_shell:
        raise ValueError("enable_cross_shell has no meaning in cyclic mode: each commit already "
                         "refreshes the backdrop for the next shell. Leave it off.")

    # Cross-shell coupling only ever runs inside a global eval, which cannot start
    # before its own warmup. A coupling warmup below the global-eval warmup would be
    # silently clamped up to it, so reject that combination rather than mislead.
    if cfg.enable_cross_shell and cfg.cross_shell_warmup_gens < cfg.global_eval_warmup_gens:
        raise ValueError(
            f"cross_shell_warmup_gens ({cfg.cross_shell_warmup_gens}) must be >= global_eval_warmup_gens "
            f"({cfg.global_eval_warmup_gens}); coupling happens during a global eval and cannot "
            f"fire before global evals begin."
        )

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

    # ─── dispatch ─────────────────────────────────────────────────────────────
    # Both schedulers are peers: same signature, same setup above, differing only in
    # how they schedule the per-shell searches. Keep it that way — a joint/cyclic
    # comparison is only meaningful while everything else is shared.
    if cfg.mode == "cyclic":
        return run_cyclic(cfg, basis, base, contract_frozen, objective, full_objective,
                          init_uncontracted, flags, _gen_sizes, _threads_shell, _n_flags,
                          WORK_DIR, RESULTS_DIR)
    return run_joint(cfg, basis, base, contract_frozen, objective, full_objective,
                     init_uncontracted, flags, _gen_sizes, _threads_shell, _n_flags,
                     WORK_DIR, RESULTS_DIR)


def run_joint(cfg, basis, base, contract_frozen, objective, full_objective,
              init_uncontracted, flags, gen_sizes, threads_shell, n_flags,
              work_dir, results_dir) -> tuple[Exponent_Set, float, float]:
    """Parallel scheduler: every optimized shell runs concurrently against the backdrop
    built once before the run, and the shells meet only in the periodic global evals.
    Called by run_optimize for mode == "joint"; peer of run_cyclic."""

    GEN_CEILING = cfg.max_generations * cfg.gen_ceiling_multiplier   # hard per-shell generation cap

    # ─── initialise per-shell optimizers ──────────────────────────────────────

    optimizers: dict[int, Shell_Optimization] = {}

    for shell_idx in range(n_flags):
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
            shell_start, objective, work_dir / f"initial_shell_{shell_idx}",
            threads=1, contract_frozen_shells=contract_frozen,   # single job — extra threads would be idle
        )
        print(f"  shell {shell_idx} ({lbl}) initial energy : {init_result.energy:.10f} Eh")

        optimizers[shell_idx] = Shell_Optimization(
            init_result,
            float(init_result.energy),
            objective,
            work_dir               = work_dir / f"cma_shell_{shell_idx}",
            generation_size        = gen_sizes[shell_idx],
            sigma                  = cfg.sigma,
            max_generations        = GEN_CEILING,
            active_shell           = shell_idx,
            overwrite              = True,
            logging                = True,
            contract_frozen_shells = contract_frozen,
            use_tempering          = use_tempering_shell,
            n_tempering_params     = shell_n_tempering,
            seed                   = cfg.seed,
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
          + (f" (last {cfg.early_stop_window} globals within {cfg.early_stop_tol:.1e} Eh)" if cfg.early_stop else ""))
    print(f"Seed              : {cfg.seed if cfg.seed is not None else 'random (per-shell runs NOT reproducible)'}\n")

    # ─── start all optimizers ─────────────────────────────────────────────────

    t0 = time.time()

    for shell_idx in sorted(optimizers):
        optimizers[shell_idx].start(threads=threads_shell[shell_idx])

    # ─── global eval infrastructure ───────────────────────────────────────────

    GLOBAL_EVAL_DIR = work_dir / "global_evals"
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

    csv_f = open(results_dir / "global_trace.csv", "w", newline="")
    log_f = open(results_dir / "global.log", "w")
    csv_w = csv.writer(csv_f)

    # t_launch is when the snapshot was taken, t_done when its energy came back. The energy
    # and the job counts describe the basis at t_launch, so t_launch is the timestamp to plot
    # energy against; t_done is one full MOLCAS job later. best_energy is logged so the
    # stopping behaviour can be re-judged offline from the trace alone.
    csv_w.writerow(TRACE_HEADER + [f"shell_{idx}_gen_at_trigger" for idx in sorted(optimizers)])
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

    def total_molcas_jobs(gens: dict, n_full_evals: int) -> dict:
        """Freeze-frame job counts. Generation jobs are derived from each shell's generation
        index rather than instrumented, since they are issued inside Shell_Optimization; the
        per-shell init evals (one per optimized shell, at startup) are counted too."""
        gen_jobs = sum(max(gens.get(idx, -1) + 1, 0) * gen_sizes[idx] for idx in optimizers)
        return job_counts(gen_jobs, len(optimizers), n_full_evals, cfg.use_contraction)

    def global_eval_worker(eval_idx, snapshot, trigger_gens, t_launch):
        eval_dir = GLOBAL_EVAL_DIR / f"eval_{eval_idx:04d}"
        results  = full_objective.evaluate_batch([snapshot], work_dir=eval_dir, threads=1)
        energy   = float(results[0].energy)
        t_done   = time.time() - t0
        delta_e  = energy - E0
        jobs     = total_molcas_jobs(trigger_gens, eval_idx + 1)   # +1: this full eval just finished

        with best_lock:
            if energy < best_state["best_energy"]:
                best_state["best_energy"]     = energy
                best_state["best_exp"]        = results[0].copy(no_energy=True)
                best_state["best_exp"].energy = energy
                best_state["best_exp"].save(results_dir, "best", overwrite=True)
            best_so_far = best_state["best_energy"]

            line = (
                f"[GlobalEval {eval_idx:4d}] T {t_launch:.1f}->{t_done:.1f}s | "
                f"Jobs {jobs['total']} (c {jobs['contracted']} / u {jobs['uncontracted']}) | "
                f"E {energy:.10f} | ΔE {delta_e:+.8f} | BestE {best_so_far:.10f}"
            )
            print(line)
            log_f.write(line + "\n")
            log_f.flush()

            csv_w.writerow(
                [eval_idx, f"{t_launch:.3f}", f"{t_done:.3f}", jobs["total"],
                 jobs["contracted"], jobs["uncontracted"], energy, delta_e, best_so_far]
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
                    args=(trigger_idx, collect_combined(), dict(current_gens), time.time() - t0),
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
    global_eval_worker(trigger_idx, collect_combined(), final_gens, time.time() - t0)

    jobs = total_molcas_jobs(final_gens, trigger_idx + 1)
    summary = (
        f"[Summary] mode joint | total walltime {time.time() - t0:.1f}s\n"
        f"          MOLCAS jobs {jobs['total']}: "
        f"{jobs['contracted']} contracted (template_cont) + {jobs['uncontracted']} uncontracted (template_full)\n"
        f"          by stage: {jobs['gens']} generation + {jobs['init']} shell init + {jobs['full']} global + 1 bootstrap\n"
        f"          best E {best_state['best_energy']:.10f} (ΔE {best_state['best_energy'] - E0:+.10f})"
    )
    print(summary)
    log_f.write(summary + "\n")

    csv_f.close()
    log_f.close()

    # E0 = uncontracted energy of the target basis before optimization (for the pipeline report)
    return best_state["best_exp"], best_state["best_energy"], E0


def run_cyclic(cfg, basis, base, contract_frozen, objective, full_objective,
               init_uncontracted, flags, gen_sizes, threads_shell, n_flags,
               work_dir, results_dir) -> tuple[Exponent_Set, float, float]:
    """Block Gauss-Seidel scheduler: optimize one shell, commit it, refresh the backdrop,
    move to the next. Called by run_optimize for mode == "cyclic"; the setup it receives is
    byte-identical to what joint mode uses, so the only difference is the schedule.

    The post-commit fully-uncontracted eval does double duty — it regenerates the
    contraction for the next shell AND is the global-energy trace point, so cyclic's trace
    lands on the same axes as joint's at no extra cost."""

    shells = [i for i in range(n_flags) if flags[i] == 1 and len(basis.exponents[i]) >= 1]
    if not shells:
        raise RuntimeError("No shells to optimize after filtering.")

    if cfg.cyclic_shell_order is None:
        order = list(shells)
    else:
        order = list(cfg.cyclic_shell_order)
        if sorted(order) != sorted(shells):
            raise ValueError(f"cyclic_shell_order {order} must be a permutation of the "
                             f"optimized shells {shells}")

    E0          = float(init_uncontracted.energy)
    best_energy = E0
    best_exp    = init_uncontracted.copy(no_energy=True)
    best_exp.energy = E0

    current = base.copy(no_energy=True)   # the committed basis; carries the contraction when on

    resume          = {}                      # shell -> resume state from its last visit
    cum_gens        = {s: 0 for s in shells}   # generations that shell has run in total
    jobs            = {"gens": 0, "init": 0, "full": 0}
    global_energies = []

    print(f"\nMode              : cyclic (Gauss-Seidel, one shell at a time)")
    print(f"Visit order       : {order}  ({', '.join(L_LABELS[s] for s in order)})")
    print(f"Cycles            : {cfg.cyclic_cycles}")
    print(f"Gens per visit    : {cfg.cyclic_gens_per_visit}"
          + (f" (early commit if last 5 agree to {cfg.cyclic_stop_tol:.1e})" if cfg.cyclic_use_stopping else ""))
    print(f"Budget per shell  : {cfg.cyclic_cycles * cfg.cyclic_gens_per_visit} gens "
          f"(match this to joint's max_generations for an equal-budget run)")
    print(f"Warm restart      : {'on (CMA continued across visits: mean, sigma, C, paths)' if cfg.cyclic_warm_restart else 'off (fresh CMA each visit)'}")
    print(f"Contraction       : {'on (refreshed after every commit)' if contract_frozen else 'off (fully uncontracted)'}")
    print(f"Early stop        : {'on' if cfg.early_stop else 'off'}"
          + (f" (last {cfg.early_stop_window} commits within {cfg.early_stop_tol:.1e} Eh)" if cfg.early_stop else ""))
    print(f"Seed              : {cfg.seed if cfg.seed is not None else 'random'}\n")

    csv_f = open(results_dir / "global_trace.csv", "w", newline="")
    log_f = open(results_dir / "cyclic.log", "w")
    cyc_f = open(results_dir / "cyclic_trace.csv", "w", newline="")
    csv_w = csv.writer(csv_f)
    cyc_w = csv.writer(cyc_f)

    # same schema joint writes, so the two modes' traces plot on one axis
    csv_w.writerow(TRACE_HEADER + [f"shell_{s}_gen_at_trigger" for s in shells])
    csv_f.flush()

    # per-visit detail: the job cost split both by stage and by template, so the
    # comparison against joint can be made on whichever accounting the paper argues for
    cyc_w.writerow(["cycle", "shell", "l", "gens_run", "committed_early", "E_before", "E_after",
                    "dE_shell", "E_global", "dE_global", "best_energy", "sigma_end",
                    "jobs_gens", "jobs_init", "jobs_full", "jobs_contracted",
                    "jobs_uncontracted", "jobs_total", "t_launch", "t_done"])
    cyc_f.flush()

    def emit(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    t0        = time.time()
    eval_idx  = 0
    stop_now  = False
    e_prev    = E0

    for cycle in range(cfg.cyclic_cycles):
        emit(f"\n───── cycle {cycle + 1}/{cfg.cyclic_cycles} ─────")

        for k in range(len(order)):
            shell = order[k]
            lbl   = L_LABELS[shell]
            n_exp = len(current.exponents[shell])

            use_tempering_shell = cfg.use_tempering and n_exp > 1
            shell_n_tempering   = min(cfg.n_tempering_params, n_exp) if use_tempering_shell else cfg.n_tempering_params

            visit_dir   = work_dir / f"cycle_{cycle:03d}" / f"shell_{shell}"
            shell_start = current.copy(no_energy=True)
            if contract_frozen:
                shell_start.uncontract_shell(shell)   # active shell free, every other shell stays contracted

            init = evaluate_initial(shell_start, objective, visit_dir, threads=1,
                                    subdir_name="init", contract_frozen_shells=contract_frozen)
            jobs["init"] += 1
            e_before      = float(init.energy)

            opt = Shell_Optimization(
                init, e_before, objective,
                work_dir               = visit_dir / "cma",
                generation_size        = gen_sizes[shell],
                sigma                  = cfg.sigma,
                max_generations        = cfg.cyclic_gens_per_visit,
                active_shell           = shell,
                overwrite              = True,
                logging                = False,
                contract_frozen_shells = contract_frozen,
                use_tempering          = use_tempering_shell,
                n_tempering_params     = shell_n_tempering,
                use_stopping           = cfg.cyclic_use_stopping,
                stop_tol               = cfg.cyclic_stop_tol,
                seed                   = cfg.seed,
                resume_state           = resume.get(shell) if cfg.cyclic_warm_restart else None,
            )
            opt.start(threads=threads_shell[shell])
            opt.wait()
            if opt.exception is not None:
                emit(f"  [ABORT] shell {shell} ({lbl}) crashed in cycle {cycle}")
                csv_f.close()
                cyc_f.close()
                log_f.close()
                raise RuntimeError(f"Shell {shell} ({lbl}) optimizer crashed in cycle {cycle}; "
                                   f"aborting run.") from opt.exception

            state    = opt.get_state()
            gens_run = max(state["generation"] + 1, 0)
            e_after  = float(state["best_energy"]) if state["best_energy"] is not None else e_before
            early    = gens_run < cfg.cyclic_gens_per_visit
            jobs["gens"]    += gens_run * gen_sizes[shell]
            cum_gens[shell] += gens_run

            if cfg.cyclic_warm_restart:
                rs = opt.get_resume_state()
                if rs is not None:
                    resume[shell] = rs

            if state["best_exp"] is not None:
                current.set_shell_exponents(shell, state["best_exp"].exponents[shell])

            # propagate forward: one fully-uncontracted eval of the committed basis. Its
            # energy is the trace point; its contraction becomes the next shell's backdrop.
            combined = current.copy(no_energy=True)
            combined.uncontract_all()
            t_launch = time.time() - t0      # the instant this energy describes
            full = full_objective.evaluate_batch([combined], work_dir=visit_dir / "global", threads=1)
            jobs["full"] += 1
            e_global      = float(full[0].energy)

            if contract_frozen:
                if full[0].resulting_contraction is None:
                    raise RuntimeError(f"commit eval for shell {shell} produced no contraction; "
                                       f"cannot refresh the backdrop.")
                current = full[0].copy(no_energy=True)
                current.change_contraction(full[0].resulting_contraction)
            else:
                current = full[0].copy(no_energy=True)
                current.uncontract_all()

            if e_global < best_energy:
                best_energy     = e_global
                best_exp        = full[0].copy(no_energy=True)
                best_exp.energy = e_global
                best_exp.save(results_dir, "best", overwrite=True)

            counts    = job_counts(jobs["gens"], jobs["init"], jobs["full"], cfg.use_contraction)
            t_done    = time.time() - t0
            sigma_end = state["sigma"] if state["sigma"] is not None else float("nan")

            emit(f"  [cycle {cycle + 1} | shell {shell} ({lbl})] {gens_run:3d} gens"
                 f"{' (early)' if early else '        '} | "
                 f"E_shell {e_before:.10f} -> {e_after:.10f} ({e_after - e_before:+.2e}) | "
                 f"E_global {e_global:.10f} ({e_global - e_prev:+.2e}) | "
                 f"Best {best_energy:.10f} | sigma {sigma_end:.3e} | "
                 f"Jobs {counts['total']} (c {counts['contracted']} / u {counts['uncontracted']})")

            csv_w.writerow([eval_idx, f"{t_launch:.3f}", f"{t_done:.3f}", counts["total"],
                            counts["contracted"], counts["uncontracted"],
                            e_global, e_global - E0, best_energy]
                           + [cum_gens[s] for s in shells])
            csv_f.flush()
            cyc_w.writerow([cycle, shell, lbl, gens_run, int(early),
                            f"{e_before:.10f}", f"{e_after:.10f}", f"{e_after - e_before:+.10f}",
                            f"{e_global:.10f}", f"{e_global - E0:+.10f}", f"{best_energy:.10f}",
                            f"{sigma_end:.6e}", counts["gens"], counts["init"], counts["full"],
                            counts["contracted"], counts["uncontracted"], counts["total"],
                            f"{t_launch:.3f}", f"{t_done:.3f}"])
            cyc_f.flush()

            eval_idx += 1
            e_prev    = e_global

            global_energies.append(e_global)
            if cfg.early_stop and len(global_energies) >= cfg.early_stop_window:
                window = global_energies[-cfg.early_stop_window:]
                spread = max(window) - min(window)
                if spread < cfg.early_stop_tol:
                    emit(f"  [EarlyStop] last {cfg.early_stop_window} commits within "
                         f"{cfg.early_stop_tol:.1e} Eh (spread {spread:.2e}); stopping.")
                    stop_now = True
                    break

        if stop_now:
            break

    counts = job_counts(jobs["gens"], jobs["init"], jobs["full"], cfg.use_contraction)
    emit(f"\n[Summary] mode cyclic | total walltime {time.time() - t0:.1f}s\n"
         f"          MOLCAS jobs {counts['total']}: "
         f"{counts['contracted']} contracted (template_cont) + {counts['uncontracted']} uncontracted (template_full)\n"
         f"          by stage: {counts['gens']} generation + {counts['init']} visit init + "
         f"{counts['full']} commit + 1 bootstrap\n"
         f"          best E {best_energy:.10f} (dE {best_energy - E0:+.10f})")

    csv_f.close()
    cyc_f.close()
    log_f.close()

    return best_exp, best_energy, E0
