import csv
import shutil
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from threading import Thread, Semaphore, Lock
from numpy import array, float64, savez, log, ceil

from ..exponent_handler import Exponent_Set
from ..objectives import Ground_Energy_Objective
from ..job_manager import Job_Manager_Config
from ..common import Executor_Type, L_LABELS
from ..cma_opt_2 import evaluate_initial
from ..tempering import from_registry
from ..cma_shell_opt import Shell_Optimization, shell_overlap_min_eig, LIN_DEP_ENERGY
from ..cbs_engine import _extrapolate_start   # N-aware geom extrapolation (handles N gaps)


@dataclass
class Sweep_Config:
    submit_dir: Path
    work_dir:   Path

    expo_file:      str = "Si.expo"
    template_cont:  str = "temp_cont.inp"
    template_full:  str = "temp_full.inp"
    run_script:     str = "run.sh"
    extract_script: str = "extract.sh"

    shells:      list = field(default_factory=lambda: [0, 1, 2, 3, 4])   # all swept independently, in parallel
    n_increases: int  = 5        # per shell: this many points beyond N_start
    n_step:      int  = 1        # gap between successive N points (1 -> 17,18,19..; 2 -> 17,19,21..)

    use_contraction: bool = True

    m_params:        int   = 2            # 2D tempering; only 2 is supported (start extrapolation, wall search and CSV are 2-parameter)
    generator:       str   = "polynomial" # tempering generator (recorded in the CSV #META header)
    # CMA step-size (CMA adapts it internally from here). 0.1, not 0.01: a step of sigma in a1
    # moves the spacing ln beta by only sigma/(N-1), so at 0.01 the energy change per step fell
    # below stop_tol and CMA stopped after 5 generations at its start, 3-16 uEh above the true
    # minimum (Li p/f, P g). Against 61x21 (ln alpha_min, ln beta) scans of Li, B and P
    # (2026-11-20), 0.1 removed every such miss (worst +0.4 uEh) at no extra generations.
    sigma:           float = 0.1
    generation_size: int   = 6            # CMA population per generation
    max_generations: int   = 100          # hard cap; the early-stop should end well before this
    use_stopping:    bool  = True         # early stop once the last 5 generation bests agree to stop_tol
    stop_tol:        float = 1e-6         # that agreement threshold (Eh). Tighten to pin the optimum
                                          # harder; the floor is the QC code's printed precision.

    # core budget: run total_threads // threads_per_shell shells concurrently, each CMA
    # run using threads_per_shell cores.
    total_threads:     int = 1
    threads_per_shell: int = 1

    use_extrapolation: bool = True
    n_fit_points:      int  = 4       # optima nearest the query used for the local extrapolation

    # CMA-ES seed handed to every CMA run unchanged. int -> all runs use it; None -> random.
    seed: int | None = None

    # a point whose best energy is the 1e6 failure sentinel is still written to the CSV but kept
    # out of the start extrapolation; more than this many failures in a row ends that shell.
    max_consecutive_fails: int = 3

    # a shell at N=1 has no span (a1 is meaningless for one exponent), so the next N starts
    # from this guessed log-span |a1| = ln(alpha_max / alpha_min), centred on the N=1 exponent.
    n1_guess_span: float = 3.0

    # optional safety for multi-ravine shells (e.g. CASPT2 intruder ravines in B p). On the first
    # N (the second if the shell starts at N=1), after the normal CMA: step the most diffuse
    # exponent tighter (spacing fixed) until the wall: the first point after which the energy rises
    # strictly for wall_confirm exponent spacings (the ravine ripple has a period of one spacing,
    # so a ridge comes back down within it; the wall keeps rising; failed points count as rises).
    # Then run a fresh CMA from the foot of the wall and keep whichever optimum is lower.
    # Later N inherit the result via the warm start.
    wall_search:  bool  = False
    wall_confirm: float = 1.0

    # a point whose swept shell has an overlap-matrix min eigenvalue below this is not run in MOLCAS
    # but scored 2e6 (CASPT2 there is noise that can come out spuriously low, and it can hang MOLCAS).
    # B p's spurious lows had 1.4e-9 and 3.2e-9; the closest real optimum on Li/B/P has 1e-5. 0 disables.
    lin_dep_tol: float = 1e-8


def run_sweep(cfg: Sweep_Config) -> tuple[Path, float | None]:
    if cfg.m_params != 2:
        raise ValueError(f"m_params={cfg.m_params}: the sweep supports only 2 (its start extrapolation, "
                         f"wall search and CSV columns are 2-parameter)")
    SUBMIT_DIR = Path(cfg.submit_dir).resolve()
    WORK_DIR   = (Path(cfg.work_dir) / "sweep").resolve()

    START_DIR   = WORK_DIR / "Start"
    RESULTS_DIR = SUBMIT_DIR / "results"
    START_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def stage(name: str) -> Path:
        dst = START_DIR / name
        shutil.copy(SUBMIT_DIR / name, dst)
        return dst

    exp_path      = stage(cfg.expo_file)
    template_cont = stage(cfg.template_cont)
    run_scr       = stage(cfg.run_script)
    extract_scr   = stage(cfg.extract_script)
    template_full = stage(cfg.template_full) if cfg.use_contraction else None

    exp = Exponent_Set.from_file(exp_path)

    job_cfg = Job_Manager_Config(
        executor_type      = Executor_Type.LOCAL_BASH,
        execution_script   = run_scr,
        extraction_script  = extract_scr,
        overwrite_existing = True,
    )
    energy_objective = Ground_Energy_Objective(template_cont, job_cfg)
    full_objective   = Ground_Energy_Objective(template_full, job_cfg) if cfg.use_contraction else None

    # ─── frozen backdrop (contracted or not) ──────────────────────────────────

    if cfg.use_contraction:
        print("=== bootstrap contraction ===")
        boot = evaluate_initial(exp, full_objective, WORK_DIR, threads=cfg.total_threads, subdir_name="bootstrap")
        if boot.resulting_contraction is None:
            raise RuntimeError("bootstrap produced no contraction")
        base = boot.copy(no_energy=True)
        base.change_contraction(boot.resulting_contraction)
        e_initial = float(boot.energy)   # uncontracted energy of the input basis (for the pipeline report)
        print(f"  bootstrap E (uncontracted): {boot.energy:.10f} Eh")
        # the frozen backdrop every shell was swept against — inverse_target loads this
        backdrop_path = base.save(RESULTS_DIR, f"{exp.atom_name}_backdrop_contracted.expo", overwrite=True)
        print(f"  saved {backdrop_path}\n")
    else:
        base      = exp.copy(no_energy=True)
        e_initial = None   # no bootstrap eval when contraction is off

    crashed     = {}   # shell -> traceback of a crash outside the CMA (the shell's later N are missing)
    cma_crashes = []   # (shell, N) whose CMA crashed; their rows hold the best energy before the crash

    def cma_converge(shell, codec, N, start_params, seed, threads):
        """One 2D CMA-ES run at fixed N, warm-started from start_params, using `threads`
        cores. Returns (best_energy, best_params, gens, history, e_start). Dirs removed after."""
        init_dir = WORK_DIR / f"s{shell}_N{N:02d}_init"
        cma_dir  = WORK_DIR / f"cma_s{shell}_N{N:02d}"

        work = base.copy(no_energy=True)
        work.apply_params(shell, codec, start_params, n=N)
        if not cfg.use_contraction:
            work.uncontract_all()

        if shell_overlap_min_eig(work.exponents[shell], shell) < cfg.lin_dep_tol:
            init = work
            init.assign_results(energy=LIN_DEP_ENERGY)
        else:
            init = evaluate_initial(work, energy_objective, init_dir, threads=threads,
                                    subdir_name="init", contract_frozen_shells=cfg.use_contraction)
        e_start = float(init.energy)

        opt = Shell_Optimization(
            init, float(init.energy), energy_objective,
            work_dir               = cma_dir,
            generation_size        = cfg.generation_size,
            sigma                  = cfg.sigma,
            max_generations        = cfg.max_generations,
            active_shell           = shell,
            overwrite              = True,
            logging                = False,
            contract_frozen_shells = cfg.use_contraction,
            use_tempering          = True,
            n_tempering_params     = min(cfg.m_params, N),   # clamp: N<M routes to the 1-D optimiser, not a degenerate 2-D
            use_stopping           = cfg.use_stopping,
            stop_tol               = cfg.stop_tol,
            seed                   = seed,
            lin_dep_tol            = cfg.lin_dep_tol,
        )
        opt.start(threads=threads)
        opt.wait()

        if opt.exception is not None:
            print(f"  [WARNING] shell {shell} N={N}: CMA crashed ({opt.exception!r}); "
                  f"recording its best energy before the crash (the start energy if none).", flush=True)
            cma_crashes.append((shell, N))

        state  = opt.get_state()
        e_best = float(state["best_energy"]) if state["best_energy"] is not None else float(init.energy)
        if state["best_exp"] is not None:
            best_params = array(codec.encode(state["best_exp"].exponents[shell]), dtype=float64)
        else:
            best_params = array(start_params, dtype=float64)
        gens    = max(state["generation"] + 1, 0)
        history = opt.history                          # in-memory; survives the rmtree below

        shutil.rmtree(init_dir, ignore_errors=True)
        shutil.rmtree(cma_dir,  ignore_errors=True)
        return e_best, best_params, gens, history, e_start

    def wall_seek(shell, codec, N, params, e_ref, threads):
        """Walk ln(alpha_min) tighter from params at fixed spacing (a0 += step, a1 fixed) in
        batches of `threads` points, until the foot of the wall: the first probed point (the CMA
        optimum included) after which the energy rises strictly for cfg.wall_confirm spacings.
        Returns (params at the foot of the wall, report string)."""
        step     = abs(float(params[1])) / (N - 1) / 4      # a quarter of the exponent spacing
        max_k    = 40
        m        = max(1, int(ceil(cfg.wall_confirm * 4)))  # spacings -> quarter-spacing steps
        wall_dir = WORK_DIR / f"wall_s{shell}_N{N:02d}"
        shifts   = [0.0]
        energies = [float(e_ref)]
        wall_k   = None
        c        = 0                                         # next foot candidate to examine
        while wall_k is None and len(shifts) - 1 < max_k + m:
            batch_shifts = [step * (len(shifts) + i) for i in range(threads)]
            batch        = []
            for i in range(len(batch_shifts)):
                work = base.copy(no_energy=True)
                work.apply_params(shell, codec, array([float(params[0]) + batch_shifts[i], float(params[1])]), n=N)
                if not cfg.use_contraction:
                    work.uncontract_all()
                batch.append(work)
            res = energy_objective.evaluate_batch(batch, work_dir=wall_dir, threads=threads, overwrite=True)
            for i in range(len(batch_shifts)):
                shifts.append(batch_shifts[i])
                energies.append(float(res[i].energy))
            while wall_k is None and c <= max_k and c + m < len(shifts):
                rising = True
                for j in range(c + 1, c + m + 1):
                    if not (energies[j] > energies[j - 1] or energies[j] >= 1e6):   # a failure counts as a rise
                        rising = False
                if rising:
                    wall_k = c
                c += 1
        shutil.rmtree(wall_dir, ignore_errors=True)

        if wall_k is None:
            k_best = int(array(energies).argmin())
            note   = f"no wall (rise sustained over {cfg.wall_confirm} spacing) within {max_k} steps; starting from the lowest probed point"
        else:
            k_best = wall_k
            note   = (f"foot of the wall at ln(alpha_min) +{shifts[wall_k]:.3f} from the CMA optimum "
                      f"({wall_k} steps, {(energies[wall_k] - e_ref) * 1e6:+.1f} uEh vs CMA)"
                      + ("; the CMA optimum is already at the foot" if wall_k == 0 else ""))
        return array([float(params[0]) + shifts[k_best], float(params[1])], dtype=float64), note

    # ─── parallel sweep: one thread per shell, gated by a core-budget semaphore ────

    n_slots   = max(1, cfg.total_threads // cfg.threads_per_shell)
    _sem      = Semaphore(n_slots)   # at most this many CMA runs at once
    _csv_lock = Lock()               # the live CSV / row list are shared across shell threads

    HEADER = ["shell", "l", "N", "E_cma", "a0", "a1",
              "start_a0", "start_a1", "start_source", "gens_to_converge"]

    # machine-readable provenance line, first row of every CSV. cbs_target reads it
    # so the atom / generator / M need not be restated by hand downstream.
    META = (f"#META atom={exp.atom_name} generator={cfg.generator} M={cfg.m_params} "
            f"n_step={cfg.n_step} use_contraction={cfg.use_contraction} "
            f"sigma={cfg.sigma}")

    # live CSV: flushed per point in completion order (crash-safe). The ordered CSV
    # (sorted by shell, N) is written from the collected rows once everything finishes.
    live_path = RESULTS_DIR / "cma_minima_live.csv"
    live_f    = open(live_path, "w", newline="")
    live_w    = csv.writer(live_f)
    live_w.writerow([META])
    live_w.writerow(HEADER)
    live_f.flush()
    all_rows  = []

    def shell_worker(shell):
        lbl      = L_LABELS[shell]
        n0       = len(base.exponents[shell])
        n_values = [n0 + i * cfg.n_step for i in range(cfg.n_increases + 1)]
        print(f"=== shell {shell} ({lbl}): N {n_values} ===", flush=True)
        if n0 == 1 and len(n_values) > 1:
            print(f"  [NOTE] shell {shell} ({lbl}) starts at N=1: it is optimized as usual, but one exponent "
                  f"has no span, so N={n_values[1]} starts from a guessed span of {cfg.n1_guess_span} "
                  f"(n1_guess_span), centred on the N=1 exponent", flush=True)

        opt_hist = []   # (N, a0, a1) of converged optima for this shell (failed points excluded)
        n_failed = 0    # consecutive failed points
        wall_n   = n_values[1] if n0 == 1 and len(n_values) > 1 else n_values[0]
        for N in n_values:
            codec = from_registry(cfg.generator, m=cfg.m_params, n=N)

            usable = [p for p in opt_hist if p[0] >= 2]
            if opt_hist and opt_hist[-1][0] == 1:
                center, src = array([opt_hist[-1][1] + cfg.n1_guess_span / 2, -cfg.n1_guess_span], dtype=float64), "guess"
            elif cfg.use_extrapolation and len(usable) >= 3:
                center, src = _extrapolate_start(opt_hist, N, cfg.n_fit_points), "geom"
            elif usable:
                # same most diffuse exponent and neighbour ratio as the last optimum
                center, src = _extrapolate_start([usable[-1]], N), "hold"
            elif n0 == 1 and N > 1:
                # the N=1 point failed: guess around the input basis' single exponent instead
                ln_a        = float(log(base.exponents[shell][0]))
                center, src = array([ln_a + cfg.n1_guess_span / 2, -cfg.n1_guess_span], dtype=float64), "guess"
            else:
                center, src = array(codec.encode(base.exponents[shell]), dtype=float64), "encode"

            seed = cfg.seed   # every CMA gets the exact same seed (None -> CMA picks random)
            with _sem:   # hold a core slot only for the actual CMA run
                e_best, best, gens, history, e_start = cma_converge(shell, codec, N, center, seed, cfg.threads_per_shell)

            if cfg.wall_search and N == wall_n and N >= 2 and e_best < 1e6:
                with _sem:
                    w_start, note                 = wall_seek(shell, codec, N, best, e_best, cfg.threads_per_shell)
                    e_w, best_w, gens_w, hist_w, _ = cma_converge(shell, codec, N, w_start, seed, cfg.threads_per_shell)
                keep = e_w < e_best
                print(f"  [WALL] shell {shell} ({lbl}) N={N}: {note}\n"
                      f"         CMA from start {e_best:.10f} at ({best[0]:+.3f},{best[1]:+.3f}) | "
                      f"CMA from wall {e_w:.10f} at ({best_w[0]:+.3f},{best_w[1]:+.3f})  "
                      f"({(e_w - e_best) * 1e6:+.1f} uEh) -> keeping {'wall' if keep else 'start'}", flush=True)
                if keep:
                    e_best, best, gens, history = e_w, best_w, gens_w, hist_w
                    center, src                 = w_start, "wall"

            row = [shell, lbl, N, f"{e_best:.10f}", f"{best[0]:.10e}", f"{best[1]:.10e}",
                   f"{center[0]:.10e}", f"{center[1]:.10e}", src, gens]
            with _csv_lock:
                live_w.writerow(row)
                live_f.flush()
                all_rows.append(row)

            # trajectory .npz dump disabled for now — uncomment to restore per-(shell, N) trajectories
            # if history:
            #     savez(
            #         RESULTS_DIR / f"traj_shell{shell}_N{N:02d}.npz",
            #         shell=shell, l=lbl, N=N, e_start=e_start, e_final=e_best, best_params=best,
            #         start_params=array(center, dtype=float64), start_source=src,
            #         gen                 = array([h["gen"]                 for h in history]),
            #         mean                = array([h["mean"]                for h in history]),
            #         sigma               = array([h["sigma"]               for h in history]),
            #         cov                 = array([h["cov"]                 for h in history]),
            #         best_energy         = array([h["best_energy"]         for h in history]),
            #         best_energy_overall = array([h["best_energy_overall"] for h in history]),
            #     )

            print(f"  shell {shell} ({lbl}) N={N:3d} [{src:>6}]: E_cma={e_best:.10f} "
                  f"at ({best[0]:+.3f},{best[1]:+.3f})  ({gens} gens)", flush=True)

            if e_best >= 1e6:
                n_failed += 1
                why = "linearly dependent, 2e6 sentinel" if e_best >= LIN_DEP_ENERGY else "1e6 sentinel"
                print(f"  [WARNING] shell {shell} ({lbl}) N={N}: failed ({why}), "
                      f"{n_failed} in a row — left out of the extrapolation", flush=True)
                if n_failed > cfg.max_consecutive_fails:
                    print(f"  [WARNING] shell {shell} ({lbl}): more than {cfg.max_consecutive_fails} "
                          f"consecutive failures — ending this shell at N={N}", flush=True)
                    break
                continue
            n_failed = 0
            opt_hist.append((N, float(best[0]), float(best[1])))

    def shell_worker_logged(shell):
        try:
            shell_worker(shell)
        except Exception:
            crashed[shell] = traceback.format_exc()
            print(f"  [ERROR] shell {shell} ({L_LABELS[shell]}) crashed; its remaining N are missing "
                  f"from the CSV\n{crashed[shell]}", flush=True)

    threads = [Thread(target=shell_worker_logged, args=(s,), daemon=True) for s in cfg.shells]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    live_f.close()

    # ordered CSV: sorted by shell then N, written once every shell has finished
    ordered_path = RESULTS_DIR / "cma_minima.csv"
    with open(ordered_path, "w", newline="") as ord_f:
        ord_w = csv.writer(ord_f)
        ord_w.writerow([META])
        ord_w.writerow(HEADER)
        for row in sorted(all_rows, key=lambda r: (r[0], r[2])):
            ord_w.writerow(row)

    print(f"\nsaved {ordered_path}       (ordered)")
    print(f"saved {live_path}  (live, completion order)")
    for s in sorted(crashed):
        print(f"[ERROR] shell {s} ({L_LABELS[s]}) crashed mid-sweep; its rows stop before the crash "
              f"(traceback above)")
    for k in range(len(cma_crashes)):
        print(f"[WARNING] shell {cma_crashes[k][0]} N={cma_crashes[k][1]}: CMA crashed; its row holds the "
              f"best energy before the crash, not a converged optimum")
    return ordered_path, e_initial
