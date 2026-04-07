from pathlib import Path
from .exponent_handler import Exponent_Set
from .objectives import Objective
from .opt_tools_new import local_exponent_removal_analysis, exponent_difference_metrics
from numpy import exp, log, delete, float64, ndarray, eye
from numpy.linalg import eigvalsh
import shutil
from datetime import datetime
import time
import cma
import csv
import inspect

def hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def update_covariance_after_removal(C: ndarray, qnumber: tuple[int, int], exp_shape: ndarray, mode: int) -> ndarray:
    idx = sum(exp_shape[:qnumber[0]]) + qnumber[1]

    if mode == 0:
        return delete(delete(C, idx, axis=0), idx, axis=1)
    elif mode == 1:

        start = sum(exp_shape[:qnumber[0]])
        end   = start + exp_shape[qnumber[0]]
        block = C[start:end, start:end]
        scale = float(block.diagonal().mean())
        if not (scale > 0):
            scale = 1.0

        C_updated                       = C.copy()
        C_updated[start:end, :]         = 0
        C_updated[:, start:end]         = 0
        C_updated[start:end, start:end] = scale * eye(end - start)
        return delete(delete(C_updated, idx, axis=0), idx, axis=1)
    else:        
        raise ValueError(f"Invalid mode {mode} for covariance update, expected 0 or 1")


def evaluate_initial_energy(
    exp: Exponent_Set,
    objective: Objective,
    work_dir: Path | str,
    *,
    threads: int = 1,
    subdir_name: str = "initial_eval",
) -> float64:
    eval_dir = Path(work_dir).resolve() / subdir_name

    energies = objective.evaluate_batch(
        [exp],
        work_dir=eval_dir,
        threads=threads,
    )

    return float64(energies[0])


def cma_fixed_exponent_count(start_exp: Exponent_Set, start_energy: float64, objective: Objective, work_dir: Path | str, generation_size: int = 30, sigma: float = 0.1, max_generations: int = 50, threads: int = 1,
                              *, overwrite: bool = False, cma_state: list | None = None, logging: bool = False, use_stopping: bool = False) -> tuple[Exponent_Set, float64, "cma.CMAEvolutionStrategy"]:

    # sig = inspect.signature(cma_fixed_exponent_count)
    # bound = sig.bind(start_exp, start_energy, objective, work_dir,
    #                 generation_size, sigma, max_generations, threads,
    #                 overwrite=overwrite, cma_state=cma_state,
    #                 logging=logging, use_stopping=use_stopping)
    # bound.apply_defaults()

    # print(
    #     f"cma_fixed_exponent_count("
    #     + ", ".join(f"{k}={v!r}" for k, v in bound.arguments.items())
    #     + ")"
    # )



    work_dir = Path(work_dir).resolve()
    if work_dir.exists() and overwrite:
        shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    elif not work_dir.exists():
        work_dir.mkdir(parents=True)
    elif work_dir.exists() and not overwrite:
        raise FileExistsError(f"{work_dir} exists, set overwrite=True to replace it")

    best_dir = work_dir / "best_per_generation"
    best_dir.mkdir(exist_ok=True)

    csv_file         = work_dir / "cma_trace.csv"
    csv_f            = open(csv_file, "w", newline="")
    csv_writer       = csv.writer(csv_f)
    n_shells_initial = len(start_exp.exponents)

    header = [
        "generation",
        "fevals",
        "time_sec",
        "best_energy_gen",
        "best_energy_overall",
        "total_x_change",
        "max_global_x_change",
    ]

    for l in range(n_shells_initial):
        header.append(f"shell_{l}_rms_x")
        header.append(f"shell_{l}_max_x")

    csv_writer.writerow(header)
    csv_f.flush()

    x0 = log(start_exp.flatten_exps())
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': generation_size})
    t0 = time.time()

    if cma_state is not None:
        es.mean  = cma_state[0]
        # es.sigma = cma_state[1]

        Cnew = cma_state[2]
        Cnew = 0.5 * (Cnew + Cnew.T)
        evals = eigvalsh(Cnew)
        print("min eig before inject:", evals.min(), "max eig:", evals.max())
        es.sm.C  = Cnew
        # es.pc    = cma_state[3]

        es.sm.update_now()

    log_file             = work_dir / "cma.log"
    log_f                = open(log_file, "a")  # always log to file
    best_energy_overall  = start_energy
    best_exp_overall     = start_exp.copy_without_energy()
    recent_best_energies = []

    for gen in range(max_generations):
        batch_dir = work_dir / f"batch_{gen}"

        population  = es.ask()
        exp_objects = []
        for vec in population:
            new_exp = start_exp.copy_without_energy()
            new_exp.update_exponent_uncontracted_from_flat_same_shape(exp(vec))
            exp_objects.append(new_exp)

        energies    = objective.evaluate_batch(exp_objects, work_dir=batch_dir, threads=threads)
        best_idx    = int(energies.argmin())
        best_energy = energies[best_idx]
        recent_best_energies.append(float(best_energy))
        if len(recent_best_energies) > 5:
            recent_best_energies.pop(0)

        if best_energy < best_energy_overall:
            best_energy_overall = best_energy
            best_exp_overall    = exp_objects[best_idx].copy_without_energy()

        best_exp_gen = exp_objects[best_idx].copy_without_energy()
        best_exp_gen.save(best_dir, f"gen_{gen:03d}")

        es.tell(population, energies)

        # logging
        elapsed = time.time() - t0
        fevals  = es.countevals
        delta_e = best_energy - start_energy
        line = (
            f"[Gen {gen:3d}] | "
            f"Fevals {fevals:6d} | "
            f"BestE {best_energy:14.8f} | "
            f"ΔE {delta_e: .8f} | "
            f"σ {es.sigma: .3e} | "
            f"T {hms(elapsed)}"
        )

        log_f.write(line + "\n")
        log_f.flush()
        if logging:
            print(line)
        
        total_rms, per_shell_rms, max_global, per_shell_max = exponent_difference_metrics(start_exp, best_exp_gen)

        total_x_change      = float(exp(total_rms))
        max_global_x_change = float(exp(max_global))
        per_shell_rms_x     = exp(per_shell_rms)
        per_shell_max_x     = exp(per_shell_max)

        # ---- CSV row ----
        row = [
            gen,
            fevals,
            float(elapsed),
            float(best_energy),
            float(best_energy_overall),
            total_x_change,
            max_global_x_change,
        ]

        for l in range(n_shells_initial):
            row.append(float(per_shell_rms_x[l]))
            row.append(float(per_shell_max_x[l]))

        csv_writer.writerow(row)
        csv_f.flush()

        stop_reason = None

        if es.sigma < 1e-3:
            stop_reason = "sigma < 1e-3"

        elif es.sigma < 1e-2 and len(recent_best_energies) == 5:
            rounded = [round(e, 5) for e in recent_best_energies]
            if len(set(rounded)) == 1:
                stop_reason = "sigma < 1e-2 and last 5 best energies equal to 5 decimals"

        if stop_reason is not None and use_stopping:
            line = f"[STOP] {stop_reason} | Gen {gen:3d} | BestE {best_energy:14.8f} | sigma {es.sigma:.3e}"
            log_f.write(line + "\n")
            log_f.flush()
            if logging:
                print(line)
            break

        # if es.stop():
        #     break

    log_f.close()
    csv_f.close()
        

    return best_exp_overall, best_energy_overall, es


def cma_culling(
    start_exp: Exponent_Set,
    objective: Objective,
    work_dir: Path | str,
    *,
    exponents_to_cull: int       = 1,
    optimize_initial: bool       = False,
    propagate_covariance: bool   = False,
    propagation_mode: int        = 1,
    generation_size: int         = 12,
    sigma: float                 = 0.1,
    max_generations: int         = 50,
    threads: int                 = 1,
    start_energy: float64 | None = None,
    overwrite: bool              = False,
    overwrite_gens: bool         = False,
    use_stopping: bool           = False,
    logging: int                 = 0,
):

    work_dir = Path(work_dir).resolve()

    if work_dir.exists() and overwrite:
        shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    elif not work_dir.exists():
        work_dir.mkdir(parents=True)
    elif work_dir.exists() and not overwrite:
        raise FileExistsError(f"{work_dir} exists, set overwrite=True to replace it")

    culling_collection = work_dir / "culling_collection"
    opti_collection    = work_dir / "opti_collection"
    best_culled_dir    = work_dir / "best_culled"
    init_culled_dir    = work_dir / "initial_culled"
    log_file           = work_dir / "culling.log"
    run_logs_dir       = work_dir / "run_logs"
    run_csvs_dir       = work_dir / "run_csvs"

    culling_collection.mkdir(parents=True)
    opti_collection.mkdir(parents=True)
    best_culled_dir.mkdir(exist_ok=True)
    init_culled_dir.mkdir(exist_ok=True)
    run_logs_dir.mkdir(exist_ok=True)
    run_csvs_dir.mkdir(exist_ok=True)

    current_exp  = start_exp.copy_without_energy()
    if start_energy is None:
        start_energy = evaluate_initial_energy(start_exp, objective, work_dir, threads=threads)
    last_energy  = start_energy
    last_es      = None  # for covariance propagation if desired
    t0           = time.time()
    
    csv_file   = work_dir / "culling_trace.csv"
    csv_f      = open(csv_file, "w", newline="")
    csv_writer = csv.writer(csv_f)
    n_shells   = len(start_exp.exponents)

    header = [
        "step",
        "l_removed",
        "q_removed",
        "n_exponents",
        "time_sec",
        "energy_in",
        "energy_cull",
        "energy_opt",
        "total_x_change",
        "max_global_x_change",
    ]

    for l in range(n_shells):
        header.append(f"shell_{l}_rms_x")
        header.append(f"shell_{l}_max_x")

    csv_writer.writerow(header)
    csv_f.flush()

    with open(log_file, "a") as log_f:
        log_f.write(
            f"\n=== CMA CULLING START {datetime.now().isoformat(timespec='seconds')} ===\n\n"
        )
        log_f.flush()

        # ---- optional initial optimization ----
        if optimize_initial:
            opt0_dir = opti_collection / "opt_dir_initial"
            current_exp.save(init_culled_dir, "culled_000_initial.expo")

            line = "[Initial] Optimizing full exponent set before culling"
            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

            optimized_exp, last_energy, last_es = cma_fixed_exponent_count(
                current_exp,
                last_energy,
                objective,
                opt0_dir,
                generation_size =generation_size,
                sigma           =sigma,
                max_generations =max_generations,
                threads         =threads,
                overwrite       =overwrite_gens,
                logging         =logging > 1,
            )

            optimized_exp.save(best_culled_dir, "culled_000_optimized.expo")

            line = (
                f"[Initial] After optimization: Energy = {last_energy:.6e} | "
                f"ΔE vs original = {last_energy - start_energy:.6e} | "
                f"T {hms(time.time() - t0)}"
            )
            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

            new_log_name = run_logs_dir / f"run_{0}_log.txt"
            old_log_name = opt0_dir / "cma.log"
            shutil.copy(old_log_name, new_log_name)

            new_csv_name = run_csvs_dir / f"run_{0}_trace.csv"
            old_csv_name = opt0_dir / "cma_trace.csv"
            shutil.copy(old_csv_name, new_csv_name)

            total_rms, per_shell_rms, max_global, per_shell_max = exponent_difference_metrics(current_exp, optimized_exp)

            total_x_change      = float(exp(total_rms))
            max_global_x_change = float(exp(max_global))
            per_shell_rms_x     = exp(per_shell_rms)
            per_shell_max_x     = exp(per_shell_max)

            # ---- CSV row ----
            row = [
                0,
                -1,
                -1,
                sum(len(shell) for shell in current_exp.exponents),
                float(time.time() - t0),
                float(start_energy),
                float(start_energy),
                float(last_energy),
                total_x_change,
                max_global_x_change,
            ]

            for l in range(n_shells):
                row.append(float(per_shell_rms_x[l]))
                row.append(float(per_shell_max_x[l]))

            csv_writer.writerow(row)
            csv_f.flush()

            current_exp = optimized_exp

        # ---- iterative culling ----
        for i in range(exponents_to_cull):

            culling_dir = culling_collection / f"culling_{i}"

            cull_energy, cull_exp, idx, label = local_exponent_removal_analysis(
                current_exp,
                last_energy,
                objective,
                culling_dir,
                threads=threads,
                print_results=logging > 1,
            )

            # figure out what was removed (optional bookkeeping)
            num_exponents = sum(len(shell) for shell in cull_exp.exponents)

            opt_dir = opti_collection / (
                "opt_dir" if overwrite_gens else f"opt_dir_{i}"
            )

            if (i > 0 or optimize_initial):
                last_info = [
                    delete(last_es.mean, idx),
                    last_es.sigma,
                    update_covariance_after_removal(last_es.sm.C, label, current_exp.lengths, mode=propagation_mode),
                    delete(last_es.pc, idx)]
            else:
                last_info = None

            cull_exp.save(init_culled_dir, f"culled_{i+1:03d}_initial.expo")

            optimized_exp, optimized_energy, last_es = cma_fixed_exponent_count(
                cull_exp,
                cull_energy,
                objective,
                opt_dir,
                generation_size = generation_size,
                sigma           = sigma,
                max_generations = max_generations,
                threads         = threads,
                overwrite       = overwrite_gens,
                cma_state       = last_info if propagate_covariance else None,
                use_stopping    = use_stopping,
                logging         = logging > 1,
            )
            
            optimized_exp.save(best_culled_dir, f"culled_{i+1:03d}_optimized.expo")
            
            delta_cull       = cull_energy      - last_energy
            delta_opt_in     = optimized_energy - last_energy
            delta_total      = optimized_energy - start_energy
            energy_recovered = optimized_energy - cull_energy
            l_curr, q_curr   = label

            line = (
                f"[Culling {i+1:>2}/{exponents_to_cull:<2}] "
                f"(l={l_curr:>2}, q={q_curr:>2}) | "
                f"E_in = {last_energy:12.6f} | "
                f"E_cull = {cull_energy:12.6f} | "
                f"E_opt = {optimized_energy:12.6f} | "
                f"ΔE_cull = {delta_cull:+12.6f} | "
                f"ΔE_opt_in = {delta_opt_in:+12.6f} | "
                f"E_recov = {energy_recovered:+12.6f} | "
                f"ΔE_total = {delta_total:+12.6f} | "
                f"N = {num_exponents:3d} | "
                f"T {hms(time.time() - t0)}"
            )

            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

            total_rms, per_shell_rms, max_global, per_shell_max = exponent_difference_metrics(cull_exp, optimized_exp)

            total_x_change = float(exp(total_rms))
            max_global_x_change = float(exp(max_global))
            per_shell_rms_x = exp(per_shell_rms)
            per_shell_max_x = exp(per_shell_max)

            # ---- CSV row ----
            row = [
                i + 1,
                l_curr,
                q_curr,
                num_exponents,
                float(time.time() - t0),
                float(last_energy),
                float(cull_energy),
                float(optimized_energy),
                total_x_change,
                max_global_x_change,
            ]

            for l in range(n_shells):
                row.append(float(per_shell_rms_x[l]))
                row.append(float(per_shell_max_x[l]))

            csv_writer.writerow(row)
            csv_f.flush()

            new_log_name = run_logs_dir / f"run_{i+1}_log.txt"
            old_log_name = opt_dir / "cma.log"
            shutil.copy(old_log_name, new_log_name)

            new_csv_name = run_csvs_dir / f"run_{i+1}_trace.csv"
            old_csv_name = opt_dir / "cma_trace.csv"
            shutil.copy(old_csv_name, new_csv_name)

            current_exp = optimized_exp
            last_energy = optimized_energy

        log_f.write(
            f"\n=== CMA CULLING END {datetime.now().isoformat(timespec='seconds')} ===\n"
        )
        log_f.flush()

    return current_exp, last_energy
