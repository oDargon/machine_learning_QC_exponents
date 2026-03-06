from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from objectives import *
from numpy import exp, log
from opt_tools_new import *
import copy
import time
import cma

def hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def cma_fixed_exponent_count(start_exp: Exponent_Set, start_energy: float64, objective: Objective, work_dir: Path | str, generation_size: int = 30, sigma: float = 0.1, max_generations: int = 50, threads: int = 1,
                              *, overwrite: bool = False, logging: bool = False) -> tuple[Exponent_Set, float64, "cma.CMAEvolutionStrategy"]:

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

    x0 = log(start_exp.flatten_exps())
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': generation_size})
    t0 = time.time()

    best_energy_overall = start_energy
    best_exp_overall    = start_exp.copy_without_energy()

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

        if best_energy < best_energy_overall:
            best_energy_overall = best_energy
            best_exp_overall    = exp_objects[best_idx].copy_without_energy()

        best_exp_gen = exp_objects[best_idx].copy_without_energy()
        best_exp_gen.save(best_dir, f"gen_{gen:03d}")

        es.tell(population, energies)

        if logging:
            elapsed = time.time() - t0
            fevals  = es.countevals
            delta_e = best_energy - start_energy
            print(
                f"[Gen {gen:3d}] | "
                f"Fevals {fevals:6d} | "
                f"BestE {best_energy:14.8f} | "
                f"ΔE {delta_e: .8f} | "
                f"σ {es.sigma: .3e} | "
                f"T {hms(elapsed)}"
            )

        # if es.stop():
        #     break

    return best_exp_overall, best_energy_overall, es


def cma_culling(
    start_exp: Exponent_Set,
    start_energy: float64,
    objective: Objective,
    work_dir: Path | str,
    exponents_to_cull: int = 1,
    *,
    optimize_initial: bool = False,
    generation_size: int   = 30,
    sigma: float           = 0.1,
    max_generations: int   = 50,
    threads: int           = 1,
    overwrite: bool        = False,
    overwrite_gens: bool   = False,
    logging: int           = 0,
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
    log_file           = work_dir / "culling.log"

    culling_collection.mkdir(parents=True)
    opti_collection.mkdir(parents=True)
    best_culled_dir.mkdir(exist_ok=True)

    current_exp  = start_exp.copy_without_energy()
    last_energy  = start_energy

    with open(log_file, "a") as log_f:
        log_f.write(
            f"\n=== CMA CULLING START {datetime.now().isoformat(timespec='seconds')} ===\n"
        )
        log_f.flush()

        # ---- optional initial optimization ----
        if optimize_initial:
            opt0_dir = opti_collection / "opt_dir_initial"

            line = "[Initial] Optimizing full exponent set before culling"
            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

            current_exp, last_energy, es = cma_fixed_exponent_count(
                current_exp,
                last_energy,
                objective,
                opt0_dir,
                generation_size=generation_size,
                sigma=sigma,
                max_generations=max_generations,
                threads=threads,
                overwrite=overwrite_gens,
                logging=logging > 1,
            )

            current_exp.save(best_culled_dir, "culled_000_initial.expo")

            line = (
                f"[Initial] After optimization: Energy = {last_energy:.6e} | "
                f"ΔE vs original = {last_energy - start_energy:.6e}"
            )
            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

        # ---- iterative culling ----
        for i in range(exponents_to_cull):

            culling_dir = culling_collection / f"culling_{i}"

            cull_energy, cull_exp = local_exponent_removal_analysis(
                current_exp,
                last_energy,
                objective,
                culling_dir,
                threads=threads,
                print_results=logging > 1,
            )

            # figure out what was removed (optional bookkeeping)
            # assuming your removal function encodes this internally
            num_exponents = sum(len(shell) for shell in cull_exp.exponents)

            opt_dir = opti_collection / (
                "opt_dir" if overwrite_gens else f"opt_dir_{i}"
            )

            optimized_exp, optimized_energy, es = cma_fixed_exponent_count(
                cull_exp,
                cull_energy,
                objective,
                opt_dir,
                generation_size=generation_size,
                sigma=sigma,
                max_generations=max_generations,
                threads=threads,
                overwrite=overwrite_gens,
                logging=logging > 1,
            )

            optimized_exp.save(best_culled_dir, f"culled_{i+1:03d}.expo")

            delta_cull   = cull_energy - last_energy
            delta_opt_in = optimized_energy - last_energy
            delta_total  = optimized_energy - start_energy

            line = (
                f"[Culling {i+1:>2}/{exponents_to_cull:<2}] "
                f"E_in = {last_energy:12.6f} | "
                f"E_cull = {cull_energy:12.6f} | "
                f"E_opt = {optimized_energy:12.6f} | "
                f"ΔE_cull = {delta_cull:+12.6f} | "
                f"ΔE_opt_in = {delta_opt_in:+12.6f} | "
                f"ΔE_total = {delta_total:+12.6f} | "
                f"N = {num_exponents:3d}"
            )

            log_f.write(line + "\n")
            log_f.flush()
            if logging > 0:
                print(line)

            current_exp = optimized_exp
            last_energy = optimized_energy

    return current_exp, last_energy
