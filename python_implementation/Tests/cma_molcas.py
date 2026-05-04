from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from exponent_opt_tools import local_exponent_removal_suggestion
from numpy import exp, log
import copy
import time
import cma

def hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def evaluate_population_objective(
    exponent_vectors: List[Sequence[float] | ndarray],
    exponent_shape: List[int],
    base_exp: Exponent_Set,
    manager: Job_Manager,
    template: str | Path,
    molcas_batch_size: Optional[int] = 1
) -> ndarray:
    energies = []

    for i in range(len(exponent_vectors)):
        new_exp = base_exp.copy(no_energy=True)

        idx = 0
        for l in range(len(exponent_shape)):
            n = exponent_shape[l]
            # assume exponent_vectors already contain log-exponents
            new_exp.exponents[l] = exp(exponent_vectors[i][idx:idx + n])
            idx += n

        manager.add_job(new_exp, template)

    manager.run_all_jobs(molcas_batch_size, 0.1)

    for job in manager.jobs:
        E = job.exponent_set.energy
        energies.append(E if E is not None else 1e6)

    manager.collect_successful_results()

    return array(energies, dtype=float) 


def cma_fixed_exponent_count(
    start_exp: Exponent_Set,
    config: Job_Manager_Config,
    work_dir: Path | str,
    template_path: str | Path,
    generation_size: int = 30,
    sigma: float         = 0.1,
    max_generations: int = 50,
    molcas_threads: int  = 1,
    *,
    overwrite: bool      = False,
    logging: bool        = False
) -> tuple[Exponent_Set, "cma.CMAEvolutionStrategy"]:
    
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

    # Validate config
    if config.custom_executor is None:
        if config.executor_type is None or config.execution_script is None:
            raise ValueError(
                "Cannot create Job_Manager: executor_type and execution_script "
                "must be provided if no custom_executor is given."
            )

    # Flatten start exponents into 1D log-vector
    x0             = []
    exponent_shape = []
    for shell_exp in start_exp.exponents:
        exponent_shape.append(len(shell_exp))
        x0.extend(log(shell_exp))
    x0 = array(x0)

    start_energy = start_exp.energy

    # Initialize CMA
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': generation_size})
    t0 = time.time()

    for gen in range(max_generations):
        batch_dir = work_dir / f"batch_{gen}"

        config.group_dir_path = batch_dir
        manager               = Job_Manager.from_config(config)

        #Get new population and run it
        population = es.ask()
        energies   = evaluate_population_objective(
            exponent_vectors = population,
            exponent_shape   = exponent_shape,
            base_exp         = start_exp,
            manager          = manager,
            template         = template_path,
            molcas_batch_size= molcas_threads
        )


        #Save best of each generation
        best_idx     = int(energies.argmin())
        best_energy  = energies[best_idx]
        best_vector  = population[best_idx]

        # reconstruct Exponent_Set
        best_exp_gen = start_exp.copy(no_energy=True)
        idx = 0
        for l, n in enumerate(exponent_shape):
            best_exp_gen.exponents[l] = exp(best_vector[idx:idx+n]) #exp since we work in log space
            idx += n
        best_exp_gen.assign_results(energy = best_energy)

        # write to disk
        best_exp_gen.save(best_dir, f"gen_{gen:03d}")

        #Update Cma with new info
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

    # Convert best vector back into Exponent_Set
    best_vector = es.result.xbest
    best_exp    = start_exp.copy(no_energy=True)

    idx = 0
    for l, n in enumerate(exponent_shape):
        best_exp.exponents[l] = exp(best_vector[idx:idx+n])
        idx += n

    best_exp.assign_results( energy = es.result.fbest )


    # print(es.mean, es.C, es.sigma)

    return best_exp, es


def cma_culling(
    start_exp: Exponent_Set,
    config: Job_Manager_Config,
    work_dir: Path | str,
    template_path: str | Path,
    exponents_to_cull: int = 1,
    *,
    optimize_initial:bool  = False,
    generation_size: int   = 30,
    sigma: float           = 0.1,
    max_generations: int   = 50,
    molcas_threads: int    = 1,
    overwrite: bool        = False,
    overwrite_gens:bool    = False,
    logging: int           = 0,
    propagate_info: int    = 0
):

    work_dir = Path(work_dir).resolve()

    # Guard for overall work_dir deletion
    if work_dir.exists() and overwrite:
        shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
    elif not work_dir.exists():
        work_dir.mkdir(parents=True)
    elif work_dir.exists() and not overwrite:
        raise FileExistsError(f"{work_dir} exists, set overwrite=True to replace it")

    culling_collection = work_dir / "culling_collection"
    opti_collection    = work_dir/ "opti_collection"
    best_culled_dir    = work_dir / "best_culled"
    log_file           = work_dir / "culling.log"

    culling_collection.mkdir(parents=True)
    opti_collection.mkdir(parents=True)
    best_culled_dir.mkdir(exist_ok=True)

    
    current_exp  = start_exp.copy(no_energy=True)
    start_energy = start_exp.energy
    last_energy  = start_energy

    current_exp.assign_results( energy = start_energy )
    removed_exponents_log = []

    with open(log_file, "a") as log_f:

        log_f.write(
            f"\n=== CMA CULLING START {datetime.now().isoformat(timespec='seconds')} ===\n"
        )
        log_f.flush()

        if optimize_initial:

            opt0_dir                   = opti_collection / "opt_dir_initial"
            init_config                = copy.deepcopy(config)
            init_config.group_dir_path = opt0_dir

            line = "[Initial] Optimizing full exponent set before culling"
            log_f.write(line + "\n")
            log_f.flush()

            if logging > 0:
                print(line)

            optimized_init_exp, es = cma_fixed_exponent_count(current_exp, init_config, opt0_dir,template_path, generation_size=generation_size, sigma=sigma,
                                                            max_generations=max_generations, molcas_threads=molcas_threads, overwrite=overwrite_gens, logging=True if logging > 1 else False)

            optimized_init_exp.save(best_culled_dir, "culled_000_initial.expo")
            delta_E_init = optimized_init_exp.energy - start_energy

            line = (
                f"[Initial] After optimization: Energy = {optimized_init_exp.energy:.6e} | "
                f"ΔE vs original = {delta_E_init:.6e}"
            )

            log_f.write(line + "\n")
            log_f.flush()

            if logging > 0:
                print(line)

            current_exp = optimized_init_exp
            last_energy = optimized_init_exp.energy

        for i in range( exponents_to_cull ):

            culling_dir                = culling_collection / f"culling_{i}"
            cull_config                = copy.deepcopy(config)
            cull_config.group_dir_path = culling_dir
            culling_manager            = Job_Manager.from_config(cull_config) 

            cull_exp, idx = local_exponent_removal_suggestion( current_exp, culling_manager, template_path, threads=molcas_threads )

            delta_E_step  = cull_exp.energy - last_energy
            delta_E_total = cull_exp.energy - start_energy
            removed_exponents_log.append((idx[0], idx[1], delta_E_step))

            num_exponents = sum(len(shell) for shell in cull_exp.exponents)

            opt_dir = opti_collection / "opt_dir"
            if not overwrite_gens:
                opt_dir = opti_collection / f"opt_dir_{i}"

            optimized_exp, es = cma_fixed_exponent_count( cull_exp, copy.deepcopy(config), opt_dir, template_path, generation_size=generation_size, sigma=sigma,
                                                    max_generations=max_generations, molcas_threads=molcas_threads, overwrite=overwrite_gens, logging = True if logging >1 else False )

            optimized_exp.save(best_culled_dir, f"culled_{i+1:03d}.expo")

            delta_E_post_step  = optimized_exp.energy - last_energy
            delta_E_post_total = optimized_exp.energy - start_energy

            line1 = (
                f"[Culling {i+1}/{exponents_to_cull}] "
                f"Removed (l={idx[0]}, q={idx[1]}) | "
                f"E_in   = {last_energy:.6e} | "
                f"E_cull = {cull_exp.energy:.6e} | "
                f"E_opt  = {optimized_exp.energy:.6e} | "
                f"N = {num_exponents}"
            )

            delta_cull   = cull_exp.energy - last_energy
            delta_opt_in = optimized_exp.energy - last_energy
            delta_total  = optimized_exp.energy - start_energy

            line = (
            f"[Culling {i+1:>2}/{exponents_to_cull:<2}] "
            f"Removed (l={idx[0]:>2}, q={idx[1]:>2}) | "
            f"E_in = {last_energy:12.6f} | "
            f"E_cull = {cull_exp.energy:12.6f} | "
            f"E_opt = {optimized_exp.energy:12.6f} | "
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
            last_energy = optimized_exp.energy




