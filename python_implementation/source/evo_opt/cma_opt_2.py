from pathlib import Path
from .exponent_handler import Exponent_Set
from .objectives import Objective
from .opt_tools_new import local_exponent_removal_analysis
from .cma_logging import FixedCountLogger
from dataclasses import dataclass
from numpy import exp, log, delete, float64, ndarray, eye, array
from numpy.linalg import eigvalsh
import shutil
import time
import pickle
import cma


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


def evaluate_initial(
    exp: Exponent_Set,
    objective: Objective,
    work_dir: Path | str,
    *,
    threads: int = 1,
    subdir_name: str = "initial_eval",
    contract_frozen_shells: bool = False,
) -> Exponent_Set:
    eval_dir = Path(work_dir).resolve() / subdir_name
    eval_exp = exp.copy(no_energy=True)
    if not contract_frozen_shells:
        eval_exp.uncontract_all()
    results  = objective.evaluate_batch([eval_exp], work_dir=eval_dir, threads=threads)
    return results[0]




def cma_fixed_exponent_count(
    start_exp: Exponent_Set,
    start_energy: float64,
    objective: Objective,
    work_dir: Path | str,
    generation_size: int = 30,
    sigma: float         = 0.01,
    max_generations: int = 50,
    threads: int         = 1,
    active_shell: int    = 0,
    *,
    overwrite: bool              = False,
    cma_state: list | None       = None,
    logging: bool                = False,
    use_stopping: bool           = False,
    contract_frozen_shells: bool = False,
    init_state_path: Path | None = None,
    memory_dir: Path | None      = None,
    update_cadence: int          = 10,
    mean_override: ndarray | None = None,
) -> tuple[Exponent_Set, float64, "cma.CMAEvolutionStrategy"]:

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

    n_shells = len(start_exp.exponents)
    if active_shell < 0:
        raise ValueError(f"Active shell cant be negative, active shell chosen: {active_shell}")
    elif active_shell > n_shells - 1:
        raise ValueError(f"Active shell out of bounds, active shell chosen {active_shell} out of max {n_shells - 1}")

    if memory_dir is not None:
        memory_dir = Path(memory_dir)

    x0 = log(start_exp.exponents[active_shell])
    es = cma.CMAEvolutionStrategy(x0, sigma, {'popsize': generation_size})

    if init_state_path is not None and Path(init_state_path).exists():
        with open(init_state_path, 'rb') as f:
            es = pickle.load(f)

    if mean_override is not None:
        # Recenter the search (e.g. an Anderson-extrapolated guess) without
        # touching the propagated covariance/sigma - x0 above only takes
        # effect when there's no prior state to unpickle, so this is the only
        # way to inject a different starting point once a run is warm-started.
        es.mean = log(array(mean_override, dtype=float64))

    t0 = time.time()

    if cma_state is not None:
        es.mean = cma_state[0]

        Cnew = cma_state[2]
        Cnew = 0.5 * (Cnew + Cnew.T)
        evals = eigvalsh(Cnew)
        print("min eig before inject:", evals.min(), "max eig:", evals.max())
        es.sm.C = Cnew
        es.sm.update_now()

    logger               = FixedCountLogger(work_dir, n_shells, active_shell, start_exp, start_energy, print_to_stdout=logging)
    best_energy_overall  = start_energy
    best_exp_overall     = start_exp.copy(no_energy=True)
    recent_best_energies = []

    for gen in range(max_generations):
        batch_dir = work_dir / f"batch_{gen}"

        population  = es.ask()
        exp_objects = []
        for vec in population:
            new_exp                         = start_exp.copy(no_energy=True)
            new_exp.exponents[active_shell] = array(exp(vec), dtype=float64)
            if contract_frozen_shells:
                new_exp.uncontract_shell(active_shell)
            else:
                new_exp.uncontract_all()
            exp_objects.append(new_exp)

        results     = objective.evaluate_batch(exp_objects, work_dir=batch_dir, threads=threads)
        energies    = array([r.energy for r in results], dtype=float64)
        best_idx    = int(energies.argmin())
        best_energy = energies[best_idx]
        recent_best_energies.append(float(best_energy))
        if len(recent_best_energies) > 5:
            recent_best_energies.pop(0)

        if best_energy < best_energy_overall:
            best_energy_overall = best_energy
            best_exp_overall    = results[best_idx].copy(no_energy=True)

        best_exp_gen = results[best_idx].copy(no_energy=True)
        best_exp_gen.save(best_dir, f"gen_{gen:03d}")

        es.tell(population, energies)

        logger.log_generation(gen, es.countevals, time.time() - t0, float(best_energy), float(best_energy_overall), es, best_exp_gen)

        if memory_dir is not None and (gen + 1) % update_cadence == 0:
            with open(memory_dir / "cma_state.pkl", 'wb') as f:
                pickle.dump(es, f)
            _mem               = best_exp_overall.copy(no_energy=True)
            _mem.energy        = float(best_energy_overall)
            _mem.save(memory_dir, "current", overwrite=True)

        stop_reason = None
        if es.sigma < 1e-4:
            stop_reason = "sigma < 1e-4"
        elif es.sigma < 1e-3 and len(recent_best_energies) == 5:
            rounded = [round(e, 5) for e in recent_best_energies]
            if len(set(rounded)) == 1:
                stop_reason = "sigma < 1e-3 and last 5 best energies equal to 5 decimals"

        if stop_reason is not None and use_stopping:
            logger.log_stop(stop_reason, gen, float(best_energy), es)
            break

    logger.close()

    if memory_dir is not None:
        with open(memory_dir / "cma_state.pkl", 'wb') as f:
            pickle.dump(es, f)
        mem_exp        = best_exp_overall.copy(no_energy=True)
        mem_exp.energy = float(best_energy_overall)
        mem_exp.save(memory_dir, "current", overwrite=True)

    return best_exp_overall, best_energy_overall, es


