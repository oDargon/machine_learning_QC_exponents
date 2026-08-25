from pathlib import Path
from threading import Thread, Lock, Event
from numpy import exp, log, float64, ndarray, array
from numpy.linalg import eigvalsh
from numpy.random import normal as _normal
from .exponent_handler import Exponent_Set
from .objectives import Objective
from .cma_logging import FixedCountLogger
from .tempering import from_registry, Tempering_Codec
import shutil
import time
import cma


def _last5_converged(recent_best_energies: list, tol: float = 1e-6) -> bool:
    """Simple early-stop test: the last 5 generation-best energies all agree to
    within `tol`. Only fires once 5 have accumulated."""
    return len(recent_best_energies) == 5 and (max(recent_best_energies) - min(recent_best_energies)) <= tol


class Shell_Optimization:

    def __init__(
        self,
        start_exp: Exponent_Set,
        start_energy: float64,
        objective: Objective,
        work_dir: Path | str,
        generation_size: int = 30,
        sigma: float         = 0.01,
        max_generations: int = 50,
        active_shell: int    = 0,
        *,
        overwrite: bool               = False,
        cma_state: list | None        = None,
        logging: bool                 = False,
        use_stopping: bool            = False,
        contract_frozen_shells: bool  = False,
        use_tempering: bool           = False,
        n_tempering_params: int       = 6,
        seed: int | None              = None,
    ) -> None:
        self._start_exp              = start_exp
        self._start_energy           = start_energy
        self._objective              = objective
        self._work_dir               = Path(work_dir)
        self._generation_size        = generation_size
        self._sigma                  = sigma
        self._max_generations        = max_generations
        self._active_shell           = active_shell
        self._overwrite              = overwrite
        self._cma_state              = cma_state
        self._logging                = logging
        self._use_stopping           = use_stopping
        self._contract_frozen_shells = contract_frozen_shells
        self._use_tempering          = use_tempering
        self._n_tempering_params     = n_tempering_params
        self._seed                   = seed

        self._lock        = Lock()
        self._stop_event  = Event()
        self._pause_event = Event()
        self._pause_event.set()

        self._best_exp       = None
        self._best_energy    = None
        self._generation     = -1
        self._sigma_snapshot = None
        self._mean_snapshot  = None
        self._history        = []   # per-generation trajectory records (see history property)

        self._pending_root_exp = None

        self._thread    = None
        self._exception = None

    @property
    def best_exp(self) -> Exponent_Set | None:
        with self._lock:
            return self._best_exp

    @property
    def best_energy(self) -> float64 | None:
        with self._lock:
            return self._best_energy

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def sigma(self) -> float | None:
        with self._lock:
            return self._sigma_snapshot

    @property
    def mean(self) -> ndarray | None:
        with self._lock:
            return self._mean_snapshot

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set() and self.is_running

    @property
    def exception(self) -> Exception | None:
        """The exception that killed the worker thread, or None. Only ever set on
        a genuine crash — a clean stop() or normal completion leaves it None."""
        return self._exception

    @property
    def history(self) -> list:
        """Per-generation trajectory, one dict per generation:
        {gen, mean, sigma, cov, best_energy, best_energy_overall}. `mean` is the
        distribution centre (param space when tempering, else log-exponents), `cov`
        is the search covariance sigma**2 * C (None in the 1-D path)."""
        with self._lock:
            return list(self._history)

    def start(self, threads: int = 1) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Shell_Optimization is already running")
        self._thread = Thread(target=self._worker, args=(threads,), daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self, wait: bool = True) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if wait and self._thread is not None:
            self._thread.join()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def get_state(self) -> dict:
        with self._lock:
            mean = self._mean_snapshot.copy() if self._mean_snapshot is not None else None
            return {
                "generation":  self._generation,
                "best_energy": self._best_energy,
                "sigma":       self._sigma_snapshot,
                "mean":        mean,
                "best_exp":    self._best_exp,
            }

    def update_root_exponent(self, new_exp: Exponent_Set) -> None:
        with self._lock:
            self._pending_root_exp = new_exp

    def _worker(self, threads: int) -> None:
        logger = None
        try:
            work_dir = self._work_dir.resolve()
            if work_dir.exists() and self._overwrite:
                shutil.rmtree(work_dir)
                work_dir.mkdir(parents=True)
            elif not work_dir.exists():
                work_dir.mkdir(parents=True)
            elif work_dir.exists() and not self._overwrite:
                raise FileExistsError(f"{work_dir} exists, set overwrite=True to replace it")

            best_dir = work_dir / "best_per_generation"
            best_dir.mkdir(exist_ok=True)

            n_shells = len(self._start_exp.exponents)
            if self._active_shell < 0:
                raise ValueError(f"Active shell cant be negative, active shell chosen: {self._active_shell}")
            elif self._active_shell > n_shells - 1:
                raise ValueError(f"Active shell out of bounds, active shell chosen {self._active_shell} out of max {n_shells - 1}")

            n_active = len(self._start_exp.exponents[self._active_shell])
            codec: Tempering_Codec | None = (
                from_registry("polynomial", m=self._n_tempering_params, n=n_active)
                if self._use_tempering else None
            )

            x0 = codec.encode(self._start_exp.exponents[self._active_shell]) if codec else log(self._start_exp.exponents[self._active_shell])

            if len(x0) == 1:
                # ── 1-D: (1+λ)-ES generation loop ────────────────────────────────
                mean_1d              = float(x0[0])
                sigma_1d             = self._sigma
                prev_best            = self._start_energy
                best_energy_overall  = self._start_energy
                best_exp_overall     = self._start_exp.copy(no_energy=True)
                recent_best_energies = []
                root_exp             = self._start_exp.copy(no_energy=True)

                for gen in range(self._max_generations):
                    batch_dir = work_dir / f"batch_{gen}"

                    if self._stop_event.is_set():
                        break
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        break

                    with self._lock:
                        pending_root           = self._pending_root_exp
                        self._pending_root_exp = None
                    if pending_root is not None:
                        root_exp = pending_root.copy(no_energy=True)

                    candidates  = [mean_1d + sigma_1d * _normal() for _ in range(self._generation_size)]
                    exp_objects = []
                    for i in range(len(candidates)):
                        new_exp = root_exp.copy(no_energy=True)
                        if codec:
                            new_exp.apply_params(self._active_shell, codec, array([candidates[i]]), n=n_active)
                        else:
                            new_exp.set_shell_exponents(self._active_shell, exp(array([candidates[i]])))
                        if not self._contract_frozen_shells:
                            new_exp.uncontract_all()
                        exp_objects.append(new_exp)

                    results  = self._objective.evaluate_batch(exp_objects, work_dir=batch_dir, threads=threads)
                    energies = array([r.energy for r in results], dtype=float64)
                    best_idx = int(energies.argmin())
                    best_e   = energies[best_idx]

                    # (1+λ): keep the incumbent mean unless an offspring actually beats it
                    if best_e < prev_best:
                        mean_1d   = candidates[best_idx]
                        prev_best = best_e
                        sigma_1d *= 1.52
                    else:
                        sigma_1d *= 0.9
                    sigma_1d = max(1e-10, sigma_1d)

                    if best_e < best_energy_overall:
                        best_energy_overall = best_e
                        best_exp_overall    = results[best_idx].copy(no_energy=True)

                    results[best_idx].copy(no_energy=True).save(best_dir, f"gen_{gen:03d}")

                    recent_best_energies.append(float(best_e))
                    if len(recent_best_energies) > 5:
                        recent_best_energies.pop(0)

                    with self._lock:
                        self._best_exp       = best_exp_overall
                        self._best_energy    = best_energy_overall
                        self._generation     = gen
                        self._sigma_snapshot = sigma_1d
                        self._mean_snapshot  = array([mean_1d])
                        self._history.append({
                            "gen":                 gen,
                            "mean":                array([mean_1d]),
                            "sigma":               float(sigma_1d),
                            "cov":                 None,
                            "best_energy":         float(best_e),
                            "best_energy_overall": float(best_energy_overall),
                        })

                    if self._use_stopping and _last5_converged(recent_best_energies):
                        break

                return

            # ── N-D: CMA-ES ──────────────────────────────────────────────────────
            cma_opts = {'popsize': self._generation_size}
            if self._seed is not None:
                cma_opts['seed'] = self._seed
            es = cma.CMAEvolutionStrategy(x0, self._sigma, cma_opts)

            if self._cma_state is not None:
                es.mean  = self._cma_state[0]
                Cnew     = self._cma_state[2]
                Cnew     = 0.5 * (Cnew + Cnew.T)
                evals    = eigvalsh(Cnew)
                print("min eig before inject:", evals.min(), "max eig:", evals.max())
                es.sm.C  = Cnew
                es.sm.update_now()

            t0 = time.time()

            logger               = FixedCountLogger(work_dir, n_shells, self._active_shell, self._start_exp, self._start_energy, print_to_stdout=self._logging, codec=codec)
            best_energy_overall  = self._start_energy
            best_exp_overall     = self._start_exp.copy(no_energy=True)
            recent_best_energies = []
            root_exp             = self._start_exp.copy(no_energy=True)

            for gen in range(self._max_generations):
                batch_dir = work_dir / f"batch_{gen}"

                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                with self._lock:
                    pending_root           = self._pending_root_exp
                    self._pending_root_exp = None
                if pending_root is not None:
                    root_exp = pending_root.copy(no_energy=True)

                population  = es.ask()
                exp_objects = []
                for i in range(len(population)):
                    vec     = population[i]
                    new_exp = root_exp.copy(no_energy=True)
                    if codec:
                        new_exp.apply_params(self._active_shell, codec, vec, n=n_active)
                    else:
                        new_exp.set_shell_exponents(self._active_shell, exp(vec))
                    # apply_params / set_shell_exponents already uncontract the active
                    # shell; when frozen shells stay contracted that's all we want,
                    # otherwise uncontract everything.
                    if not self._contract_frozen_shells:
                        new_exp.uncontract_all()
                    exp_objects.append(new_exp)

                results     = self._objective.evaluate_batch(exp_objects, work_dir=batch_dir, threads=threads)
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

                try:
                    cov = (float(es.sigma) ** 2) * array(es.sm.C, dtype=float64)   # search covariance
                except Exception:
                    cov = None

                with self._lock:
                    self._best_exp       = best_exp_overall
                    self._best_energy    = best_energy_overall
                    self._generation     = gen
                    self._sigma_snapshot = es.sigma
                    self._mean_snapshot  = es.mean.copy()
                    self._history.append({
                        "gen":                 gen,
                        "mean":                es.mean.copy(),
                        "sigma":               float(es.sigma),
                        "cov":                 cov,
                        "best_energy":         float(best_energy),
                        "best_energy_overall": float(best_energy_overall),
                    })

                if self._use_stopping and _last5_converged(recent_best_energies):
                    logger.log_stop("last 5 best energies within 1e-6", gen, float(best_energy), es)
                    break

        except Exception as exc:
            self._exception = exc
            raise
        finally:
            if logger is not None:
                logger.close()
