from pathlib import Path
from .exponent_handler import Exponent_Set
from .job_manager import Job_Manager, Job_Manager_Config
from numpy import ndarray, array, float64
from typing import Optional, List, TypeAlias
import copy
from abc import ABC, abstractmethod


class Objective(ABC):
    def __init__(self, manager_cfg: Job_Manager_Config | None = None):
        
        self.default_manager_cfg: Job_Manager_Config = manager_cfg

    def evaluate_batch(
        self,
        exps: List[Exponent_Set],
        *,
        manager_cfg: Job_Manager_Config | None = None,
        work_dir: Path | str | None            = None,
        threads: int                           = 1,
        rasorbs: None | Path | str | List[Path | str] | List[List[Path | str]] = None,
        names: List[str] | None                = None,
        overwrite: bool                        = False,
        **kwargs
        ) -> List[Exponent_Set]:

        cfg = copy.deepcopy(manager_cfg or self.default_manager_cfg)
        if cfg is None:
            raise ValueError("No Job_Manager_Config provided")

        if work_dir is not None:
            cfg.group_dir_path = work_dir

        if overwrite is True:
            cfg.overwrite_existing = True

        manager = Job_Manager.from_config(cfg)

        if names is not None and len(exps) != len(names):
            raise ValueError(f"If names array is passed, it must be the same length as the number of exponents passed. You passed, {len(exps)} exponents and {len(names)} names")

        return self.evaluate_batch_with_manager(exps, manager, threads, rasorbs=rasorbs, names=names, **kwargs)

    @abstractmethod
    def evaluate_batch_with_manager(
        self,
        exponents:
        List[Exponent_Set],
        manager: Job_Manager,
        threads: int = 1,
        rasorbs: None | Path | str | List[Path | str] | List[List[Path | str]] = None,
        names: List[str] | None = None,
        **kwargs
        ) -> List[Exponent_Set]:
        ...

    def _validate_batch_rasorbs(
        self,
        rasorbs: None | Path | str | list[Path | str] | list[list[Path | str]],
    ) -> str:
        if rasorbs is None:
            return "none"

        if isinstance(rasorbs, (str, Path)):
            return "single"

        if not isinstance(rasorbs, list):
            raise ValueError(
                "rasorbs must be None, a str, a Path, a list of str/Path, "
                "or a list of lists of str/Path."
            )

        if len(rasorbs) == 0:
            return "empty"

        if all(isinstance(item, (str, Path)) for item in rasorbs):
            return "flat"

        if all(isinstance(item, list) for item in rasorbs):
            expected_len = len(rasorbs[0])

            for i, sublist in enumerate(rasorbs):
                if len(sublist) != expected_len:
                    raise ValueError(
                        "If rasorbs is a list of lists, all sublists must have the same length. "
                        f"Sublist 0 has length {expected_len}, but sublist {i} has length {len(sublist)}."
                    )

                for j, item in enumerate(sublist):
                    if not isinstance(item, (str, Path)):
                        raise ValueError(
                            "If rasorbs is a list of lists, each element must be a str or Path. "
                            f"Invalid element at rasorbs[{i}][{j}]: {item!r}"
                        )

            return "nested"

        raise ValueError(
            "Invalid rasorbs structure. A top-level list must be either:\n"
            "- a flat list of only str/Path, or\n"
            "- a list of lists, where each inner list contains only str/Path."
        )


class Ground_Energy_Objective(Objective):
    def __init__(
        self,
        template_file: Path | str, manager_cfg: Job_Manager_Config | None = None
        ):

        super().__init__(manager_cfg)
        self.template_file = template_file

    def evaluate_batch_with_manager(
        self,
        exponents: List[Exponent_Set],
        manager: Job_Manager,
        threads: int = 1,
        rasorbs: None | Path | str | List[Path | str] | List[List[Path | str]] = None,
        names: List[str] | None = None,
        **kwargs
        ) -> List[Exponent_Set]:

        rasorb_mode = self._validate_batch_rasorbs(rasorbs)

        if rasorb_mode == "nested" and len(rasorbs) != len(exponents):
            raise ValueError(
                "If rasorbs is a list of lists, the outer list must have the same "
                f"length as exponents. Got {len(rasorbs)} rasorb entries for "
                f"{len(exponents)} exponents."
            )

        for i, exp in enumerate(exponents):
            if rasorb_mode in ("none", "empty"):
                job_rasorbs = None
            elif rasorb_mode == "single":
                job_rasorbs = rasorbs
            elif rasorb_mode == "flat":
                job_rasorbs = rasorbs
            elif rasorb_mode == "nested":
                job_rasorbs = rasorbs[i]
            else:
                raise RuntimeError(f"Unexpected rasorb mode: {rasorb_mode}")

            manager.add_job(exp, self.template_file, name=names[i] if names is not None else None,rasorbs=job_rasorbs)

        manager.run_all_jobs(threads)

        manager.collect_successful_results()

        for job in manager.jobs:
            if job.exponent_set.energy is None:
                job.exponent_set.energy = 1e6

        return [job.exponent_set for job in manager.jobs]

#DEPRECATED################################################################################################################################################
class Ground_Energy_Objective_GCA(Objective):
    def __init__(self, template_file_ground: Path | str, template_file_cation: Path | str, template_file_anion: Path | str, manager_cfg: Job_Manager_Config | None = None, *, ratio: List[float] = None):

        super().__init__(manager_cfg)
        self.template_file_g = template_file_ground
        self.template_file_c = template_file_cation
        self.template_file_a = template_file_anion
        self.ratios          = ratio if ratio is not None else [1,1,1]
        self.norm            = sum(self.ratios)

        if len(self.ratios) != 3:
            raise ValueError("ratios must have length 3: [ground, cation, anion]")

    def evaluate_batch_with_manager(self, exponents: List[Exponent_Set], manager: Job_Manager, threads: int = 1, names: List[str] | None = None, **kwargs) -> ndarray:
        
        energies = [] #In reality, this is the average energy of the three energy calcualtions

        for i in range(len(exponents)):
            manager.add_job(exponents[i], self.template_file_g, name = names[i] + "_ground" if names is not None else None)
            manager.add_job(exponents[i], self.template_file_c, name = names[i] + "_cation" if names is not None else None)
            manager.add_job(exponents[i], self.template_file_a, name = names[i] + "_anion"  if names is not None else None)

        manager.run_all_jobs(threads)

        for i in range(len(exponents)):
            E_g = manager.jobs[3*i].exponent_set.energy   or 1e6
            E_c = manager.jobs[3*i+1].exponent_set.energy or 1e6
            E_a = manager.jobs[3*i+2].exponent_set.energy or 1e6

            energies.append( (E_g*self.ratios[0] + E_c*self.ratios[1] + E_a*self.ratios[2])/self.norm )

        manager.collect_successful_results()

        return array(energies, dtype=float64)