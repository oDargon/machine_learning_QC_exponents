from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
import copy





class Objective(ABC):
    def __init__(self, manager_cfg: Job_Manager_Config | None = None):
        
        self.default_manager_cfg: Job_Manager_Config = manager_cfg

    def evaluate_batch(self, exps: List[Exponent_Set], *, manager_cfg: Job_Manager_Config | None = None, work_dir: Path | str | None = None, threads: int = 1, names: List[str] | None = None, overwrite: bool = False, **kwargs) -> ndarray:

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

        return self.evaluate_batch_with_manager(exps, manager, threads, names=names, **kwargs)

    @abstractmethod
    def evaluate_batch_with_manager( self, exponents: List[Exponent_Set], manager: Job_Manager, threads: int = 1, names: List[str] | None = None, **kwargs) -> ndarray:
        ...


class Ground_Energy_Objective(Objective):
    def __init__(self, template_file: Path | str, manager_cfg: Job_Manager_Config | None = None):

        super().__init__(manager_cfg)
        self.template_file = template_file

    def evaluate_batch_with_manager(self, exponents: List[Exponent_Set], manager: Job_Manager, threads: int = 1, names: List[str] | None = None, **kwargs) -> ndarray:
        
        energies = []

        for i in range(len(exponents)):
            manager.add_job(exponents[i], self.template_file, name = names[i] if names is not None else None)

        manager.run_all_jobs(threads, 0.5)

        for job in manager.jobs:
            energies.append(job.exponent_set.energy or 1e6)

        manager.collect_successful_results()

        return array(energies, dtype=float64)


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

        manager.run_all_jobs(threads, 0.5)

        for i in range(len(exponents)):
            E_g = manager.jobs[3*i].exponent_set.energy   or 1e6
            E_c = manager.jobs[3*i+1].exponent_set.energy or 1e6
            E_a = manager.jobs[3*i+2].exponent_set.energy or 1e6

            energies.append( (E_g*self.ratios[0] + E_c*self.ratios[1] + E_a*self.ratios[2])/self.norm )

        manager.collect_successful_results()

        return array(energies, dtype=float64)