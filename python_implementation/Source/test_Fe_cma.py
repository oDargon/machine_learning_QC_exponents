from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from cma_opt import *
from opt_tools_new import *
from objectives import *

BASE_DIR = Path(__file__).resolve().parent

exp_path       = BASE_DIR / "Fe_1.expo"
template_dir_g = BASE_DIR / "Fe_template_1.inp"
submit_scr     = BASE_DIR / "run.sh"
work_dir       = BASE_DIR / "Fe_ground_test"

exp = Exponent_Set.from_file(exp_path)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging    = False,
    overwrite_existing = False,
    # full_logging=True 
   )


O              = Ground_Energy_Objective( template_dir_g, C)
start_energy_g = -1271.90809463

cma_fixed_exponent_count( exp, start_energy_g, O, work_dir, 24, 0.1, 200, 12, logging=True, overwrite=True )