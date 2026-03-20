from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from cma_opt import *
from opt_tools_new import *
from objectives import *

BASE_DIR = Path(__file__).resolve().parent

exp_path       = BASE_DIR / "FCa4_1.expo"
template_dir_g = BASE_DIR / "FCa4_template_1.inp"
submit_scr     = BASE_DIR / "run.sh"
work_dir       = BASE_DIR / "FCa4_ground_test_prop"

exp = Exponent_Set.from_file(exp_path)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging     = False,
    overwrite_existing  = False,
    custom_poll_interval= 1
    # full_logging=True 
   )


O              = Ground_Energy_Objective( template_dir_g, C)
start_energy_g = -2818.50859432

# cma_fixed_exponent_count( exp, start_energy_g, O, work_dir, 24, 0.1, 300, 12, logging=True, overwrite=True )



O_GCA            = Ground_Energy_Objective( template_dir_g, C)
cma_culling( exp, start_energy_g, O_GCA, work_dir, 15, generation_size=12, max_generations=300, threads=12, overwrite=True, logging=2, optimize_initial=True, propagate_covariance=True )