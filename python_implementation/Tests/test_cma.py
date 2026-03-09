from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from cma_molcas import *
import cma

BASE_DIR = Path(__file__).resolve().parent

exp_path     = BASE_DIR / "Be_HF_1.expo"
template_dir = BASE_DIR / "Be_template_1.inp"
submit_scr   = BASE_DIR / "run.sh"
work_dir     = BASE_DIR / "cma_test"

exp = Exponent_Set.from_file(exp_path)

# exp.remove_exponent_uncontracted(0,10)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging    = False,
    overwrite_existing = False 
   )

# cma_fixed_exponent_count( exp, C, work_dir, template_dir, 30, 0.2, 100, overwrite=True, molcas_threads=6, logging=True )

# cma_culling( exp, C, work_dir, template_dir, 15, logging=2, molcas_threads=12, overwrite=True, max_generations= 10, generation_size=24, optimize_initial=True )
