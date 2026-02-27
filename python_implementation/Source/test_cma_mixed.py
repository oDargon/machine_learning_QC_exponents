from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from cma_molcas_cNa import *
from cma_molcas import *
import cma

BASE_DIR = Path(__file__).resolve().parent

exp_path      = BASE_DIR / "Be_HF_1.expo"
template_dir1 = BASE_DIR / "Be_template_1.inp"
template_dir2 = BASE_DIR / "Be_template_1_an.inp"
template_dir3 = BASE_DIR / "Be_template_1_cat.inp"
submit_scr    = BASE_DIR / "run.sh"
work_dir      = BASE_DIR / "cma_test_energy"
work_dir2     = BASE_DIR / "cma_test_energy3"


exp = Exponent_Set.from_file(exp_path)

# exp.remove_exponent_uncontracted(0,10)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging    = False,
    overwrite_existing = False 
   )

# cma_fixed_exponent_count( exp, C, work_dir, template_dir1, 48, 0.2, 100, overwrite=True, molcas_threads=6, logging=True )

# cma_culling(exp, C, work_dir, template_dir1, 20, logging=2, molcas_threads=12, overwrite=True, max_generations= 150, generation_size=60, optimize_initial=True, overwrite_gens=True )

cma_culling_mixed( exp, C, work_dir2, [template_dir1,template_dir2,template_dir3], 20, logging=2, molcas_threads=12, overwrite=True, max_generations= 100, generation_size=60, optimize_initial=True, overwrite_gens=True )
