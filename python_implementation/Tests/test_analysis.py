from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from exponent_opt_tools import *

BASE_DIR = Path(__file__).resolve().parent

exp_path     = BASE_DIR / "Be_HF_1.expo"
template_dir = BASE_DIR / "Be_template_1.inp"
submit_scr   = BASE_DIR / "run.sh"

exp = Exponent_Set.from_file(exp_path)

M = Job_Manager(
    ExecutorType.LOCAL_BASH,
    submit_scr,
    "Results_test_removing",
    manager_logging=True,
    overwrite_existing=True
)

M1 = M.copy_without_jobs("Results_Local_stability")

local_exponent_removal_analysis(exp,M, template_dir, print_results=True)
local_exponent_sensitivity_analysis(exp,M1, template_dir, x=0.05, print_results=True)