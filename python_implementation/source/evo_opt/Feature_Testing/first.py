import sys
from pathlib import Path

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR))

from exponent_handler import *
from molcas_handler import *
from job_manager import *
from common import Executor_Type

exp_path     = BASE_DIR / "Be_HF_1.expo"
template_dir = BASE_DIR / "Be_template_1.inp"
submit_scr   = BASE_DIR / "run.sh"
extract_scr  = BASE_DIR / "extract.sh"
dest_dir     = BASE_DIR / "RUN1"

exp = Exponent_Set.from_file(exp_path)

print(BASE_DIR)

M = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=dest_dir,
    full_logging=True,
    overwrite_existing=True
)


# M.add_job(exp, template_dir)

for i in range(10):
    exp_copy = exp.copy(no_energy=True)
    exp_copy.exponents[0] *= (1 + 0.001 * i)
    M.add_job(exp_copy, template_dir)

M.run_all_jobs( 3 )