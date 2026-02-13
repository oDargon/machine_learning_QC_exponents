from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *

# Get directory where THIS script lives
BASE_DIR = Path(__file__).resolve().parent

exp_path     = BASE_DIR.parent / "Resources" / "Be_HF_1.expo"
template_dir = BASE_DIR.parent / "Templates" / "Be_template_1.inp"
submit_scr   = BASE_DIR / "run.sh"
dest_dir     = BASE_DIR / "RUN1"

exp = Exponent_Set.from_file(exp_path)

M = Job_Manager(
    ExecutorType.LOCAL_BASH,
    submit_scr,
    dest_dir,
    full_logging=True
)


# M.add_job(exp, template_dir)

for i in range(10):
    exp_copy = exp.copy()
    exp_copy.exponents[0] *= (1 + 0.001 * i)
    M.add_job(exp_copy, template_dir)

M.run_all_jobs( 3 )

