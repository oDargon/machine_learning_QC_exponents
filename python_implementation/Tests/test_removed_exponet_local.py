from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *

BASE_DIR = Path(__file__).resolve().parent

exp_path     = BASE_DIR / "Be_HF_1.expo"
template_dir = BASE_DIR / "Be_template_1.inp"
submit_scr   = BASE_DIR / "run.sh"

exp = Exponent_Set.from_file(exp_path)

removed_exps = []


for i in range(len(exp.exponents)):
    for j in range(exp.lengths[i]):
        exp_copy = exp.copy(no_energy=True)
        exp_copy.remove_exponent_uncontracted(i, j)
        removed_exps.append(exp_copy)


n_steps = 11
variation = 0.05  # ±5% variation

for i in range(len(removed_exps)):
    
    M = Job_Manager(
    ExecutorType.LOCAL_BASH,
    submit_scr,
    f"Results_test_removing_local_{i}",
    manager_logging=True,
    overwrite_existing=True 
   )
    
    for l in range(len(removed_exps[i].exponents)):
        for q in range(removed_exps[i].lengths[l]):
            for step in range(n_steps):
                factor = 1 - variation + (2 * variation / (n_steps - 1)) * step
                exp_copy = removed_exps[i].copy(no_energy=True)
                exp_copy.exponents[l][q] *= factor
                M.add_job(exp_copy, template_dir, name=f"Removed_exp_{l}_{q}_{step}")

    M.run_all_jobs(1, 0.1)            
    M.collect_successful_results()