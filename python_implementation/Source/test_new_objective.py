from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from cma_opt import *
from opt_tools_new import *
from objectives import *

BASE_DIR = Path(__file__).resolve().parent

exp_path       = BASE_DIR / "Be_HF_1.expo"
template_dir_g = BASE_DIR / "Be_template_1.inp"
template_dir_c = BASE_DIR / "Be_template_1_cat.inp"
template_dir_a = BASE_DIR / "Be_template_1_an.inp"
submit_scr     = BASE_DIR / "run.sh"
work_dir       = BASE_DIR / "objective_test"

exp = Exponent_Set.from_file(exp_path)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging    = False,
    overwrite_existing = False 
   )

# O = Ground_Energy_Objective( template_dir_g, C)
# O_G            = Ground_Energy_Objective( template_dir_g, C)
# start_energy_g = -14.66075971

# local_exponent_removal_analysis( exp, start_energy_g, O_G, work_dir, print_results=True, threads=12, overwrite=True )

# O_GCA            = Ground_Energy_Objective_GCA( template_dir_g, template_dir_c, template_dir_a, C)
# start_energy_gca = (-14.66075971 + -14.31934168 + -14.64284503)/3

# local_exponent_removal_analysis( exp, start_energy_gca, O_GCA, work_dir, print_results=True, threads=12, overwrite=True )'




# O_G            = Ground_Energy_Objective( template_dir_g, C)
# start_energy_g = -14.66075971

# # local_exponent_removal_analysis( exp, start_energy_g, O_G, work_dir, print_results=True, threads=12, overwrite=True )

# cma_fixed_exponent_count( exp, start_energy_g, O_G, work_dir, 36, 0.1, 10, 12, overwrite=True, logging=True )



O_GCA            = Ground_Energy_Objective_GCA( template_dir_g, template_dir_c, template_dir_a, C)
start_energy_gca = (-14.66075971 + -14.31934168 + -14.64284503)/3

# local_exponent_removal_analysis( exp, start_energy_gca, O_GCA, work_dir, print_results=True, threads=12, overwrite=True )

# cma_fixed_exponent_count( exp, start_energy_gca, O_GCA, work_dir, 18, 0.1, 20, 12, overwrite=True, logging=True )

cma_culling( exp, start_energy_gca, O_GCA, work_dir, 20, generation_size=24, max_generations=100, threads=12, overwrite=True, logging=2, optimize_initial=True )

