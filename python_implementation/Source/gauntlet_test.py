from exponent_handler import *
from gauntlet         import *


BASE_DIR = Path(__file__).resolve().parent

exp_path       = BASE_DIR / "Be_HF_1.expo"
template_dir_1 = BASE_DIR / "gaunt_mol1.inp"
template_dir_2 = BASE_DIR / "gaunt_mol2.inp"
template_dir_3 = BASE_DIR / "gaunt_mol3.inp"
submit_scr     = BASE_DIR / "run.sh"
work_dir       = "/home/dzemail/Desktop/Code_Projects/PHD_work/ML_EXP/python_implementation/Results/gauntlet_dir"

exp = Exponent_Set.from_file(exp_path)

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    manager_logging    = False,
    overwrite_existing = False 
   )


G = Gauntlet( [template_dir_1,template_dir_2,template_dir_3], C, work_dir )

G.show_molecules_and_methods()

G.run_gauntlet(exp, ["Be1", "Be2", "Be3"], ["rasscf1","rasscf2","rasscf3"])

