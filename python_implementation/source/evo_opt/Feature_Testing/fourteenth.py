import sys
import yaml
import shutil
import subprocess
from pathlib import Path

WORK_DIR   = Path.cwd() / "Optimization"
SUBMIT_DIR = Path.cwd()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager_Config
from evo_opt.common import Executor_Type
from evo_opt.cma_opt_2 import evaluate_initial

exp_path    = SUBMIT_DIR / "Si.expo"
template    = SUBMIT_DIR / "template.inp"
submit_scr  = SUBMIT_DIR / "run.sh"
extract_scr = SUBMIT_DIR / "extract.sh"

ACTIVE_SHELL    = 2
GENERATION_SIZE = 12
THREADS         = 12
SIGMA           = 0.01
MAX_GENERATIONS = 100

WORK_DIR.mkdir(parents=True, exist_ok=True)

exp = Exponent_Set.from_file(exp_path)

cfg = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = WORK_DIR,
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)
objective = Ground_Energy_Objective(template, cfg)

start_exp    = evaluate_initial(exp, objective, WORK_DIR / "initial_eval", threads=THREADS)
start_energy = start_exp.energy

SHELL_CYCLE_DIR = WORK_DIR / "shell_0_cycle_0"
INIT_DIR        = SHELL_CYCLE_DIR / "init"
MEM_DIR         = SHELL_CYCLE_DIR / "memory"

INIT_DIR.mkdir(parents=True, exist_ok=True)
MEM_DIR.mkdir(parents=True, exist_ok=True)

shutil.copy(template,    INIT_DIR / "template.inp")
shutil.copy(submit_scr,  INIT_DIR / "run.sh")
shutil.copy(extract_scr, INIT_DIR / "extract.sh")
start_exp.save(INIT_DIR, "current", overwrite=True)

spec = {
    "active_shell":    ACTIVE_SHELL,
    "generation_size": GENERATION_SIZE,
    "threads":         THREADS,
    "sigma":           SIGMA,
    "max_generations": MAX_GENERATIONS,
    "start_energy":    float(start_energy),
}

CONFIG_PATH = INIT_DIR / "config.yaml"
with open(CONFIG_PATH, "w") as f:
    yaml.safe_dump(spec, f)

subprocess.run(
    ["cmafex", str(CONFIG_PATH), str(INIT_DIR), str(MEM_DIR)],
    cwd=SHELL_CYCLE_DIR,
    check=True,
)
