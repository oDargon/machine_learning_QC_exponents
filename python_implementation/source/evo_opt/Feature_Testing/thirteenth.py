import sys
import csv
import yaml
import copy
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
cfg_yaml    = SUBMIT_DIR / "run.yaml"

CORES_PER_SHELL    = [12, 11, 7, 5, 1]
SIGMA              = 0.01
MAX_GENERATIONS    = 10
CYCLES             = 5
MU                 = 0.95
SHELLS_TO_OPTIMIZE = [0, 1, 2, 3]
WARMSTART_CMA      = True

MEMORY_FILE = WORK_DIR / "current_best.expo"
L_LABELS    = ["s", "p", "d", "f", "g", "h"]

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

start_exp    = evaluate_initial(exp, objective, WORK_DIR, threads=1)
start_energy = start_exp.energy

current_exp    = start_exp
current_energy = start_energy

opti_dirs = []
for i in range(len(exp.exponents)):
    opti_dirs.append(Path(WORK_DIR / f"Shell_{i}").resolve())
    opti_dirs[i].mkdir(parents=True)

with open(cfg_yaml) as f:
    base_spec = yaml.safe_load(f)

WORK_DIR.mkdir(parents=True, exist_ok=True)

SEP      = "=" * 72
LOG_FILE = WORK_DIR / "cyclic_log.txt"
CSV_FILE = WORK_DIR / "cyclic_log.csv"

print(f"Log  : {LOG_FILE}")
print(f"CSV  : {CSV_FILE}")

CSV_HEADER = [
    "cycle", "shell", "shell_label", "n_exponents", "popsize", "max_generations",
    "E_before", "E_after", "dE", "dE_total", "exp_change_pct",
]

with open(LOG_FILE, "w") as log, open(CSV_FILE, "w", newline="") as csv_f:
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(CSV_HEADER)
    csv_f.flush()
    for i in range(CYCLES):
        for j in SHELLS_TO_OPTIMIZE:
            lbl       = L_LABELS[j] if j < len(L_LABELS) else str(j)
            n_exp     = len(current_exp.exponents[j])
            gen_size  = CORES_PER_SHELL[j]

            print(f"{SEP}")
            print(f"  Cycle {i + 1}/{CYCLES}  |  Shell {j} ({lbl})  |  {n_exp} exponents  |  popsize={gen_size}  |  sigma={SIGMA}")
            print(f"{SEP}")

            CYCLE_DIR = Path(opti_dirs[j] / f"Cycle_{i}").resolve()
            CYCLE_DIR.mkdir(parents=True)

            shutil.copy(template,    CYCLE_DIR / "template.inp")
            shutil.copy(submit_scr,  CYCLE_DIR / "run.sh")
            shutil.copy(extract_scr, CYCLE_DIR / "extract.sh")
            current_exp.save(CYCLE_DIR, "current", overwrite=True)

            spec                         = copy.deepcopy(base_spec)
            spec["exp_path"]             = "current.expo"
            spec["template_path"]        = "template.inp"
            spec["run_script_path"]      = "run.sh"
            spec["extract_script_path"]  = "extract.sh"
            spec["active_shell"]         = j
            spec["generation_size"]      = CORES_PER_SHELL[j]
            spec["threads"]              = CORES_PER_SHELL[j]
            spec["sigma"]                = SIGMA
            spec["max_generations"]      = MAX_GENERATIONS
            spec["start_energy"]         = float(current_energy)
            spec["work_dir"]             = "."
            spec["memory_path"]          = str(MEMORY_FILE)
            if WARMSTART_CMA:
                spec["cma_state_path"]   = str(opti_dirs[j] / "current_cma_state.pkl")

            with open(CYCLE_DIR / "run.yaml", "w") as f:
                yaml.safe_dump(spec, f)

            pre_energy = current_energy
            pre_exp_j  = current_exp.exponents[j].copy()

            subprocess.run(["cmafex", str(CYCLE_DIR / "run.yaml")], check=True)

            new_exp      = Exponent_Set.load(MEMORY_FILE)
            delta_e      = new_exp.energy - pre_energy
            cumulative_e = new_exp.energy - start_energy
            delta_pct    = float((abs(new_exp.exponents[j] - pre_exp_j) / pre_exp_j).mean()) * 100

            log.write(f"{SEP}\n")
            log.write(f"Cycle {i + 1:2d}/{CYCLES}  |  Shell {j} ({lbl})  |  {n_exp} exponents  |  popsize={gen_size}  |  max_gen={MAX_GENERATIONS}\n")
            log.write(f"  E_before  = {pre_energy:20.10f} Eh\n")
            log.write(f"  E_after   = {new_exp.energy:20.10f} Eh\n")
            log.write(f"  dE        = {delta_e:+20.10f} Eh\n")
            log.write(f"  dE_total  = {cumulative_e:+20.10f} Eh\n")
            log.write(f"  exp_chg   = {delta_pct:19.4f} %\n")
            log.write(f"{SEP}\n\n")
            log.flush()

            csv_writer.writerow([
                i + 1, j, lbl, n_exp, gen_size, MAX_GENERATIONS,
                f"{pre_energy:.10f}", f"{new_exp.energy:.10f}",
                f"{delta_e:.10f}", f"{cumulative_e:.10f}", f"{delta_pct:.4f}",
            ])
            csv_f.flush()

            blended              = new_exp.copy(no_energy=True)
            blended.exponents[j] = MU * new_exp.exponents[j] + (1 - MU) * pre_exp_j
            current_exp          = blended
            current_energy       = new_exp.energy
