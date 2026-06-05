from pathlib import Path
from random import uniform

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.job_manager import Job_Manager_Config, Executor_Type
from evo_opt.objectives import Ground_Energy_Objective


BASE_DIR = Path(__file__).resolve().parent

exp_path       = BASE_DIR / "F_1.expo"
template_dir_g = BASE_DIR / "F_template_1.inp"
submit_scr     = BASE_DIR / "run.sh"
extrac_scr     = BASE_DIR / "extr.sh"
work_dir       = BASE_DIR / "F_ground_test"


# ============================================================
# Load initial exponent set
# ============================================================

base_exp = Exponent_Set.from_file(exp_path)


# ============================================================
# Job manager config
# ============================================================

C = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extrac_scr,
    manager_logging      = False,
    overwrite_existing   = True,
    custom_poll_interval = 1,
)

C.group_dir_path = work_dir


# ============================================================
# Objective
# ============================================================

O = Ground_Energy_Objective(template_dir_g, C)


# ============================================================
# Generate 100 randomized exponent sets
# ============================================================

trial_exponents = []
names           = []

for i in range(100):

    new_exp = base_exp.copy(no_energy=True)

    # Randomly perturb every exponent
    for l in range(len(new_exp.exponents)):

        for q in range(len(new_exp.exponents[l])):

            scale = uniform(0.8, 1.2)

            new_exp.exponents[l][q] *= scale

    trial_exponents.append(new_exp)
    names.append(f"trial_{i:03d}")


# ============================================================
# Evaluate all jobs
# ============================================================

results = O.evaluate_batch(
    trial_exponents,
    threads=4,
    names=names,
)


# ============================================================
# Print energies
# ============================================================

print("\nResults")
print("-" * 60)

for name, exp in zip(names, results):
    print(f"{name:15s} Energy = {exp.energy}")


# ============================================================
# Find best result
# ============================================================

best_exp = min(results, key=lambda e: e.energy)

print("\nBest Energy:")
print(best_exp.energy)

print("\nBest Exponent Set:")
print(best_exp)