from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *

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

exp_copy = exp.copy(no_energy=True)
M.add_job(exp_copy, template_dir)

for l in range(len(exp.exponents)):
    for q in range(exp.lengths[l]):
        exp_copy = exp.copy(no_energy=True)
        exp_copy.remove_exponent_uncontracted(l, q)
        M.add_job(exp_copy, template_dir, name=f"Removed_exp_{l}_{q}")


M.run_all_jobs(1, 0.1)


ref_energy = M.jobs[0].exponent_set.energy

energy_diffs = [
    (job.expo_name, job.exponent_set.energy - ref_energy)
    for job in M.jobs[1:]   # skip reference
]

# Print all differences
print("\nEnergy differences (relative to reference):")
for name, diff in energy_diffs:
    print(f"{name:20s}  ΔE = {diff:.8f}")

# Find smallest difference (most negative OR closest to zero)
best_job_name, best_diff = min(energy_diffs, key=lambda x: x[1])

print("\nBest removal candidate:")
print(f"{best_job_name}  with ΔE = {best_diff:.8f}")