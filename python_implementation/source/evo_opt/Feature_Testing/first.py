import sys
from pathlib import Path

BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR))

from evo_opt.exponent_handler import *
from evo_opt.molcas_handler import *
from evo_opt.job_manager import *
from evo_opt.common import Executor_Type

exp_path     = BASE_DIR / "exp.expo"
template_dir = BASE_DIR / "template.inp"
submit_scr   = BASE_DIR / "run.sh"
extract_scr  = BASE_DIR / "extract.sh"
dest_dir     = BASE_DIR / "RUN1"

exp = Exponent_Set.from_file(exp_path)

print(BASE_DIR)

# ── Run 1: uncontracted ──────────────────────────────────────────────────────

M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=dest_dir,
    full_logging=True,
    overwrite_existing=True,
)

for i in range(10):
    exp_copy = exp.copy(no_energy=True)
    exp_copy.exponents[0] *= (1 + 0.001 * i)
    M1.add_job(exp_copy, template_dir)

M1.run_all_jobs(1)

# ── Run 2: contracted (using ANO contractions from Run 1) ────────────────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=BASE_DIR / "RUN2",
    full_logging=True,
    overwrite_existing=True,
)

pairs = []  # (run1_job, run2_job) for energy comparison

for job in M1.jobs:
    if job.status != Job_Status.COMPLETED:
        continue
    exp_set = job.exponent_set
    if exp_set.resulting_contraction is None:
        print(f"[Info] Job {job.job_id}: no ANO contractions found, skipping contracted run.")
        continue

    contracted_exp = Exponent_Set(
        atom_name=exp_set.atom_name,
        exponents=[e.copy() for e in exp_set.exponents],
        contractions=exp_set.resulting_contraction,
        method=exp_set.method,
    )

    job2 = M2.add_job(contracted_exp, template_dir)
    pairs.append((job, job2))

M2.run_all_jobs(1)

# ── Energy comparison ────────────────────────────────────────────────────────

print("\n" + "─" * 58)
print(f"{'Job':>4} | {'Uncontracted':>22} | {'Contracted':>22}")
print("─" * 58)

for job1, job2 in pairs:
    e1 = job1.exponent_set.energy
    e2 = job2.exponent_set.energy
    s1 = f"{e1:.10f}" if e1 is not None else "FAILED"
    s2 = f"{e2:.10f}" if e2 is not None else "FAILED"
    print(f"{job1.job_id:>4} | {s1:>22} | {s2:>22}")

print("─" * 58)
