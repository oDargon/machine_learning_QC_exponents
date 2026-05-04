import sys
import json
from pathlib import Path
from numpy import linspace

WORK_DIR   = Path(sys.argv[1])
SUBMIT_DIR = Path(sys.argv[2])
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import *
from evo_opt.molcas_handler import *
from evo_opt.job_manager import *
from evo_opt.common import Executor_Type

exp_path     = SUBMIT_DIR / "exp.expo"
template_dir = SUBMIT_DIR / "template.inp"
submit_scr   = SUBMIT_DIR / "run.sh"
extract_scr  = SUBMIT_DIR / "extract.sh"

MAX_JOBS   = 1
N_STEPS    = 11
STEP_FRACS = linspace(-0.20, 0.20, N_STEPS)  # -20% … 0% … +20%

exp = Exponent_Set.from_file(exp_path)

# ── Run 1: single base calculation to obtain the contraction ─────────────────

M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "BASE_RUN",
    full_logging=True,
    overwrite_existing=True,
)

M1.add_job(exp.copy(no_energy=True), template_dir)
M1.run_all_jobs(1)

base_job = M1.jobs[0]

if base_job.status != Job_Status.COMPLETED:
    raise RuntimeError("Base job failed — cannot proceed without a contraction.")

base_exp_set = base_job.exponent_set

if base_exp_set.resulting_contraction is None:
    raise RuntimeError("Base job produced no ANO contraction — check your template.")

base_contraction    = base_exp_set.resulting_contraction
base_energy_uncontr = base_exp_set.energy

print(f"Base uncontracted energy: {base_energy_uncontr}")
print(f"Contraction obtained for {len(base_contraction)} shells:")
for i, mat in enumerate(base_contraction):
    print(f"  Shell {i}: {mat.shape[1]} primitives -> {mat.shape[0]} contracted")

# ── Build shared job list ─────────────────────────────────────────────────────
# Each entry: (l, q, base_val, frac, exp_varied)
# Built once, reused for both Run 2 and Run 3.

variations = []
for l, shell_exps in enumerate(exp.exponents):
    for q in range(len(shell_exps)):
        base_val = float(shell_exps[q])
        for frac in STEP_FRACS:
            exp_copy = exp.copy(no_energy=True)
            exp_copy.exponents[l][q] = base_val * (1.0 + frac)
            variations.append((l, q, base_val, float(frac), exp_copy))

# ── Run 2: uncontracted scan ──────────────────────────────────────────────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "SCAN_UNCONTR",
    full_logging=True,
    overwrite_existing=True,
)

for l, q, base_val, frac, exp_varied in variations:
    M2.add_job(exp_varied.copy(no_energy=True), template_dir)

M2.run_all_jobs(MAX_JOBS)

# ── Run 3: scan with fixed base contraction ───────────────────────────────────

M3 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "SCAN_FIXED_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

for l, q, base_val, frac, exp_varied in variations:
    contracted_exp = Exponent_Set(
        atom_name=exp_varied.atom_name,
        exponents=[e.copy() for e in exp_varied.exponents],
        contractions=[c.copy() for c in base_contraction],
        method=exp_varied.method,
    )
    M3.add_job(contracted_exp, template_dir)

M3.run_all_jobs(MAX_JOBS)

# ── Save results ──────────────────────────────────────────────────────────────

results = []
for (l, q, base_val, frac, _), job2, job3 in zip(variations, M2.jobs, M3.jobs):
    e_uncontr = job2.exponent_set.energy if job2.status == Job_Status.COMPLETED else None
    e_contr   = job3.exponent_set.energy if job3.status == Job_Status.COMPLETED else None
    results.append({
        "shell":              l,
        "exponent_index":     q,
        "base_value":         base_val,
        "step_fraction":      frac,
        "exponent_value":     base_val * (1.0 + frac),
        "base_uncontracted":  base_energy_uncontr,
        "uncontracted":       e_uncontr,
        "contracted":         e_contr,
        "difference":         (e_contr - e_uncontr) if (e_uncontr is not None and e_contr is not None) else None,
    })

out_path = WORK_DIR / "scan_fixed_contraction.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {out_path}")
