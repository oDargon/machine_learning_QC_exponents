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

MAX_JOBS  = 1
N_STEPS   = 11
STEP_FRACS = linspace(-0.10, 0.10, N_STEPS)  # -10% … 0% … +10%

exp = Exponent_Set.from_file(exp_path)

# ── Run 1: uncontracted scan ─────────────────────────────────────────────────

M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "SCAN_RUN1",
    full_logging=True,
    overwrite_existing=True,
)

# One job per (shell, exponent, step).  Record metadata in the same order.
meta = []  # list of (l, q, base_value, step_frac)

for l, shell_exps in enumerate(exp.exponents):
    for q in range(len(shell_exps)):
        base = float(shell_exps[q])
        for frac in STEP_FRACS:
            exp_copy = exp.copy(no_energy=True)
            exp_copy.exponents[l][q] = base * (1.0 + frac)
            M1.add_job(exp_copy, template_dir)
            meta.append((l, q, base, float(frac)))

M1.run_all_jobs(MAX_JOBS)

# ── Run 2: contracted (ANO from Run 1) ───────────────────────────────────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "SCAN_RUN2",
    full_logging=True,
    overwrite_existing=True,
)

pairs = []  # (job1, job2, meta_entry)

for job1, m in zip(M1.jobs, meta):
    if job1.status != Job_Status.COMPLETED:
        continue
    exp_set = job1.exponent_set
    if exp_set.resulting_contraction is None:
        print(f"[Info] Job {job1.job_id}: no ANO contractions, skipping contracted run.")
        continue

    contracted_exp = Exponent_Set(
        atom_name=exp_set.atom_name,
        exponents=[e.copy() for e in exp_set.exponents],
        contractions=exp_set.resulting_contraction,
        method=exp_set.method,
    )

    job2 = M2.add_job(contracted_exp, template_dir)
    pairs.append((job1, job2, m))

M2.run_all_jobs(MAX_JOBS)

# ── Save results ─────────────────────────────────────────────────────────────

results = []
for job1, job2, (l, q, base, frac) in pairs:
    e1 = job1.exponent_set.energy
    e2 = job2.exponent_set.energy
    results.append({
        "shell":            l,
        "exponent_index":   q,
        "base_value":       base,
        "step_fraction":    frac,
        "exponent_value":   base * (1.0 + frac),
        "uncontracted":     e1,
        "contracted":       e2,
        "difference":       (e2 - e1) if (e1 is not None and e2 is not None) else None,
    })

out_path = WORK_DIR / "scan_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to {out_path}")
