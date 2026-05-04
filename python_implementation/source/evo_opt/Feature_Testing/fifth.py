import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

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

MAX_JOBS   = 4
M_SAMPLES  = 20
MAX_FRAC   = 0.05
SEED       = 42

rng = np.random.default_rng(SEED)

def build_exp_set_from_flat(base_exp: Exponent_Set, flat_exponents: np.ndarray) -> Exponent_Set:
    new_exponents = []
    idx = 0
    for l in range(len(base_exp.exponents)):
        n = len(base_exp.exponents[l])
        shell_vals = flat_exponents[idx : idx + n].copy()
        shell_vals[::-1].sort()
        new_exponents.append(shell_vals)
        idx += n
    return Exponent_Set(
        atom_name=base_exp.atom_name,
        exponents=new_exponents,
        method=base_exp.method,
    )

# ── Sample random points ──────────────────────────────────────────────────────

exp = Exponent_Set.from_file(exp_path)

flat_base = np.array([
    float(exp.exponents[l][q])
    for l in range(len(exp.exponents))
    for q in range(len(exp.exponents[l]))
])
n_exponents = len(flat_base)

fracs        = rng.uniform(-MAX_FRAC, MAX_FRAC, size=(M_SAMPLES, n_exponents))
sampled_flat = flat_base[None, :] * (1.0 + fracs)

print(f"Number of exponents : {n_exponents}")
print(f"Sampling {M_SAMPLES} points  ±{MAX_FRAC*100:.0f}%")

# ── Run 1: uncontracted at each sample point ─────────────────────────────────

M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_UNCONTR",
    full_logging=True,
    overwrite_existing=True,
)

for i in range(M_SAMPLES):
    M1.add_job(build_exp_set_from_flat(exp, sampled_flat[i]), template_dir)

M1.run_all_jobs(MAX_JOBS)

# ── Run 2: contracted at each point using that point's own contraction ────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_SELF_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

# Track which Run 1 job each Run 2 job corresponds to
r2_to_r1 = []

for job1 in M1.jobs:
    if job1.status != Job_Status.COMPLETED:
        continue
    if job1.exponent_set.resulting_contraction is None:
        print(f"[Info] Job {job1.job_id}: no ANO contraction, skipping.")
        continue

    exp_set = job1.exponent_set
    contracted_exp = Exponent_Set(
        atom_name=exp_set.atom_name,
        exponents=[e.copy() for e in exp_set.exponents],
        contractions=[c.copy() for c in exp_set.resulting_contraction],
        method=exp_set.method,
    )
    M2.add_job(contracted_exp, template_dir)
    r2_to_r1.append(job1)

M2.run_all_jobs(MAX_JOBS)

# ── Collect results ───────────────────────────────────────────────────────────

samples   = []
e_uncontr = []
e_contr   = []

for job1, job2 in zip(r2_to_r1, M2.jobs):
    eu = job1.exponent_set.energy if job1.status == Job_Status.COMPLETED else None
    ec = job2.exponent_set.energy if job2.status == Job_Status.COMPLETED else None

    contr = job1.exponent_set.resulting_contraction
    n_contr_per_shell = [mat.shape[0] for mat in contr] if contr is not None else None
    n_contr_total     = sum(n_contr_per_shell) if n_contr_per_shell is not None else None

    samples.append({
        "i":               job1.job_id,
        "exps":            sampled_flat[job1.job_id].tolist(),
        "e_uncontr":       eu,
        "e_contr":         ec,
        "n_contr_shells":  n_contr_per_shell,
        "n_contr_total":   n_contr_total,
    })

    if eu is not None and ec is not None:
        e_uncontr.append(eu)
        e_contr.append(ec)

e_uncontr = np.array(e_uncontr)
e_contr   = np.array(e_contr)
n_valid   = len(e_uncontr)

print(f"\n{n_valid}/{M_SAMPLES} pairs completed.")

if n_valid >= 2:
    rho, pval    = spearmanr(e_uncontr, e_contr)
    rank_u       = np.argsort(np.argsort(e_uncontr))
    rank_c       = np.argsort(np.argsort(e_contr))
    rank_agree   = float(np.mean(rank_u == rank_c))
    delta        = e_contr - e_uncontr
else:
    rho = pval = rank_agree = delta = None

W = 52
print("\n" + "═" * W)
print(f"{'RESULTS':^{W}}")
print("═" * W)
print(f"  Samples            {n_valid:>4} / {M_SAMPLES}")

if n_valid >= 2:
    print(f"  Spearman ρ         {rho:>+.4f}")
    print(f"  p-value            {pval:>.2e}")
    print("─" * W)
    print(f"  Rank agreement     {rank_agree:.1%}  ({int(rank_agree*n_valid)}/{n_valid} identical)")
    print("─" * W)
    print(f"  ΔE  mean           {delta.mean():>+.6f}  Hartree")
    print(f"  ΔE  std            {delta.std():>.6f}  Hartree")
    print(f"  ΔE  min / max      {delta.min():>+.6f} / {delta.max():>+.6f}")
    print("─" * W)
    print(f"  {'Sample':>6}  {'Uncontracted':>18}  {'Contracted':>18}")
    print("─" * W)
    for s in samples:
        eu, ec = s["e_uncontr"], s["e_contr"]
        su = f"{eu:.8f}" if eu is not None else "  FAILED "
        sc = f"{ec:.8f}" if ec is not None else "  FAILED "
        print(f"  {s['i']:>6}  {su:>18}  {sc:>18}")
else:
    print("  Not enough valid pairs for statistics.")

print("═" * W)

# ── Save ──────────────────────────────────────────────────────────────────────

out = {
    "meta": {
        "n_samples":  M_SAMPLES,
        "n_valid":    n_valid,
        "max_frac":   MAX_FRAC,
        "seed":       SEED,
        "n_exponents": n_exponents,
    },
    "stats": {
        "spearman_rho":       float(rho)        if rho        is not None else None,
        "spearman_pval":      float(pval)       if pval       is not None else None,
        "rank_agreement":     rank_agree        if rank_agree is not None else None,
        "delta_mean":         float(delta.mean()) if delta    is not None else None,
        "delta_std":          float(delta.std())  if delta    is not None else None,
    },
    "samples": samples,
}

out_path = SUBMIT_DIR / "self_contraction_rank_test.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to {out_path}")
