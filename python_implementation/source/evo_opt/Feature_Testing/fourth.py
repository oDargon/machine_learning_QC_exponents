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
M_SAMPLES  = 20    # number of points in the cloud
MAX_FRAC   = 0.05  # each exponent may vary at most ±5%
SEED       = 42

rng = np.random.default_rng(SEED)

def build_exp_set_from_flat(base_exp: Exponent_Set, flat_exponents: np.ndarray) -> Exponent_Set:
    """Return a new Exponent_Set with exponents replaced by flat_exponents, preserving descending order per shell."""
    new_exponents = []
    idx = 0
    for l in range(len(base_exp.exponents)):
        n = len(base_exp.exponents[l])
        shell_vals = flat_exponents[idx : idx + n].copy()
        shell_vals[::-1].sort()  # ensure descending order
        new_exponents.append(shell_vals)
        idx += n

    return Exponent_Set(
        atom_name=base_exp.atom_name,
        exponents=new_exponents,
        method=base_exp.method,
    )

# ── Run 1: base calculation to obtain the contraction ────────────────────────

exp = Exponent_Set.from_file(exp_path)

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

# Flatten all exponent values for convenient sampling
flat_base = np.array([
    float(exp.exponents[l][q])
    for l in range(len(exp.exponents))
    for q in range(len(exp.exponents[l]))
])
n_exponents = len(flat_base)

print(f"Base uncontracted energy : {base_energy_uncontr}")
print(f"Number of exponents      : {n_exponents}")
print(f"Contraction per shell:")
for i, mat in enumerate(base_contraction):
    print(f"  Shell {i}: {mat.shape[1]} primitives -> {mat.shape[0]} contracted")
print(f"Sampling {M_SAMPLES} points with max per-exponent perturbation ±{MAX_FRAC*100:.0f}%")

# ── Sample M points in exponent space ────────────────────────────────────────

fracs        = rng.uniform(-MAX_FRAC, MAX_FRAC, size=(M_SAMPLES, n_exponents))
sampled_flat = flat_base[None, :] * (1.0 + fracs)   # shape (M_SAMPLES, n_exponents)

# ── Run 2: uncontracted energies for each sample ─────────────────────────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_UNCONTR",
    full_logging=True,
    overwrite_existing=True,
)

for i in range(M_SAMPLES):
    M2.add_job(build_exp_set_from_flat(exp, sampled_flat[i]), template_dir)

M2.run_all_jobs(MAX_JOBS)

# ── Run 3: frozen-contraction energies for each sample ───────────────────────

M3 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_FIXED_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

for i in range(M_SAMPLES):
    exp_varied = build_exp_set_from_flat(exp, sampled_flat[i])
    contracted_exp = Exponent_Set(
        atom_name=exp_varied.atom_name,
        exponents=[e.copy() for e in exp_varied.exponents],
        contractions=[c.copy() for c in base_contraction],
        method=exp_varied.method,
    )
    M3.add_job(contracted_exp, template_dir)

M3.run_all_jobs(MAX_JOBS)

# ── Collect results and compute rank correlation ──────────────────────────────

results   = []
e_uncontr = []
e_contr   = []

for i, (job2, job3) in enumerate(zip(M2.jobs, M3.jobs)):
    eu = job2.exponent_set.energy if job2.status == Job_Status.COMPLETED else None
    ec = job3.exponent_set.energy if job3.status == Job_Status.COMPLETED else None

    results.append({
        "sample_index":      i,
        "fracs":             fracs[i].tolist(),
        "exponent_values":   sampled_flat[i].tolist(),
        "energy_uncontr":    eu,
        "energy_contr":      ec,
    })

    if eu is not None and ec is not None:
        e_uncontr.append(eu)
        e_contr.append(ec)

e_uncontr = np.array(e_uncontr)
e_contr   = np.array(e_contr)

n_valid = len(e_uncontr)
print(f"\n{n_valid}/{M_SAMPLES} sample pairs completed successfully.")

if n_valid >= 2:
    rho, pval = spearmanr(e_uncontr, e_contr)
    print(f"Spearman rank correlation  ρ = {rho:.4f}  (p = {pval:.2e})")

    rank_u = np.argsort(np.argsort(e_uncontr))
    rank_c = np.argsort(np.argsort(e_contr))
    rank_agreement = np.mean(rank_u == rank_c)
    print(f"Exact rank agreement       = {rank_agreement:.2%}  ({int(rank_agreement*n_valid)}/{n_valid} positions identical)")

    delta = e_contr - e_uncontr
    print(f"Energy offset (contr-uncontr): mean={delta.mean():.6f}  std={delta.std():.6f} Hartree")
else:
    rho, pval, rank_agreement = None, None, None
    print("Not enough valid pairs to compute rank correlation.")

# ── Save results ──────────────────────────────────────────────────────────────

summary = {
    "n_samples":              M_SAMPLES,
    "n_valid":                n_valid,
    "max_frac":               MAX_FRAC,
    "seed":                   SEED,
    "base_energy_uncontr":    base_energy_uncontr,
    "spearman_rho":           float(rho)            if rho            is not None else None,
    "spearman_pval":          float(pval)           if pval           is not None else None,
    "exact_rank_agreement":   float(rank_agreement) if rank_agreement is not None else None,
    "samples":                results,
}

out_path = WORK_DIR / "rank_correlation_test.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved full results to {out_path}")
