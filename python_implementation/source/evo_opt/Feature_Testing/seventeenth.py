import sys
import json
from pathlib import Path
from scipy.stats import spearmanr
from numpy import array, float64, mean, std
from numpy.random import default_rng

WORK_DIR   = Path.cwd() / "Sampling"
SUBMIT_DIR = Path.cwd()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.job_manager import Job_Manager, Job_Status
from evo_opt.common import Executor_Type

exp_path    = SUBMIT_DIR / "exp.expo"
template    = SUBMIT_DIR / "template.inp"
submit_scr  = SUBMIT_DIR / "run.sh"
extract_scr = SUBMIT_DIR / "extract.sh"

M_SAMPLES = 12
MAX_FRAC  = 0.05
MAX_JOBS  = 4
SEED      = 42

rng = default_rng(SEED)

exp = Exponent_Set.from_file(exp_path)

flat_base = array([
    float(exp.exponents[l][q])
    for l in range(len(exp.exponents))
    for q in range(len(exp.exponents[l]))
], dtype=float64)

n_exponents  = len(flat_base)
fracs        = rng.uniform(-MAX_FRAC, MAX_FRAC, size=(M_SAMPLES, n_exponents))
sampled_flat = flat_base[None, :] * (1.0 + fracs)

print(f"Exponents : {n_exponents}")
print(f"Samples   : {M_SAMPLES}  ±{MAX_FRAC * 100:.0f}%")


def build_exp_from_flat(base, flat):
    new_exponents = []
    idx = 0
    for l in range(len(base.exponents)):
        n          = len(base.exponents[l])
        shell_vals = flat[idx : idx + n].copy()
        shell_vals[::-1].sort()
        new_exponents.append(shell_vals)
        idx += n
    return Exponent_Set(atom_name=base.atom_name, exponents=new_exponents, method=base.method)


M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path     = WORK_DIR / "UNCONTR",
    full_logging       = True,
    overwrite_existing = True,
)

for i in range(M_SAMPLES):
    sample = build_exp_from_flat(exp, sampled_flat[i])
    sample.uncontract_all()
    M1.add_job(sample, template)

M1.run_all_jobs(MAX_JOBS)

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path     = WORK_DIR / "SELF_CONTR",
    full_logging       = True,
    overwrite_existing = True,
)

m2_to_m1 = []
for i in range(M_SAMPLES):
    job1 = M1.jobs[i]
    if job1.status != Job_Status.COMPLETED:
        continue
    if job1.exponent_set.resulting_contraction is None:
        continue
    exp_s      = job1.exponent_set
    contracted = Exponent_Set(
        atom_name    = exp_s.atom_name,
        exponents    = [e.copy() for e in exp_s.exponents],
        contractions = [c.copy() for c in exp_s.resulting_contraction],
        method       = exp_s.method,
    )
    M2.add_job(contracted, template)
    m2_to_m1.append(i)

M2.run_all_jobs(MAX_JOBS)

eu_list = []
es_list = []
for k in range(len(m2_to_m1)):
    i    = m2_to_m1[k]
    job2 = M2.jobs[k]
    if job2.status != Job_Status.COMPLETED:
        continue
    eu_list.append(float(M1.jobs[i].exponent_set.energy))
    es_list.append(float(job2.exponent_set.energy))

eu_arr  = array(eu_list, dtype=float64)
es_arr  = array(es_list, dtype=float64)
delta   = es_arr - eu_arr
n_valid = len(eu_arr)

rho, pval = spearmanr(eu_arr, es_arr)

W   = 68
SEP = "=" * W

print(f"\n{SEP}")
print(f"{'RESULTS':^{W}}")
print(f"{SEP}")
print(f"  Valid pairs   : {n_valid} / {M_SAMPLES}")
print(f"\n  Uncontracted energies")
print(f"    mean  = {mean(eu_arr):+.10f}  Eh")
print(f"    std   = {std(eu_arr):.10f}  Eh")
print(f"\n  Self-contracted energies")
print(f"    mean  = {mean(es_arr):+.10f}  Eh")
print(f"    std   = {std(es_arr):.10f}  Eh")
print(f"\n  Contraction energy loss (contracted - uncontracted)")
print(f"    mean  = {mean(delta):+.10f}  Eh")
print(f"    std   = {std(delta):.10f}  Eh")
print(f"\n  Spearman rank correlation")
print(f"    rho   = {rho:+.10f}")
print(f"    p     = {pval:.4e}")
print(f"{SEP}")

out = {
    "meta": {
        "n_samples":   M_SAMPLES,
        "n_valid":     n_valid,
        "max_frac":    MAX_FRAC,
        "seed":        SEED,
        "n_exponents": n_exponents,
    },
    "uncontr": {
        "mean": float(mean(eu_arr)),
        "std":  float(std(eu_arr)),
    },
    "self_contr": {
        "mean": float(mean(es_arr)),
        "std":  float(std(es_arr)),
    },
    "loss": {
        "mean": float(mean(delta)),
        "std":  float(std(delta)),
    },
    "spearman": {
        "rho":  float(rho),
        "pval": float(pval),
    },
}

out_path = WORK_DIR / "rank_test.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to {out_path}")
