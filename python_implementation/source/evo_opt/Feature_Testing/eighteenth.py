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

exp_path        = SUBMIT_DIR / "exp.expo"
template_nosym  = SUBMIT_DIR / "inp1.inp"
template_sym    = SUBMIT_DIR / "inp2.inp"
submit_scr      = SUBMIT_DIR / "run.sh"
extract_scr     = SUBMIT_DIR / "extract.sh"

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


M_NOSYM = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path     = WORK_DIR / "NOSYM",
    full_logging       = True,
    overwrite_existing = True,
)

for i in range(M_SAMPLES):
    sample = build_exp_from_flat(exp, sampled_flat[i])
    sample.uncontract_all()
    M_NOSYM.add_job(sample, template_nosym)

M_NOSYM.run_all_jobs(MAX_JOBS)

M_SYM = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path     = WORK_DIR / "SYM",
    full_logging       = True,
    overwrite_existing = True,
)

for i in range(M_SAMPLES):
    sample = build_exp_from_flat(exp, sampled_flat[i])
    sample.uncontract_all()
    M_SYM.add_job(sample, template_sym)

M_SYM.run_all_jobs(MAX_JOBS)

en_list = []
es_list = []
for i in range(M_SAMPLES):
    job_n = M_NOSYM.jobs[i]
    job_s = M_SYM.jobs[i]
    if job_n.status != Job_Status.COMPLETED:
        continue
    if job_s.status != Job_Status.COMPLETED:
        continue
    en_list.append(float(job_n.exponent_set.energy))
    es_list.append(float(job_s.exponent_set.energy))

en_arr  = array(en_list, dtype=float64)
es_arr  = array(es_list, dtype=float64)
delta   = es_arr - en_arr
n_valid = len(en_arr)

rho, pval = spearmanr(en_arr, es_arr)

W   = 68
SEP = "=" * W

print(f"\n{SEP}")
print(f"{'RESULTS':^{W}}")
print(f"{SEP}")
print(f"  Valid pairs   : {n_valid} / {M_SAMPLES}")
print(f"\n  No-symmetry energies (inp1)")
print(f"    mean  = {mean(en_arr):+.10f}  Eh")
print(f"    std   = {std(en_arr):.10f}  Eh")
print(f"\n  Symmetry energies (inp2)")
print(f"    mean  = {mean(es_arr):+.10f}  Eh")
print(f"    std   = {std(es_arr):.10f}  Eh")
print(f"\n  Energy difference (sym - nosym)")
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
    "nosym": {
        "mean": float(mean(en_arr)),
        "std":  float(std(en_arr)),
    },
    "sym": {
        "mean": float(mean(es_arr)),
        "std":  float(std(es_arr)),
    },
    "delta": {
        "mean": float(mean(delta)),
        "std":  float(std(delta)),
    },
    "spearman": {
        "rho":  float(rho),
        "pval": float(pval),
    },
}

out_path = WORK_DIR / "sym_vs_nosym.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to {out_path}")
