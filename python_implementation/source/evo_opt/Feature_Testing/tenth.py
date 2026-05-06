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
TOP_FRAC   = 0.10
DERIV_STEP = 0.01  # fractional step for symmetric finite differences: α ± δ·α

RUNS_DIR = WORK_DIR / "RUNS"

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


# ── Run 0: base calculation → center energy + contraction ────────────────────

exp = Exponent_Set.from_file(exp_path)

M0 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "BASE_RUN",
    full_logging=True,
    overwrite_existing=True,
)

M0.add_job(exp.copy(no_energy=True), template_dir)
M0.run_all_jobs(1)

base_job = M0.jobs[0]
if base_job.status != Job_Status.COMPLETED:
    raise RuntimeError("Base job failed — cannot proceed without a contraction.")

base_exp_set        = base_job.exponent_set
base_energy_uncontr = base_exp_set.energy

if base_exp_set.resulting_contraction is None:
    raise RuntimeError("Base job produced no ANO contraction — check your template.")

base_contraction  = base_exp_set.resulting_contraction
n_contr_per_shell = [mat.shape[0] for mat in base_contraction]
n_contr_total     = sum(n_contr_per_shell)

print(f"Base uncontracted energy : {base_energy_uncontr}")
print(f"Contraction per shell:")
for i, mat in enumerate(base_contraction):
    print(f"  Shell {i}: {mat.shape[1]} primitives -> {mat.shape[0]} contracted")
print(f"Total contracted functions: {n_contr_total}")

# ── Build exponent map and sample cloud ───────────────────────────────────────

# Global index k → (shell l, position q within shell)
exp_map = [
    (l, q)
    for l in range(len(exp.exponents))
    for q in range(len(exp.exponents[l]))
]

flat_base = np.array([
    float(exp.exponents[l][q])
    for l, q in exp_map
])
n_exponents = len(flat_base)

fracs        = rng.uniform(-MAX_FRAC, MAX_FRAC, size=(M_SAMPLES, n_exponents))
sampled_flat = flat_base[None, :] * (1.0 + fracs)

print(f"\nNumber of exponents : {n_exponents}")
print(f"Sampling {M_SAMPLES} points  ±{MAX_FRAC*100:.0f}%")
print(f"Derivative step     : ±{DERIV_STEP*100:.0f}%  ({2*n_exponents} derivative jobs)")

# ── Derivative runs: symmetric finite differences per exponent ────────────────

M_dplus = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "DERIV_PLUS",
    full_logging=True,
    overwrite_existing=True,
)

M_dminus = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "DERIV_MINUS",
    full_logging=True,
    overwrite_existing=True,
)

for k, (l, q) in enumerate(exp_map):
    alpha_k = float(exp.exponents[l][q])

    exp_plus = exp.copy(no_energy=True)
    exp_plus.exponents[l][q] = alpha_k * (1.0 + DERIV_STEP)
    M_dplus.add_job(exp_plus, template_dir)

    exp_minus = exp.copy(no_energy=True)
    exp_minus.exponents[l][q] = alpha_k * (1.0 - DERIV_STEP)
    M_dminus.add_job(exp_minus, template_dir)

print(f"\nRunning derivative jobs (+δ)...")
M_dplus.run_all_jobs(MAX_JOBS)
print(f"Running derivative jobs (-δ)...")
M_dminus.run_all_jobs(MAX_JOBS)

# ── Compute ∂C[l]/∂α_k for each exponent k ───────────────────────────────────

dC_per_exponent = []
n_deriv_failed  = 0

for k, (l, q) in enumerate(exp_map):
    alpha_k  = float(exp.exponents[l][q])
    job_plus  = M_dplus.jobs[k]
    job_minus = M_dminus.jobs[k]

    ok_plus  = (job_plus.status  == Job_Status.COMPLETED and
                job_plus.exponent_set.resulting_contraction  is not None)
    ok_minus = (job_minus.status == Job_Status.COMPLETED and
                job_minus.exponent_set.resulting_contraction is not None)

    if ok_plus and ok_minus:
        C_plus  = job_plus.exponent_set.resulting_contraction[l]
        C_minus = job_minus.exponent_set.resulting_contraction[l]

        if C_plus.shape != C_minus.shape:
            print(f"[Warning] k={k} (l={l}, q={q}): C_+ and C_- shape mismatch, using zero.")
            dC_k = np.zeros_like(base_contraction[l])
            n_deriv_failed += 1
        else:
            dC_k = (C_plus - C_minus) / (2.0 * DERIV_STEP * alpha_k)
    else:
        print(f"[Warning] k={k} (l={l}, q={q}): derivative run failed, using zero gradient.")
        dC_k = np.zeros_like(base_contraction[l])
        n_deriv_failed += 1

    dC_per_exponent.append(dC_k)

print(f"\nDerivatives computed: {n_exponents - n_deriv_failed}/{n_exponents} succeeded.")

# ── Run 1: uncontracted at each sample point ──────────────────────────────────

M1 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "CLOUD_UNCONTR",
    full_logging=True,
    overwrite_existing=True,
)

for i in range(M_SAMPLES):
    M1.add_job(build_exp_set_from_flat(exp, sampled_flat[i]), template_dir)

M1.run_all_jobs(MAX_JOBS)

# ── Run 2: each point contracted with its own ANO from Run 1 ─────────────────

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "CLOUD_SELF_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

r2_to_r1 = []

for job1 in M1.jobs:
    if job1.status != Job_Status.COMPLETED:
        continue
    if job1.exponent_set.resulting_contraction is None:
        print(f"[Info] Job {job1.job_id}: no ANO contraction, skipping self-contracted run.")
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

# ── Run 3: frozen center contraction at each sample point ─────────────────────

M3 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "CLOUD_FROZEN_CONTR",
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

# ── Run 4: first-order approximated contraction at each sample point ──────────

M4 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=RUNS_DIR / "CLOUD_FIRSTORDER_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

for i in range(M_SAMPLES):
    delta_exp = sampled_flat[i] - flat_base

    # C_approx[l] = C_base[l] + Σ_{k in shell l} (∂C[l]/∂α_k) · Δα_k
    C_approx = [base_contraction[l].copy() for l in range(len(exp.exponents))]
    for k, (l, q) in enumerate(exp_map):
        C_approx[l] = C_approx[l] + dC_per_exponent[k] * delta_exp[k]

    exp_varied = build_exp_set_from_flat(exp, sampled_flat[i])
    fo_exp = Exponent_Set(
        atom_name=exp_varied.atom_name,
        exponents=[e.copy() for e in exp_varied.exponents],
        contractions=C_approx,
        method=exp_varied.method,
    )
    M4.add_job(fo_exp, template_dir)

M4.run_all_jobs(MAX_JOBS)

# ── Collect results ───────────────────────────────────────────────────────────

# Self-contraction pairs (Run 1 vs Run 2)
samples_self = []
eu_self, ec_self = [], []

for job1, job2 in zip(r2_to_r1, M2.jobs):
    eu = job1.exponent_set.energy if job1.status == Job_Status.COMPLETED else None
    ec = job2.exponent_set.energy if job2.status == Job_Status.COMPLETED else None
    samples_self.append({"i": job1.job_id, "e_uncontr": eu, "e_contr": ec})
    if eu is not None and ec is not None:
        eu_self.append(eu)
        ec_self.append(ec)

eu_self = np.array(eu_self)
ec_self = np.array(ec_self)
n_self  = len(eu_self)

# Frozen-contraction pairs (Run 1 vs Run 3)
samples_frozen = []
eu_frozen, ec_frozen = [], []

for i, (job1, job3) in enumerate(zip(M1.jobs, M3.jobs)):
    eu = job1.exponent_set.energy if job1.status == Job_Status.COMPLETED else None
    ec = job3.exponent_set.energy if job3.status == Job_Status.COMPLETED else None
    samples_frozen.append({
        "i": i, "fracs": fracs[i].tolist(), "exps": sampled_flat[i].tolist(),
        "e_uncontr": eu, "e_contr": ec,
    })
    if eu is not None and ec is not None:
        eu_frozen.append(eu)
        ec_frozen.append(ec)

eu_frozen = np.array(eu_frozen)
ec_frozen = np.array(ec_frozen)
n_frozen  = len(eu_frozen)

# First-order contraction pairs (Run 1 vs Run 4)
samples_fo = []
eu_fo, ec_fo = [], []

for i, (job1, job4) in enumerate(zip(M1.jobs, M4.jobs)):
    eu = job1.exponent_set.energy if job1.status == Job_Status.COMPLETED else None
    ec = job4.exponent_set.energy if job4.status == Job_Status.COMPLETED else None
    samples_fo.append({
        "i": i, "fracs": fracs[i].tolist(), "exps": sampled_flat[i].tolist(),
        "e_uncontr": eu, "e_contr": ec,
    })
    if eu is not None and ec is not None:
        eu_fo.append(eu)
        ec_fo.append(ec)

eu_fo = np.array(eu_fo)
ec_fo = np.array(ec_fo)
n_fo  = len(eu_fo)

print(f"\nSelf-contraction    : {n_self}/{M_SAMPLES} pairs")
print(f"Frozen contraction  : {n_frozen}/{M_SAMPLES} pairs")
print(f"First-order approx  : {n_fo}/{M_SAMPLES} pairs")

# ── Statistics helpers ────────────────────────────────────────────────────────

def spearman_stats(eu, ec):
    if len(eu) < 2:
        return None, None, None, None
    rho, pval  = spearmanr(eu, ec)
    rank_u     = np.argsort(np.argsort(eu))
    rank_c     = np.argsort(np.argsort(ec))
    rank_agree = float(np.mean(rank_u == rank_c))
    return float(rho), float(pval), rank_agree, ec - eu


def top_k_idx(energies, frac):
    k = max(2, int(np.ceil(len(energies) * frac)))
    return np.argsort(energies)[:k]


def compute_stats(eu, ec):
    if len(eu) < 2:
        return dict(all=None, top=None, top_k=None)
    rho_a, pval_a, agree_a, delta_a = spearman_stats(eu, ec)
    tk = top_k_idx(eu, TOP_FRAC)
    rho_t, pval_t, agree_t, delta_t = spearman_stats(eu[tk], ec[tk])
    return dict(
        all  = (rho_a, pval_a, agree_a, delta_a, len(eu)),
        top  = (rho_t, pval_t, agree_t, delta_t, len(tk)),
        top_k = tk,
    )


stats_self   = compute_stats(eu_self,   ec_self)
stats_frozen = compute_stats(eu_frozen, ec_frozen)
stats_fo     = compute_stats(eu_fo,     ec_fo)

# ── Printing helpers ──────────────────────────────────────────────────────────

W = 60

def _sep(title=""):
    pad = max(0, W - 6 - len(title))
    return f"  ── {title} " + "─" * pad


def _block_lines(stats, label):
    lines = []
    if stats["all"] is None:
        lines.append("  Not enough valid pairs.")
        return lines
    rho_a, pval_a, agree_a, delta_a, n_a = stats["all"]
    rho_t, pval_t, agree_t, delta_t, n_t = stats["top"]
    lines.append(_sep("All samples"))
    lines.append(f"  Spearman ρ         {rho_a:>+.4f}")
    lines.append(f"  p-value            {pval_a:>.2e}")
    lines.append(f"  Rank agreement     {agree_a:.1%}  ({int(agree_a*n_a)}/{n_a})")
    lines.append(f"  ΔE  mean           {delta_a.mean():>+.6f}  Ha")
    lines.append(f"  ΔE  std            {delta_a.std():>.6f}  Ha")
    lines.append(f"  ΔE  min / max      {delta_a.min():>+.6f} / {delta_a.max():>+.6f}")
    lines.append("")
    lines.append(_sep(f"Top {TOP_FRAC*100:.0f}%  ({n_t} samples)"))
    if rho_t is not None:
        lines.append(f"  Spearman ρ         {rho_t:>+.4f}")
        lines.append(f"  p-value            {pval_t:>.2e}")
        lines.append(f"  Rank agreement     {agree_t:.1%}  ({int(agree_t*n_t)}/{n_t})")
        lines.append(f"  ΔE  mean           {delta_t.mean():>+.6f}  Ha")
        lines.append(f"  ΔE  std            {delta_t.std():>.6f}  Ha")
    else:
        lines.append("  Not enough samples.")
    return lines


def print_section(title, n_valid, stats):
    print()
    print("═" * W)
    print(f"{title:^{W}}")
    print(f"  Valid pairs : {n_valid} / {M_SAMPLES}")
    print()
    for line in _block_lines(stats, title):
        print(line)


# ── Pretty print ──────────────────────────────────────────────────────────────

print("\n" + "═" * W)
print(f"{'CONTRACTION RANK TEST':^{W}}")
print("═" * W)
print(f"  Samples             {M_SAMPLES}")
print(f"  Exponents           {n_exponents}")
print(f"  Perturbation        ±{MAX_FRAC*100:.0f}%")
print(f"  Derivative step     ±{DERIV_STEP*100:.0f}%  ({n_deriv_failed} failed)")
print(f"  Base energy         {base_energy_uncontr}")
print(f"  Contracted funcs    {n_contr_total}")

print_section("SELF-CONTRACTION  (uncontr vs own ANO)",         n_self,   stats_self)
print_section("FROZEN CONTRACTION  (uncontr vs center ANO)",    n_frozen, stats_frozen)
print_section("FIRST-ORDER CONTRACTION  (uncontr vs ∇C·Δα)",   n_fo,     stats_fo)
print("═" * W)

# ── Save JSON ─────────────────────────────────────────────────────────────────

def _safe(x):
    return float(x) if x is not None else None


def _stats_json(stats):
    def _block(tup):
        if tup is None:
            return None
        rho, pval, agree, delta, n = tup
        return {
            "n_samples":      n,
            "spearman_rho":   _safe(rho),
            "spearman_pval":  _safe(pval),
            "rank_agreement": _safe(agree),
            "delta_mean":     _safe(delta.mean()),
            "delta_std":      _safe(delta.std()),
            "delta_min":      _safe(delta.min()),
            "delta_max":      _safe(delta.max()),
        }
    return {
        "all_samples": _block(stats["all"]),
        "top10":       _block(stats["top"]),
    }


out = {
    "meta": {
        "n_samples":          M_SAMPLES,
        "max_frac":           MAX_FRAC,
        "seed":               SEED,
        "n_exponents":        n_exponents,
        "deriv_step":         DERIV_STEP,
        "n_deriv_failed":     n_deriv_failed,
        "base_energy":        base_energy_uncontr,
        "n_contr_per_shell":  n_contr_per_shell,
        "n_contr_total":      n_contr_total,
    },
    "self_contraction": {
        "n_valid": n_self,
        "stats":   _stats_json(stats_self),
        "samples": samples_self,
    },
    "frozen_contraction": {
        "n_valid": n_frozen,
        "stats":   _stats_json(stats_frozen),
        "samples": samples_frozen,
    },
    "firstorder_contraction": {
        "n_valid": n_fo,
        "stats":   _stats_json(stats_fo),
        "samples": samples_fo,
    },
}

json_path = SUBMIT_DIR / "contraction_rank_test.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nJSON  saved to {json_path}")

# ── Save log ──────────────────────────────────────────────────────────────────

log_path = SUBMIT_DIR / "contraction_rank_test.log"

with open(log_path, "w") as f:
    def w(line=""):
        f.write(line + "\n")

    w("CONTRACTION RANK TEST")
    w("=" * W)
    w()
    w(f"  Base exponents       : {exp_path}")
    w(f"  Samples              : {M_SAMPLES}")
    w(f"  Perturbation         : ±{MAX_FRAC*100:.0f}%")
    w(f"  Derivative step      : ±{DERIV_STEP*100:.0f}%  ({n_deriv_failed} failed)")
    w(f"  Seed                 : {SEED}")
    w(f"  N exponents          : {n_exponents}")
    w(f"  Base energy (uncontr): {base_energy_uncontr}")
    w(f"  Contraction per shell:")
    for i, mat in enumerate(base_contraction):
        w(f"    Shell {i}: {mat.shape[1]} primitives -> {mat.shape[0]} contracted")
    w(f"  Total contracted     : {n_contr_total}")
    w()

    def write_section(title, n_valid, stats):
        w("=" * W)
        w(title)
        w(f"Valid pairs : {n_valid} / {M_SAMPLES}")
        w()
        for line in _block_lines(stats, title):
            w(line)
        w()

    write_section("SELF-CONTRACTION  (uncontracted vs each point's own ANO)",  n_self,   stats_self)
    write_section("FROZEN CONTRACTION  (uncontracted vs fixed center ANO)",     n_frozen, stats_frozen)
    write_section("FIRST-ORDER CONTRACTION  (uncontracted vs ∇C·Δα)",          n_fo,     stats_fo)

    w("=" * W)
    w("PER-SAMPLE RESULTS")
    w("─" * W)
    w(f"  {'i':>4}  {'Uncontr':>14}  {'Self':>14}  {'Frozen':>14}  {'1st-ord':>14}"
      f"  {'ΔE self':>11}  {'ΔE frz':>11}  {'ΔE 1st':>11}")
    w(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*14}"
      f"  {'─'*11}  {'─'*11}  {'─'*11}")

    self_by_id   = {s["i"]: s for s in samples_self}
    frozen_by_id = {s["i"]: s for s in samples_frozen}
    fo_by_id     = {s["i"]: s for s in samples_fo}

    for i in range(M_SAMPLES):
        eu  = frozen_by_id.get(i, {}).get("e_uncontr")
        esc = self_by_id.get(i, {}).get("e_contr")
        efc = frozen_by_id.get(i, {}).get("e_contr")
        efo = fo_by_id.get(i, {}).get("e_contr")

        def _e(v):
            return f"{v:.8f}" if v is not None else "   FAILED  "
        def _d(v):
            return f"{v-eu:>+.5f}" if (eu is not None and v is not None) else "    N/A   "

        w(f"  {i:>4}  {_e(eu):>14}  {_e(esc):>14}  {_e(efc):>14}  {_e(efo):>14}"
          f"  {_d(esc):>11}  {_d(efc):>11}  {_d(efo):>11}")

    w()
    w("=" * W)

print(f"Log   saved to {log_path}")
