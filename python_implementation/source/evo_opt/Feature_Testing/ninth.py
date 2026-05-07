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
from evo_opt.shell_overlap import project_contraction

exp_path     = SUBMIT_DIR / "exp.expo"
template_dir = SUBMIT_DIR / "template.inp"
submit_scr   = SUBMIT_DIR / "run.sh"
extract_scr  = SUBMIT_DIR / "extract.sh"

MAX_JOBS   = 4
M_SAMPLES  = 20
MAX_FRAC   = 0.05
SEED       = 42
TOP_FRAC   = 0.10

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

def rank_stats(pairs):
    if len(pairs) < 2:
        return None, None, None, None
    a, b       = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    rho, pval  = spearmanr(a, b)
    rank_a     = np.argsort(np.argsort(a))
    rank_b     = np.argsort(np.argsort(b))
    agree      = float(np.mean(rank_a == rank_b))
    delta      = b - a
    return rho, pval, agree, delta

def top_pairs(pairs, frac=TOP_FRAC):
    k = max(2, round(len(pairs) * frac))
    return sorted(pairs, key=lambda p: p[0])[:k]

def contraction_loss(pairs, delta):
    if delta is None or len(pairs) < 1:
        return None
    eu_arr  = np.array([p[0] for p in pairs])
    rel_pct = delta / np.abs(eu_arr) * 100.0
    return {"delta": delta, "rel_pct": rel_pct, "n": len(pairs)}

# ── Run 0: base calculation to obtain the frozen contraction ──────────────────

exp = Exponent_Set.from_file(exp_path)

M0 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "BASE_RUN",
    full_logging=True,
    overwrite_existing=True,
)

M0.add_job(exp.copy(no_energy=True), template_dir)
M0.run_all_jobs(1)

base_job = M0.jobs[0]
if base_job.status != Job_Status.COMPLETED:
    raise RuntimeError("Base job failed — cannot obtain frozen contraction.")
if base_job.exponent_set.resulting_contraction is None:
    raise RuntimeError("Base job produced no ANO contraction — check your template.")

base_contraction    = base_job.exponent_set.resulting_contraction
base_energy_uncontr = base_job.exponent_set.energy
print(f"Base uncontracted energy: {base_energy_uncontr}")

# ── Sample random points ──────────────────────────────────────────────────────

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

# ── Run 2: self-contracted — each point uses its own uncontracted contraction ──

M2 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_SELF_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

r2_to_r1 = []

for job1 in M1.jobs:
    if job1.status != Job_Status.COMPLETED:
        continue
    if job1.exponent_set.resulting_contraction is None:
        print(f"[Info] M1 job {job1.job_id}: no ANO contraction, skipping.")
        continue
    exp_set = job1.exponent_set
    M2.add_job(Exponent_Set(
        atom_name=exp_set.atom_name,
        exponents=[e.copy() for e in exp_set.exponents],
        contractions=[c.copy() for c in exp_set.resulting_contraction],
        method=exp_set.method,
    ), template_dir)
    r2_to_r1.append(job1)

M2.run_all_jobs(MAX_JOBS)

# ── Run 3: frozen-contraction — all points use the base contraction ───────────

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
    M3.add_job(Exponent_Set(
        atom_name=exp_varied.atom_name,
        exponents=[e.copy() for e in exp_varied.exponents],
        contractions=[c.copy() for c in base_contraction],
        method=exp_varied.method,
    ), template_dir)

M3.run_all_jobs(MAX_JOBS)

# ── Run 4: projected-contraction — base contraction projected to each point ───
# For each sample point B, Löwdin-reorthogonalize base_contraction (computed at
# the base exponents) into the primitive basis of B via the cross-overlap S_AB.

M4 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "CLOUD_PROJ_CONTR",
    full_logging=True,
    overwrite_existing=True,
)

print("\nProjecting base contraction to each sample point...")
for i in range(M_SAMPLES):
    exp_varied  = build_exp_set_from_flat(exp, sampled_flat[i])
    proj_contr  = project_contraction(base_contraction, exp, exp_varied)
    M4.add_job(Exponent_Set(
        atom_name=exp_varied.atom_name,
        exponents=[e.copy() for e in exp_varied.exponents],
        contractions=[c.copy() for c in proj_contr],
        method=exp_varied.method,
    ), template_dir)

print("Projections done.")
M4.run_all_jobs(MAX_JOBS)

# ── Collect results ───────────────────────────────────────────────────────────

job2_by_id = {job1.job_id: job2 for job1, job2 in zip(r2_to_r1, M2.jobs)}

samples        = []
pairs_u_self   = []
pairs_u_fixed  = []
pairs_u_proj   = []

for job1, job3, job4 in zip(M1.jobs, M3.jobs, M4.jobs):
    eu = job1.exponent_set.energy if job1.status == Job_Status.COMPLETED else None

    job2 = job2_by_id.get(job1.job_id)
    es   = job2.exponent_set.energy if job2 is not None and job2.status == Job_Status.COMPLETED else None

    ef   = job3.exponent_set.energy if job3.status == Job_Status.COMPLETED else None
    ep   = job4.exponent_set.energy if job4.status == Job_Status.COMPLETED else None

    contr = job1.exponent_set.resulting_contraction if job1.status == Job_Status.COMPLETED else None
    n_contr_per_shell = [mat.shape[0] for mat in contr] if contr is not None else None
    n_contr_total     = sum(n_contr_per_shell) if n_contr_per_shell is not None else None

    samples.append({
        "i":               job1.job_id,
        "exps":            sampled_flat[job1.job_id].tolist(),
        "e_uncontr":       eu,
        "e_self_contr":    es,
        "e_fixed_contr":   ef,
        "e_proj_contr":    ep,
        "n_contr_shells":  n_contr_per_shell,
        "n_contr_total":   n_contr_total,
    })

    if eu is not None and es is not None:
        pairs_u_self.append((eu, es))
    if eu is not None and ef is not None:
        pairs_u_fixed.append((eu, ef))
    if eu is not None and ep is not None:
        pairs_u_proj.append((eu, ep))

rho_us,  pval_us,  agree_us,  delta_us  = rank_stats(pairs_u_self)
rho_uf,  pval_uf,  agree_uf,  delta_uf  = rank_stats(pairs_u_fixed)
rho_up,  pval_up,  agree_up,  delta_up  = rank_stats(pairs_u_proj)

top_us  = top_pairs(pairs_u_self)
top_uf  = top_pairs(pairs_u_fixed)
top_up  = top_pairs(pairs_u_proj)
rho_us_top, pval_us_top, agree_us_top, delta_us_top = rank_stats(top_us)
rho_uf_top, pval_uf_top, agree_uf_top, delta_uf_top = rank_stats(top_uf)
rho_up_top, pval_up_top, agree_up_top, delta_up_top = rank_stats(top_up)

loss_us = contraction_loss(pairs_u_self,  delta_us)
loss_uf = contraction_loss(pairs_u_fixed, delta_uf)
loss_up = contraction_loss(pairs_u_proj,  delta_up)

# ── Print results ─────────────────────────────────────────────────────────────

W = 68
summary_lines = []

def emit(line=""):
    print(line)
    summary_lines.append(line)

def print_pair_stats(label_a, label_b, full_stats, top_stats):
    rho,     pval,     agree,     delta,     n     = full_stats
    rho_top, pval_top, agree_top, delta_top, n_top = top_stats
    emit(f"\n  {label_a}  vs  {label_b}")
    for tag, (r, p, ag, dl, cnt) in [
        (f"full — {n} pairs",                   (rho,     pval,     agree,     delta,     n)),
        (f"top {TOP_FRAC:.0%} — {n_top} pairs", (rho_top, pval_top, agree_top, delta_top, n_top)),
    ]:
        emit(f"\n    [{tag}]")
        emit("  " + "─" * (W - 2))
        if r is None:
            emit("    Not enough valid pairs.")
            continue
        emit(f"    Spearman ρ     {r:>+.10f}   p = {p:.4e}")
        emit(f"    Rank agreement {ag:.1%}  ({int(ag*cnt)}/{cnt} identical)")
        emit(f"    ΔE  mean       {dl.mean():>+.8f}  Hartree")
        emit(f"    ΔE  std        {dl.std():>.8f}  Hartree")
        emit(f"    ΔE  min / max  {dl.min():>+.8f} / {dl.max():>+.8f}")

def print_loss_stats(label, loss):
    emit(f"\n  Energy loss from contraction — {label}")
    emit("─" * W)
    if loss is None:
        emit("  No valid pairs.")
        return
    d, r = loss["delta"], loss["rel_pct"]
    emit(f"  ΔE  mean         {d.mean():>+.10f}  Hartree")
    emit(f"  ΔE  std          {d.std():>.10f}  Hartree")
    emit(f"  ΔE  min / max    {d.min():>+.10f} / {d.max():>+.10f}  Hartree")
    emit(f"  Rel mean         {r.mean():>+.8f}  %")
    emit(f"  Rel std          {r.std():>.8f}  %")
    emit(f"  Rel min / max    {r.min():>+.8f} / {r.max():>+.8f}  %")

emit("\n" + "═" * W)
emit(f"{'RESULTS':^{W}}")
emit("═" * W)
emit(f"  Total samples : {M_SAMPLES}")
emit(f"  Base energy   : {base_energy_uncontr}")

print_pair_stats("Uncontr", "Self-contr ",
    (rho_us, pval_us, agree_us, delta_us, len(pairs_u_self)),
    (rho_us_top, pval_us_top, agree_us_top, delta_us_top, len(top_us)))
print_pair_stats("Uncontr", "Fixed-contr",
    (rho_uf, pval_uf, agree_uf, delta_uf, len(pairs_u_fixed)),
    (rho_uf_top, pval_uf_top, agree_uf_top, delta_uf_top, len(top_uf)))
print_pair_stats("Uncontr", "Proj-contr ",
    (rho_up, pval_up, agree_up, delta_up, len(pairs_u_proj)),
    (rho_up_top, pval_up_top, agree_up_top, delta_up_top, len(top_up)))

print_loss_stats("self-contr ", loss_us)
print_loss_stats("fixed-contr", loss_uf)
print_loss_stats("proj-contr ", loss_up)

emit("\n" + "─" * W)
emit(f"  {'#':>4}  {'Uncontr':>16}  {'Self-contr':>16}  {'Fixed-contr':>16}  {'Proj-contr':>16}")
emit("─" * W)
for s in samples:
    eu = f"{s['e_uncontr']:.10f}"    if s["e_uncontr"]     is not None else "   FAILED   "
    es = f"{s['e_self_contr']:.10f}" if s["e_self_contr"]  is not None else "   FAILED   "
    ef = f"{s['e_fixed_contr']:.10f}"if s["e_fixed_contr"] is not None else "   FAILED   "
    ep = f"{s['e_proj_contr']:.10f}" if s["e_proj_contr"]  is not None else "   FAILED   "
    emit(f"  {s['i']:>4}  {eu:>16}  {es:>16}  {ef:>16}  {ep:>16}")

emit("═" * W)

# ── Save ──────────────────────────────────────────────────────────────────────

def stats_dict(rho, pval, agree, delta, n):
    return {
        "n_valid":        n,
        "spearman_rho":   float(rho)          if rho   is not None else None,
        "spearman_pval":  float(pval)         if pval  is not None else None,
        "rank_agreement": agree               if agree is not None else None,
        "delta_mean":     float(delta.mean()) if delta is not None else None,
        "delta_std":      float(delta.std())  if delta is not None else None,
    }

def loss_dict(loss):
    if loss is None:
        return None
    d, r = loss["delta"], loss["rel_pct"]
    return {
        "n_valid":      loss["n"],
        "delta_mean":   float(d.mean()),
        "delta_std":    float(d.std()),
        "delta_min":    float(d.min()),
        "delta_max":    float(d.max()),
        "rel_mean_pct": float(r.mean()),
        "rel_std_pct":  float(r.std()),
        "rel_min_pct":  float(r.min()),
        "rel_max_pct":  float(r.max()),
    }

out = {
    "meta": {
        "n_samples":           M_SAMPLES,
        "max_frac":            MAX_FRAC,
        "seed":                SEED,
        "n_exponents":         n_exponents,
        "base_energy_uncontr": base_energy_uncontr,
    },
    "stats": {
        "uncontr_vs_self_contr": {
            "full":    stats_dict(rho_us,     pval_us,     agree_us,     delta_us,     len(pairs_u_self)),
            "top_pct": stats_dict(rho_us_top, pval_us_top, agree_us_top, delta_us_top, len(top_us)),
        },
        "uncontr_vs_fixed_contr": {
            "full":    stats_dict(rho_uf,     pval_uf,     agree_uf,     delta_uf,     len(pairs_u_fixed)),
            "top_pct": stats_dict(rho_uf_top, pval_uf_top, agree_uf_top, delta_uf_top, len(top_uf)),
        },
        "uncontr_vs_proj_contr": {
            "full":    stats_dict(rho_up,     pval_up,     agree_up,     delta_up,     len(pairs_u_proj)),
            "top_pct": stats_dict(rho_up_top, pval_up_top, agree_up_top, delta_up_top, len(top_up)),
        },
    },
    "contraction_loss": {
        "self_contr":  loss_dict(loss_us),
        "fixed_contr": loss_dict(loss_uf),
        "proj_contr":  loss_dict(loss_up),
    },
    "samples":  samples,
    "summary":  "\n".join(summary_lines),
}

out_path = SUBMIT_DIR / "four_way_rank_test.json"
log_path = SUBMIT_DIR / "four_way_rank_test.log"

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

with open(log_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nSaved to {out_path}")
print(f"Log    to {log_path}")
