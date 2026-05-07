import sys
import json
import numpy as np
from pathlib import Path

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

M_ITER = 10   # number of contraction feedback iterations

exp = Exponent_Set.from_file(exp_path)

# ── Iteration 0: uncontracted base calculation ────────────────────────────────

M0 = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "ITER_00",
    full_logging=True,
    overwrite_existing=True,
)

M0.add_job(exp.copy(no_energy=True), template_dir)
M0.run_all_jobs(1)

base_job = M0.jobs[0]
if base_job.status != Job_Status.COMPLETED:
    raise RuntimeError("Iteration 0 (uncontracted) failed.")
if base_job.exponent_set.resulting_contraction is None:
    raise RuntimeError("Iteration 0 produced no ANO contraction — check your template.")

base_energy      = base_job.exponent_set.energy
base_contraction = base_job.exponent_set.resulting_contraction

print(f"Iter  0  (uncontracted)  energy = {base_energy:.10f}")

iterations = [{
    "iter":    0,
    "type":    "uncontracted",
    "energy":  base_energy,
    "delta_e": None,
    "contr_frob_change": None,
}]

current_contraction = base_contraction
prev_energy         = base_energy

# ── Iterations 1..M: feed contraction back in ────────────────────────────────

for it in range(1, M_ITER + 1):

    Mi = Job_Manager(
        Executor_Type.LOCAL_BASH,
        submit_scr,
        extract_scr,
        group_dir_path=WORK_DIR / f"ITER_{it:02d}",
        full_logging=True,
        overwrite_existing=True,
    )

    contracted_exp = Exponent_Set(
        atom_name=exp.atom_name,
        exponents=[e.copy() for e in exp.exponents],
        contractions=[c.copy() for c in current_contraction],
        method=exp.method,
    )
    Mi.add_job(contracted_exp, template_dir)
    Mi.run_all_jobs(1)

    job = Mi.jobs[0]

    if job.status != Job_Status.COMPLETED:
        print(f"Iter {it:>2}  FAILED — stopping.")
        break
    if job.exponent_set.resulting_contraction is None:
        print(f"Iter {it:>2}  no ANO contraction produced — stopping.")
        break

    energy          = job.exponent_set.energy
    new_contraction = job.exponent_set.resulting_contraction

    delta_e      = energy - prev_energy
    frob_change  = sum(
        float(np.linalg.norm(nc - oc, "fro"))
        for nc, oc in zip(new_contraction, current_contraction)
    )

    print(f"Iter {it:>2}  (contracted)     energy = {energy:.10f}   "
          f"ΔE = {delta_e:>+.10f}   ΔC = {frob_change:.6e}")

    iterations.append({
        "iter":              it,
        "type":              "contracted",
        "energy":            energy,
        "delta_e":           delta_e,
        "contr_frob_change": frob_change,
    })

    current_contraction = new_contraction
    prev_energy         = energy

# ── Summary ───────────────────────────────────────────────────────────────────

n_done = len(iterations)
W = 72

summary_lines = []
def emit(line=""):
    print(line)
    summary_lines.append(line)

emit("\n" + "═" * W)
emit(f"{'SELF-CONSISTENCY RESULTS':^{W}}")
emit("═" * W)
emit(f"  Starting energy (uncontracted) : {base_energy:.10f}  Hartree")
emit(f"  Iterations completed           : {n_done - 1} / {M_ITER}")

if n_done > 1:
    final = iterations[-1]
    emit(f"  Final energy                   : {final['energy']:.10f}  Hartree")
    emit(f"  Total energy change            : {final['energy'] - base_energy:>+.10f}  Hartree")

emit("\n" + "─" * W)
emit(f"  {'Iter':>4}  {'Type':>14}  {'Energy':>20}  {'ΔE':>18}  {'ΔC (Frob)':>12}")
emit("─" * W)

for rec in iterations:
    e_str  = f"{rec['energy']:.10f}"
    de_str = f"{rec['delta_e']:>+.10f}" if rec["delta_e"] is not None else "       —      "
    dc_str = f"{rec['contr_frob_change']:.6e}"  if rec["contr_frob_change"] is not None else "     —    "
    emit(f"  {rec['iter']:>4}  {rec['type']:>14}  {e_str:>20}  {de_str:>18}  {dc_str:>12}")

emit("═" * W)

# ── Save ──────────────────────────────────────────────────────────────────────

out = {
    "meta": {
        "m_iter":       M_ITER,
        "n_completed":  n_done - 1,
        "base_energy":  base_energy,
    },
    "iterations": iterations,
    "summary":    "\n".join(summary_lines),
}

out_path  = SUBMIT_DIR / "self_consistency_test.json"
log_path  = SUBMIT_DIR / "self_consistency_test.log"

with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

with open(log_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nSaved to {out_path}")
print(f"Log    to {log_path}")
