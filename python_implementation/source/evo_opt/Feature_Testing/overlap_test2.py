import sys
import numpy as np
from pathlib import Path

WORK_DIR   = Path(sys.argv[1])
SUBMIT_DIR = Path(sys.argv[2])
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.molcas_handler import *
from evo_opt.job_manager import *
from evo_opt.common import Executor_Type
from evo_opt.shell_overlap import primitive_shell_overlaps

exp_path     = SUBMIT_DIR / "exp.expo"
template_dir = SUBMIT_DIR / "template.inp"
submit_scr   = SUBMIT_DIR / "run.sh"
extract_scr  = SUBMIT_DIR / "extract.sh"

exp = Exponent_Set.from_file(exp_path)

# ── Run uncontracted MOLCAS calculation to get the ANO contraction ────────────

M = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path=WORK_DIR / "OVERLAP_TEST2",
    full_logging=True,
    overwrite_existing=True,
)

M.add_job(exp.copy(no_energy=True), template_dir)
M.run_all_jobs(1)

job = M.jobs[0]
if job.status != Job_Status.COMPLETED:
    raise RuntimeError("MOLCAS job failed.")
if job.exponent_set.resulting_contraction is None:
    raise RuntimeError("No ANO contraction produced — check your template.")

C_shells = job.exponent_set.resulting_contraction   # list of (n_contracted, n_prim) per shell
energy   = job.exponent_set.energy
print(f"Job completed.  Energy: {energy:.10f}")

# ── Compute primitive self-overlap S (A = B, same exponent set) ───────────────

print("\nComputing primitive self-overlaps via PySCF...")
S_shells = primitive_shell_overlaps(exp, exp)
print("Done.")

# ── Check C @ S @ C.T = I per shell ──────────────────────────────────────────

W = 68
out_lines = []

def emit(line=""):
    print(line)
    out_lines.append(line)

emit("\n" + "═" * W)
emit(f"{'C @ S @ C.T  =?=  I   (orthonormality check)':^{W}}")
emit("═" * W)
emit(f"  Energy : {energy:.10f}  Hartree")
emit(f"  Shells : {len(exp.exponents)}")

for l, (C, S) in enumerate(zip(C_shells, S_shells)):
    emit(f"\n  Shell {l}  (l={l})")
    emit("  " + "─" * (W - 2))

    if S is None:
        emit("  Overlap not computed for this shell.")
        continue

    n_contracted, n_prim = C.shape
    emit(f"  Primitives : {n_prim}   Contracted : {n_contracted}")

    CSCt = C @ S @ C.T   # should be identity (n_contracted x n_contracted)
    I    = np.eye(n_contracted)
    diff = CSCt - I

    emit(f"  max |C S Cᵀ - I| : {np.abs(diff).max():.6e}")
    emit(f"  Frobenius norm    : {np.linalg.norm(diff):.6e}")
    emit(f"\n  C @ S @ Cᵀ :")
    for row in CSCt:
        emit("    " + "  ".join(f"{v:>+.8f}" for v in row))

emit("\n" + "═" * W)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = SUBMIT_DIR / "overlap_test2.txt"
with open(out_path, "w") as f:
    f.write("\n".join(out_lines) + "\n")

print(f"\nSaved to {out_path}")
