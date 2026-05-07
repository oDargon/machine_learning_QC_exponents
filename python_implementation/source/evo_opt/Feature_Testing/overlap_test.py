import numpy as np
from pathlib import Path

CWD = Path.cwd()

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.shell_overlap import primitive_shell_overlaps

exp_path = CWD / "exp.expo"

PERTURB_FRAC = 0.05   # max per-exponent perturbation (±5%)
SEED         = 42

rng = np.random.default_rng(SEED)

# ── Load base exponent set ────────────────────────────────────────────────────

exp_a = Exponent_Set.from_file(exp_path)

print(f"Loaded: {exp_path}")
print(f"Atom   : {exp_a.atom_name}")
print(f"Shells : {len(exp_a.exponents)}")
for l, exps in enumerate(exp_a.exponents):
    print(f"  Shell {l} (l={l}): {len(exps)} primitives")

# ── Build perturbed exponent set ──────────────────────────────────────────────

perturbed_exponents = [
    exps * (1.0 + rng.uniform(-PERTURB_FRAC, PERTURB_FRAC, size=exps.shape))
    for exps in exp_a.exponents
]

exp_b = Exponent_Set(
    atom_name=exp_a.atom_name,
    exponents=perturbed_exponents,
    method=exp_a.method,
)

print(f"\nPerturbed set created (±{PERTURB_FRAC*100:.0f}% per exponent, seed={SEED})")

# ── Compute overlaps ──────────────────────────────────────────────────────────

print("\nComputing primitive shell overlaps via PySCF...")
overlaps = primitive_shell_overlaps(exp_a, exp_b)
print("Done.")

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = CWD / "overlap_test.txt"
with open(out_path, "w") as f:

    f.write(f"Overlap test\n")
    f.write(f"Atom    : {exp_a.atom_name}\n")
    f.write(f"Perturb : ±{PERTURB_FRAC*100:.0f}%  seed={SEED}\n")
    f.write(f"Shells  : {len(exp_a.exponents)}\n\n")

    W = 60
    f.write("═" * W + "\n")

    for l, S in enumerate(overlaps):
        if S is None:
            continue
        f.write(f"\nShell {l}  (l={l})  shape {S.shape}\n")
        f.write("─" * W + "\n")
        f.write(f"Exponents A : {' '.join(f'{v:.8e}' for v in exp_a.exponents[l])}\n")
        f.write(f"Exponents B : {' '.join(f'{v:.8e}' for v in exp_b.exponents[l])}\n")
        f.write("\nOverlap matrix (rows = A primitives, cols = B primitives):\n")
        for row in S:
            f.write("  " + "  ".join(f"{v:>+.8f}" for v in row) + "\n")
        f.write(f"\nDiagonal (A_i · B_i) : {' '.join(f'{v:.8f}' for v in np.diag(S))}\n")
        if S.shape[0] == S.shape[1]:
            f.write(f"det : {np.linalg.det(S):>+.8e}\n")
        f.write(f"min : {S.min():>+.8f}   max : {S.max():>+.8f}\n")

    f.write("\n" + "═" * W + "\n")

print(f"Saved to {out_path}")
