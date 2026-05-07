from typing import List, Optional
from numpy import ndarray, array
from pyscf import gto

from .exponent_handler import Exponent_Set


def primitive_shell_overlaps(
    exp_a: Exponent_Set,
    exp_b: Exponent_Set,
    shells: Optional[List[int]] = None,
) -> List[Optional[ndarray]]:
    """
    Compute per-shell primitive GTO overlap matrices between two Exponent_Sets.

    Both sets must have the same number of shells, with shell index mapping
    directly to angular momentum (shell 0 = s, 1 = p, 2 = d, ...).
    The two sets may have different numbers of primitives per shell.

    Returns a list of length n_shells. Each entry is either:
      - ndarray of shape (n_prim_a, n_prim_b)  for a computed shell
      - None                                    for shells not requested

    Parameters
    ----------
    exp_a, exp_b : Exponent_Set
        The two basis sets. Must have the same number of shells.
    shells : list of int, optional
        Shell indices to compute. Defaults to all shells.
    """
    if len(exp_a.exponents) != len(exp_b.exponents):
        raise ValueError(
            f"Exponent sets must have the same number of shells "
            f"({len(exp_a.exponents)} vs {len(exp_b.exponents)})"
        )

    n_shells = len(exp_a.exponents)
    if shells is None:
        shells = list(range(n_shells))

    results: List[Optional[ndarray]] = [None] * n_shells

    for l in shells:
        alphas_a = exp_a.exponents[l]
        alphas_b = exp_b.exponents[l]
        n_a = len(alphas_a)
        n_b = len(alphas_b)

        if n_a == 0 or n_b == 0:
            results[l] = array([]).reshape(n_a, n_b)
            continue

        all_alphas = list(alphas_a) + list(alphas_b)
        n_total    = n_a + n_b
        n_comp     = 2 * l + 1   # spherical components per primitive

        # Build one shell with all primitives uncontracted (identity contraction).
        # PySCF shell format: [l, [alpha1, c_1, c_2, ...], [alpha2, ...], ...]
        # where c_k is the coefficient of primitive i in contracted function k.
        shell_def = [l]
        for i, alpha in enumerate(all_alphas):
            coeffs    = [0.0] * n_total
            coeffs[i] = 1.0
            shell_def.append([float(alpha)] + coeffs)

        mol = gto.Mole()
        mol.atom   = [['He', (0, 0, 0)]]
        mol.basis  = {'He': [shell_def]}
        mol.unit   = 'Bohr'
        mol.cart   = False   # spherical GTOs
        mol.verbose = 0
        mol.build()

        S = mol.intor('int1e_ovlp')  # shape (n_total*n_comp, n_total*n_comp)

        # For same-center same-l, each (i,j) primitive pair produces an
        # (n_comp x n_comp) diagonal block s_ij * I.  Striding with step n_comp
        # picks one representative value per pair.
        n_func_a  = n_a * n_comp
        S_cross   = S[:n_func_a, n_func_a:]          # (n_a*n_comp, n_b*n_comp)
        results[l] = S_cross[::n_comp, ::n_comp]     # (n_a, n_b)

    return results


def project_contraction(
    C_shells: List[ndarray],
    exp_a: Exponent_Set,
    exp_b: Exponent_Set,
    shells: Optional[List[int]] = None,
) -> List[ndarray]:
    """
    Project a contraction from point A to point B on the exponent manifold.

    For each shell l, computes D = C @ S_AB where S_AB is the primitive
    cross-overlap (n_prim_A, n_prim_B).  Reorthogonalization is left to
    the calling program (e.g. MOLCAS handles non-orthonormal contracted
    bases via the generalized eigenvalue problem).

    Parameters
    ----------
    C_shells : list of ndarray, shape (n_contracted, n_prim_A) per shell,
        indexed by shell index l (same length as exp_a.exponents).
    exp_a : Exponent_Set   — point A (where C_shells was computed)
    exp_b : Exponent_Set   — point B (target)
    shells : list of int, optional
        Shell indices to project. Defaults to all shells.

    Returns
    -------
    List of ndarray, shape (n_contracted, n_prim_B) per shell,
    indexed by shell index l (same convention as C_shells).
    """
    n_shells = len(exp_a.exponents)
    if len(exp_b.exponents) != n_shells:
        raise ValueError("exp_a and exp_b must have the same number of shells")
    if len(C_shells) != n_shells:
        raise ValueError(
            f"C_shells has {len(C_shells)} entries but exp_a has {n_shells} shells — "
            "C_shells must be indexed by shell index l"
        )

    if shells is None:
        shells = list(range(n_shells))

    S_AB_list = primitive_shell_overlaps(exp_a, exp_b, shells=shells)

    projected: List[ndarray] = [None] * n_shells

    for l in shells:
        projected[l] = C_shells[l] @ S_AB_list[l]   # (n_contracted, n_prim_B)

    return projected
