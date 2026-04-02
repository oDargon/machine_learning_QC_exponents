from pathlib import Path
from .exponent_handler import Exponent_Set
from .objectives import Objective
from numpy import argmax, argmin, array, diag, float64,ndarray, log, mean, sqrt, empty
from numpy.linalg import eigh
from typing import List, Dict



def local_exponent_removal_analysis(in_exp: Exponent_Set, in_energy: float64, objective: Objective, work_dir: Path | str,  * ,print_results: bool = False, threads: int = 1, overwrite: bool = False):

    exponents = []
    labels    = []

    i = 0
    for l in range(len(in_exp.exponents)):
        for q in range(len(in_exp.exponents[l])):
            new_exp = in_exp.copy_without_energy() 
            exponents.append(new_exp)
            exponents[i].remove_exponent_uncontracted( l, q ) 
            labels.append( (l, q) )
            i += 1

    energies = objective.evaluate_batch(
        exponents,
        work_dir=work_dir,
        names=[ f"Removed_{l}_{q}" for l, q in labels ],
        threads=threads,
        overwrite=overwrite
    ) #This is technicaly the 'objective values' which need not be the energy.

    deltas   = energies - in_energy
    best_idx = argmin(deltas)

    if print_results:
        print("\nLocal Exponent Removal Analysis")
        print("-" * 40)

        for (l, q), dE in zip(labels, deltas):
            print(f"l={l:2d}  q={q:2d}   ΔE = {dE: .6e}")

        l_best, q_best = labels[best_idx]
        print("\nLowest energy(objective) removal")
        print("-" * 40)
        print(f"l={l_best}  q={q_best}   ΔE = {deltas[best_idx]: .6e}")

    return energies[best_idx], exponents[best_idx].copy_without_energy(), best_idx, labels[best_idx]

def least_important_indices(cov: ndarray):

    cov = array(cov)

    # --- Method 1: diagonal variance ---
    variances      = diag(cov)
    variance_index = argmax(variances)

    # --- Method 2: eigen decomposition ---
    eigvals, eigvecs = eigh(cov)

    idx_max          = argmax(eigvals)
    largest_eigval   = eigvals[idx_max]
    largest_eigvec   = eigvecs[:, idx_max]

    # parameter most aligned with flattest direction
    eigen_index = argmax(abs(largest_eigvec))

    return



def flat_log_diff(exp1: Exponent_Set, exp2: Exponent_Set):
    if not exp1.same_shape_as(exp2):
        raise ValueError("Exponent sets must have the same shape to compare.")

    flat1 = exp1.flatten_exps()
    flat2 = exp2.flatten_exps()

    return log(flat1 / flat2)

def total_log_difference(exp1: Exponent_Set, exp2: Exponent_Set) -> float:
    diff = flat_log_diff(exp1, exp2)
    return float(sqrt(mean(diff * diff)))

def max_log_difference(exp1: Exponent_Set, exp2: Exponent_Set) -> float:
    diff = flat_log_diff(exp1, exp2)
    return float(max(abs(diff)))

def per_shell_log_difference(exp1: Exponent_Set, exp2: Exponent_Set):
    if not exp1.same_shape_as(exp2):
        raise ValueError("Exponent sets must have the same shape to compare.")

    n_shells = len(exp1.exponents)
    result = empty(n_shells, dtype=float64)

    for l in range(n_shells):
        shell1 = exp1.exponents[l]
        shell2 = exp2.exponents[l]

        if len(shell1) == 0:
            result[l] = 0.0
            continue

        diff = log(shell1 / shell2)
        result[l] = sqrt(mean(diff * diff))

    return result

def per_shell_max_log_difference(exp1: Exponent_Set, exp2: Exponent_Set):
    if not exp1.same_shape_as(exp2):
        raise ValueError("Exponent sets must have the same shape to compare.")

    n_shells = len(exp1.exponents)
    result = empty(n_shells, dtype=float64)

    for l in range(n_shells):
        shell1 = exp1.exponents[l]
        shell2 = exp2.exponents[l]

        if len(shell1) == 0:
            result[l] = 0.0
            continue

        diff = log(shell1 / shell2)
        result[l] = max(abs(diff))

    return result

def exponent_difference_metrics(exp1: Exponent_Set, exp2: Exponent_Set):
    if not exp1.same_shape_as(exp2):
        raise ValueError("Exponent sets must have the same shape to compare.")

    # ---- flat metrics ----
    flat1 = exp1.flatten_exps()
    flat2 = exp2.flatten_exps()

    if len(flat1) == 0:
        total_rms = 0.0
        max_global = 0.0
    else:
        diff = log(flat1 / flat2)
        total_rms = float(sqrt(mean(diff * diff)))
        max_global = float(max(abs(diff)))

    # ---- per-shell metrics ----
    n_shells = len(exp1.exponents)
    per_shell_rms = empty(n_shells, dtype=float64)
    per_shell_max = empty(n_shells, dtype=float64)

    for l in range(n_shells):
        shell1 = exp1.exponents[l]
        shell2 = exp2.exponents[l]

        if len(shell1) == 0:
            per_shell_rms[l] = 0.0
            per_shell_max[l] = 0.0
            continue

        d = log(shell1 / shell2)
        per_shell_rms[l] = sqrt(mean(d * d))
        per_shell_max[l] = max(abs(d))

    return [total_rms, per_shell_rms, max_global, per_shell_max]