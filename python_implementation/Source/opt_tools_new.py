from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *
from objectives import *
from numpy import argmax, argmin, array, diag
from numpy.linalg import eigh




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
