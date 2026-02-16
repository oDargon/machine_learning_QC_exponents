from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *



def local_exponent_removal_analysis(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_path : Path | str,
    *,
    print_results: bool = False
):
    
    # Reference job
    ref_exp = exponent_set.copy_without_energy()
    job_manager.add_job(ref_exp, template_path, name="reference")

    # Removal jobs
    for l in range(len(exponent_set.exponents)):
        for q in range(exponent_set.lengths[l]):
            exp_copy = exponent_set.copy_without_energy()
            exp_copy.remove_exponent_uncontracted(l, q)
            job_manager.add_job(exp_copy, template_path, name=f"Removed_{l}_{q}")

    # Run jobs
    job_manager.run_all_jobs(1, 0.1)
    ref_energy = job_manager.jobs[0].exponent_set.energy

    results = []
    for job in job_manager.jobs[1:]:
        l, q = map(int, job.expo_name.replace("Removed_", "").split("_"))
        delta_E = job.exponent_set.energy - ref_energy
        results.append({"l": l, "q": q, "delta_E": delta_E})

    if print_results:
        print("\nLocal Exponent Removal Analysis:")
        for r in results:
            print(f"l={r['l']} q={r['q']}  ΔE = {r['delta_E']:.6e}")

        best = min(results, key=lambda x: x["delta_E"])
        print("\nLowest energy removal:")
        print(f"l={best['l']} q={best['q']}  ΔE = {best['delta_E']:.6e}")

    return results



def local_exponent_sensitivity_analysis(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_path : Path | str,
    x:float=0.05,
    *,
    print_results:bool=True
):
    # Reference
    ref_exp = exponent_set.copy_without_energy()
    job_manager.add_job(ref_exp, template_path, name="reference")

    for l in range(len(exponent_set.exponents)):
        for q in range(exponent_set.lengths[l]):
            alpha = exponent_set.exponents[l][q]
            h     = x * alpha

            # Minus
            exp_minus = exponent_set.copy_without_energy()
            exp_minus.exponents[l][q] = alpha - h
            job_manager.add_job(exp_minus, template_path, name=f"Minus_{l}_{q}")

            # Plus
            exp_plus = exponent_set.copy_without_energy()
            exp_plus.exponents[l][q] = alpha + h
            job_manager.add_job(exp_plus, template_path, name=f"Plus_{l}_{q}")

    # Run jobs
    job_manager.run_all_jobs(1, 0.1)

    ref_energy = job_manager.jobs[0].exponent_set.energy
    job_dict = {job.expo_name: job for job in job_manager.jobs}

    results = []
    for l in range(len(exponent_set.exponents)):
        for q in range(exponent_set.lengths[l]):
            alpha = exponent_set.exponents[l][q]
            h     = x * alpha

            E0      = ref_energy
            E_minus = job_dict[f"Minus_{l}_{q}"].exponent_set.energy
            E_plus  = job_dict[f"Plus_{l}_{q}"].exponent_set.energy

            first_deriv  = (E_plus - E_minus) / (2 * h)
            second_deriv = (E_plus - 2*E0 + E_minus) / (h ** 2)

            results.append({
                "l": l,
                "q": q,
                "first_derivative": first_deriv,
                "second_derivative": second_deriv
            })

    if print_results:
        print("\nLocal Exponent Sensitivity Analysis:")
        for r in results:
            print(
                f"l={r['l']} q={r['q']}  "
                f"dE/dα = {r['first_derivative']:.6e}  "
                f"d²E/dα² = {r['second_derivative']:.6e}"
            )

    return results