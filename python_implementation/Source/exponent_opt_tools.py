from pathlib import Path
from exponent_handler import *
from molcas_handler import *
from job_manager import *



def local_exponent_removal_analysis(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_path: Path | str,
    *,
    print_results: bool = False,
    threads: int        = 1
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
    job_manager.run_all_jobs(threads, 0.1)
    ref_energy = job_manager.jobs[0].exponent_set.energy

    results = []
    best_job = None
    best_delta_E = float('inf')

    for job in job_manager.jobs[1:]:
        l, q = map(int, job.expo_name.replace("Removed_", "").split("_"))
        delta_E = job.exponent_set.energy - ref_energy
        results.append({"l": l, "q": q, "delta_E": delta_E})

        # Track the job with lowest delta_E
        if delta_E < best_delta_E:
            best_delta_E = delta_E
            best_job = job

    if print_results:
        print("\nLocal Exponent Removal Analysis:")
        for r in results:
            print(f"l={r['l']} q={r['q']}  ΔE = {r['delta_E']:.6e}")

        print("\nLowest energy removal:")
        print(f"l={results[results.index(min(results, key=lambda x: x['delta_E']))]['l']} "
              f"q={results[results.index(min(results, key=lambda x: x['delta_E']))]['q']}  "
              f"ΔE = {best_delta_E:.6e}")

    # Return both the results and the exponent_set with lowest ΔE
    return [results, best_job.exponent_set]

def local_exponent_removal_suggestion(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_path: Path | str,
    *,
    threads: int = 1
):

    results, best_exp_set = local_exponent_removal_analysis( exponent_set, job_manager, template_path, threads=threads)

    # Find the l, q of the lowest ΔE
    best_result = min(results, key=lambda x: x['delta_E'])
    best_index  = [best_result['l'], best_result['q']]

    return best_exp_set, best_index



def local_exponent_removal_analysis_mixed(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_paths: list[Path | str],
    *,
    print_results: bool = False,
    threads: int        = 1
):

    # Reference job
    ref_exp = exponent_set.copy_without_energy()
    job_manager.add_job(ref_exp, template_paths[0], name="reference_g")
    job_manager.add_job(ref_exp, template_paths[1], name="reference_a")
    job_manager.add_job(ref_exp, template_paths[2], name="reference_c")

    # Removal jobs
    for l in range(len(exponent_set.exponents)):
        for q in range(exponent_set.lengths[l]):
            for c in range(3):
                exp_copy = exponent_set.copy_without_energy()
                exp_copy.remove_exponent_uncontracted(l, q)
                job_manager.add_job(exp_copy, template_paths[c], name=f"Removed_{l}_{q}_{c}")

    # Run jobs
    job_manager.run_all_jobs(threads, 0.1)
    ref_energy = (job_manager.jobs[0].exponent_set.energy + job_manager.jobs[1].exponent_set.energy + job_manager.jobs[2].exponent_set.energy)/3

    results = []
    best_job = None
    best_delta_E = float('inf')

    for i in range(len(exponent_set.exponents)):
        job_g = job_manager.jobs[3*(i+1)]
        job_a = job_manager.jobs[3*(i+1)+1]
        job_c = job_manager.jobs[3*(i+1)+2]

        l, q, c  = map(int, job_g.expo_name.replace("Removed_", "").split("_"))
        new_E = (job_g.exponent_set.energy + job_a.exponent_set.energy + job_c.exponent_set.energy)/3 
        delta_E = new_E-ref_energy
        results.append({"l": l, "q": q, "delta_E": delta_E})

        if delta_E < best_delta_E:
            best_delta_E = delta_E
            job_g.exponent_set.energy = new_E
            best_job = job_g

    if print_results:
        print("\nLocal Exponent Removal Analysis:")
        for r in results:
            print(f"l={r['l']} q={r['q']}  ΔE = {r['delta_E']:.6e}")

        print("\nLowest energy removal:")
        print(f"l={results[results.index(min(results, key=lambda x: x['delta_E']))]['l']} "
              f"q={results[results.index(min(results, key=lambda x: x['delta_E']))]['q']}  "
              f"ΔE = {best_delta_E:.6e}")

    # Return both the results and the exponent_set with lowest ΔE
    return [results, best_job.exponent_set]




def local_exponent_removal_suggestion_mixed(
    exponent_set: Exponent_Set,
    job_manager: Job_Manager,
    template_path: Path | str,
    *,
    threads: int = 1
):

    results, best_exp_set = local_exponent_removal_analysis_mixed( exponent_set, job_manager, template_path, threads=threads)

    # Find the l, q of the lowest ΔE
    best_result = min(results, key=lambda x: x['delta_E'])
    best_index  = [best_result['l'], best_result['q']]

    return best_exp_set, best_index












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