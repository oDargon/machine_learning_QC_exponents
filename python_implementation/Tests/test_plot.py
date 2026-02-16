import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from exponent_handler import Exponent_Set

# --------------------
# Settings
# --------------------
results_dir = Path("temp/Run_results_20260215_024840")  # replace with actual path
variations_per_exponent = 101
variation_range = 0.1  # ±5%
N_exponents = 14

# --------------------
# Load energies only
# --------------------
energies = []

print(results_dir)

counter = 0
for j in range(N_exponents):
    energies.append([])  # Initialize list for this exponent
    for i in range(variations_per_exponent):
        exp_file = results_dir / f"Be_{counter}.expo"
        if exp_file.exists():
            exp_set = Exponent_Set.from_file(exp_file)
            energies[j].append(exp_set.energy)
        else:
            print(f"Warning: {exp_file} not found.")
        counter += 1

for j in range(N_exponents):
    plt.plot(
        np.linspace(1 - variation_range, 1 + variation_range, variations_per_exponent),
        energies[j],
        label=f"Exponent {j+1}"
    )
    plt.xlabel("Variation Factor")
    plt.ylabel("Energy")
    plt.title("Energy vs Exponent Variation")
    plt.legend()
    plt.show()
