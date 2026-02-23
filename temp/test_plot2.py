import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from exponent_handler import Exponent_Set

# --------------------
# Settings
# --------------------
results_dir = Path("temp/Run_results_20260216_230842")  # replace with actual path
variations_per_exponent = 11
variation_range = 0.5  # ±5%


exp = Exponent_Set.from_file(results_dir / "Be_0.expo")

Energy_results = []
variations     = np.linspace(1 - variation_range, 1 + variation_range, variations_per_exponent)

index = 0

for l in range(len(exp.exponents)):
    for q in range(exp.lengths[l]):
        energies = []
        for i in range(variations_per_exponent):
            exp_file = results_dir / f"Be_{index}.expo"
            if exp_file.exists():
                exp_set = Exponent_Set.from_file(exp_file)
                energies.append(exp_set.energy)
            else:
                print(f"Warning: {exp_file} not found.")
            index += 1
        Energy_results.append(energies)



current_index = 0  # which energy curve we're showing

fig, ax = plt.subplots()

line, = ax.plot(variations, Energy_results[current_index], marker='o')
ax.set_xlabel("Variation factor")
ax.set_ylabel("Energy")
title = ax.set_title(f"Energy variation set {current_index+1}/{len(Energy_results)}")

def update_plot():
    line.set_ydata(Energy_results[current_index])
    title.set_text(f"Energy variation set {current_index+1}/{len(Energy_results)}")
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

def on_key(event):
    global current_index
    
    if event.key == "right":
        if current_index < len(Energy_results) - 1:
            current_index += 1
            update_plot()
            
    elif event.key == "left":
        if current_index > 0:
            current_index -= 1
            update_plot()

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()