from pathlib import Path
from evo_opt.exponent_handler import Exponent_Set
from evo_opt.Ploting.density_shells import plot_shell_densities

HERE = Path(__file__).resolve().parent

a = Exponent_Set.from_file(HERE / "Be_6dir.expo")
b = Exponent_Set.from_file(HERE / "Be_6dir_all.expo")
c = Exponent_Set.from_file(HERE / "Be_6dir_mix.expo")
d = Exponent_Set.from_file(HERE / "Be_6dir_m1.expo")
e = Exponent_Set.from_file(HERE / "Be_6dir_p4.expo")


plot_shell_densities(
    [a, d, e, b],
    names=["base", "m1", "p4", "all"],
)


# s1 = Exponent_Set.from_file(HERE / "Se_6dir_orig.expo")
# s2 = Exponent_Set.from_file(HERE / "Se_6dir_opt.expo")
# s3 = Exponent_Set.from_file(HERE / "Se_6dir_opt_mix.expo")

# plot_shell_densities(
#     [s1, s2, s3],
#     names=["base", "opt", "mix"],
# )
