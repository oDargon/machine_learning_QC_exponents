from argparse import ArgumentParser
from math import exp
from pathlib import Path

from source.evo_opt.exponent_handler import Exponent_Set
from source.evo_opt.opt_tools_new import exponent_difference_metrics


def format_shape(lengths: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in lengths) + "]"


def main() -> None:
    parser = ArgumentParser(prog="expcomp")
    parser.add_argument("exp1", type=Path, help="Path to first exponent set")
    parser.add_argument("exp2", type=Path, help="Path to second exponent set")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-shell metrics as well",
    )

    args = parser.parse_args()

    exp1 = Exponent_Set.load(args.exp1)
    exp2 = Exponent_Set.load(args.exp2)

    if not exp1.same_shape_as(exp2):
        raise ValueError("Exponent sets must have the same shape to compare.")

    total_rms, per_shell_rms, max_global, per_shell_max = exponent_difference_metrics(exp1, exp2)

    print(f"Shape: {format_shape(exp1.lengths)}")
    print(
        f"Total x-change: x{exp(total_rms):.6f}    "
        f"Max global x-change: x{exp(max_global):.6f}"
    )

    if args.verbose:
        print("Per-shell differences:")
        for l, (rms, mx) in enumerate(zip(per_shell_rms, per_shell_max)):
            print(
                f"  l={l:<2} "
                f"RMS x{exp(float(rms)):>9.6f}   "
                f"MAX x{exp(float(mx)):>9.6f}"
            )