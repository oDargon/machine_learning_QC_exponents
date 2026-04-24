
import argparse
import sys
from pathlib import Path
from .exponent_handler import Exponent_Set
from .parsing import make_input_from_template


def main():
    parser = argparse.ArgumentParser(
        description="Replace placeholders in a template input file using data from a .expo file."
    )
    parser.add_argument("expo_file", help="Input .expo file")
    parser.add_argument("template_file", help="Template input file")
    parser.add_argument(
        "output_name",
        nargs="?",
        default="new",
        help="Output file name (without .inp, default: new)",
    )

    args          = parser.parse_args()
    expo_path     = Path(args.expo_file)
    template_path = Path(args.template_file)

    # append .inp automatically
    output_path = Path(f"{args.output_name}.inp")

    # ---- validation ----
    if expo_path.suffix != ".expo":
        print(f"Error: '{expo_path}' is not a .expo file", file=sys.stderr)
        sys.exit(1)

    if not expo_path.exists():
        print(f"Error: '{expo_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not template_path.exists():
        print(f"Error: '{template_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    if output_path.exists():
        print(f"Error: '{output_path}' already exists", file=sys.stderr)
        sys.exit(1)

    # ---- main logic ----
    exponent_set = Exponent_Set.from_file(expo_path)
    make_input_from_template(output_path, template_path, exponent_set)

    