from argparse import ArgumentParser
from pathlib import Path
from numpy import array
from .parsing import ATOMIC_NUMBERS
from .exponent_handler import Exponent_Set


def main() -> None:
    parser = ArgumentParser(prog="txt2expo")
    parser.add_argument("input_txt", type=Path, help="Path to input text file")
    parser.add_argument("-o", "--output", type=Path, help="Output .expo file")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if not args.input_txt.is_file():
        raise ValueError(f"Invalid file: {args.input_txt}")

    lines = args.input_txt.read_text(encoding="utf-8").splitlines()

    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        raise ValueError("Input file is empty")

    symbol = nonempty[0]
    symbol = symbol[0].upper() + symbol[1:].lower()

    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f"Invalid element symbol: {symbol}")

    data_lines = lines[1:]
    if not data_lines:
        raise ValueError("No exponent data found after element symbol")

    try:
        data = [array(line.split(), dtype=float) for line in data_lines]
    except ValueError as e:
        raise ValueError(f"Failed to parse numeric data: {e}") from e

    if args.verbose:
        print(f"Element: {symbol}")
        print(f"Shell lengths: {[len(row) for row in data]}")

    
    exp_set = Exponent_Set(1, symbol, data, None, None, contracted= False)
    exp_set.save(Path.cwd(), f"{symbol}.expo", overwrite=True)

