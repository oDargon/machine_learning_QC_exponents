from pathlib import Path
import re
from .exponent_handler import Exponent_Set
from .job_manager import Remote_Pullback_Policy
from .executors import Executor_Type
from typing import Optional



BASIS_PLACEHOLDER = "{BASIS}"
ADVANCED_PATTERN  = re.compile(r"(NUMS|EXPS|CONT)(\d+)")
ATOMIC_NUMBERS    = {
    "H": 1,   "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,   "O": 8,   "F": 9,   "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,  "S": 16,  "Cl": 17, "Ar": 18,
    "K": 19,  "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53,  "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,  "Np": 93, "Pu": 94,
    "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
    "Lr": 103,
    "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}

def format_basis_block(exponent_set: Exponent_Set) -> str:
    symbol = exponent_set.atom_name.capitalize()

    try:
        atomic_number = ATOMIC_NUMBERS[symbol]
    except KeyError as e:
        raise ValueError(f"Unknown atomic symbol '{exponent_set.atom_name}'") from e

    max_angular_momentum = len(exponent_set.exponents) - 1

    parts = [
        f"{exponent_set.atom_name}.....  / inline",
        f"    {atomic_number}      {max_angular_momentum}"
    ]

    for i in range(len(exponent_set.exponents)):
        nums = (
            f"    {exponent_set.lengths[i]} "
            f"{exponent_set.n_contracted[i]}"
        )
        exps = " ".join(f"{v:.10f}" for v in exponent_set.exponents[i])
        cont = "\n".join(
            " ".join(f"{value:.10f}" for value in row)
            for row in exponent_set.contractions[i]
        )

        parts.append(f"{nums}\n{exps}\n{cont}")

    return "\n".join(parts)

def make_replacer(exponent_set: Exponent_Set, job_id: Optional[str] = None):
    def replacer(match) -> str:
        kind, num_str = match.groups()
        index = int(num_str)

        try:
            if kind == "NUMS":
                return f"    {exponent_set.lengths[index]} {exponent_set.n_contracted[index]}"
            elif kind == "EXPS":
                values = exponent_set.exponents[index]
                return " ".join(f"{v:.10f}" for v in values)
            elif kind == "CONT":
                matrix = exponent_set.contractions[index]
                return "\n".join(
                    " ".join(f"{value:.10f}" for value in row)
                    for row in matrix
                )
            else:
                raise ValueError(f"Unknown placeholder kind: {kind}")
        except IndexError as e:
            msg = (
                f"Placeholder {kind}{num_str} exceeds Exponent_Set size "
                f"{len(exponent_set.exponents)}"
            )
            if job_id is not None:
                msg += f" for job {job_id}"
            raise IndexError(msg) from e

    return replacer

def make_input_from_template(input_file: str | Path, template_file: str | Path, exponent_set: Exponent_Set, job_id: Optional[str] = None) -> None:
    input_file    = Path(input_file)
    template_file = Path(template_file)
    text          = template_file.read_text()

    if BASIS_PLACEHOLDER in text:
        if ADVANCED_PATTERN.search(text):
            msg = (
                f"Template mixes {BASIS_PLACEHOLDER} with NUMS/EXPS/CONT "
                f"placeholders. Use one style only."
            )
            if job_id is not None:
                msg += f" Job: {job_id}"
            raise ValueError(msg)

        new_text = text.replace(BASIS_PLACEHOLDER, format_basis_block(exponent_set))

    else:
        new_text = ADVANCED_PATTERN.sub(
            make_replacer(exponent_set, job_id=job_id),
            text,
        )

        if ADVANCED_PATTERN.search(new_text):
            msg = "Unresolved placeholders remain in template"
            if job_id is not None:
                msg += f" for job {job_id}"
            raise ValueError(msg)

    input_file.write_text(new_text)

def parse_pullback_policy(value: str | None) -> Remote_Pullback_Policy:
    if value is None:
        return Remote_Pullback_Policy.STANDARD

    normalized = value.strip().lower()

    for member in Remote_Pullback_Policy:
        if member.value == normalized:
            return member

    allowed = ", ".join(m.value for m in Remote_Pullback_Policy)
    raise ValueError(f"Unknown remote_pullback_policy '{value}'. Allowed values: {allowed}")

def parse_executor_type(value: str) -> Executor_Type:
    normalized = value.strip().lower()

    for member in Executor_Type:
        if member.value == normalized:
            return member

    allowed = ", ".join(m.value for m in Executor_Type)
    raise ValueError(f"Unknown executor_type '{value}'. Allowed values: {allowed}")