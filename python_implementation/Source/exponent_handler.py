from typing import Optional, List
from numpy import ndarray, float64, eye, array
from pathlib import Path
import os


class Exponent_Set:
    def __init__(
        self,
        label: Optional[int]         = None,
        atom_name: Optional[str]     = None,
        exponents: Optional[List]    = None,
        contractions: Optional[List] = None,
        method: Optional[str]        = None,
        *,
        contracted: Optional[bool]   = None,
        energy: Optional[bool]       = None
    ):
        # ---- metadata ----
        self.label: Optional[int] = label
        self.atom_name: str       = atom_name if atom_name is not None else "X"
        self.method: str          = method if method is not None else "Unknown"

        # ---- raw inputs ----
        self._raw_exponents    = exponents
        self._raw_contractions = contractions
        self._contracted_flag  = contracted

        # ---- core data (declared, filled later) ----
        self.exponents: List[ndarray]    = []
        self.contractions: List[ndarray] = []
        self.lengths: List[int]          = []
        self.n_contracted: List[int]     = []
        self.contracted: bool            = False

        # ---- simulation state ----
        self.energy: Optional[float] = energy if energy is not None else None
        self.used: bool              = False
        self.is_copy: bool           = False

        # ---- normalize & validate ----
        self._initialize()

    # ---------------- validation helpers ----------------

    def _initialize(self):
        self.exponents.clear()
        self.contractions.clear()
        self.lengths.clear()
        self.n_contracted.clear()

        # normalize exponents
        if self._raw_exponents is not None:
            for i, exp in enumerate(self._raw_exponents):
                exp_arr = array(exp, dtype=float64)
                self._validate_exponents(exp_arr, i)
                self.exponents.append(exp_arr)

        # normalize contractions
        if self._raw_contractions is not None:
            for i, cont in enumerate(self._raw_contractions):
                cont_arr = array(cont, dtype=float64)
                self._validate_contractions(cont_arr, self.exponents[i], i)
                self.contractions.append(cont_arr)
        else:
            for exp in self.exponents:
                n = exp.shape[0]
                self.contractions.append(eye(n, dtype=float64))

        # infer contraction status from matrices
        has_nontrivial_contraction = any(
            not self._is_identity(cont)
            for cont in self.contractions
        )

        # resolve contracted flag (user overrides inference)
        if self._contracted_flag is not None:
            self.contracted = self._contracted_flag
        else:
            self.contracted = has_nontrivial_contraction

        # derived dimensions
        for exp, cont in zip(self.exponents, self.contractions):
            self.lengths.append(exp.shape[0])
            self.n_contracted.append(cont.shape[1])

    def _is_identity(self, mat):
        n, m = mat.shape
        if n != m:
            return False
        return (mat == eye(n, dtype=mat.dtype)).all()

    @staticmethod
    def _validate_exponents(exp: ndarray, idx: int):
        if exp.ndim != 1:
            raise ValueError(f"exponents[{idx}] must be 1D")

    @staticmethod
    def _validate_contractions(cont: ndarray, exp: ndarray, idx: int):
        if cont.ndim != 2:
            raise ValueError(f"contractions[{idx}] must be 2D")
        if cont.shape[0] != exp.shape[0]:
            raise ValueError(
                f"contractions[{idx}] rows must match number of exponents"
            )


    # ---------------- core behavior ----------------

    def copy(self):
        exponents_copy = [exp.copy() for exp in self.exponents]
        contractions_copy = [cont.copy() for cont in self.contractions]

        new = Exponent_Set(
            label=self.label,
            atom_name=self.atom_name,
            exponents=exponents_copy,
            contractions=contractions_copy,
            method=self.method,
            contracted=self.contracted,
        )

        new.is_copy = True
        new.used = False
        new.energy = None

        return new


    def assign_results(self, *, energy: Optional[float] = None):
        if energy is not None:
            self.energy = float(energy)
        self.used = True

    # ---------------- presentation ----------------

    def __str__(self):
        lines = []

        l_max = len(self.exponents) - 1 if self.exponents else -1

        lines.append(f"Exponent_set(label={self.label}, atom={self.atom_name})")
        lines.append( f"method:\n{self.method}")
        lines.append(
            f"  (l_max)={l_max}, "
            f"contracted={self.contracted}, copy={self.is_copy}, used={self.used}"
        )

        for l, exp in enumerate(self.exponents):
            lines.append(f"  l = {l}")
            lines.append(
                "    exponents: "
                + " ".join(f"{v:.6f}" for v in exp)
            )

            if self.contracted:
                cont = self.contractions[l]
                lines.append("    contraction coefficients:")
                for row in cont:
                    lines.append(
                        "      " + " ".join(f"{v:.6f}" for v in row)
                    )

        if self.energy is not None:
            lines.append(f"  Energy: {self.energy:.10f}")
        else:
            lines.append("  Energy: not computed")

        return "\n".join(lines)

    def print_exponents(self):
        print(f"Exponents for set with label {self.label}:")
        for i in range(len(self.exponents)):
            print(f"    l = {i}")
            print(f"    f{self.exponents[i]}")


    def save(self, directory: str = ".", filename: Optional[str] = None, *, overwrite: bool = False,) -> str:
        # ---------- filename ----------
        if filename is None:
            atom     = self.atom_name if self.atom_name is not None else "X"
            label    = self.label if self.label is not None else "nolabel"
            filename = f"{atom}_{label}.expo"

        if not filename.endswith(".expo"):
            filename += ".expo"

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"File already exists: {path}")

        # ---------- defaults ----------
        atom       = self.atom_name if self.atom_name is not None else "X"
        method     = self.method if self.method is not None else "Unknown"
        energy     = "NONE" if self.energy is None else f"{self.energy:.16e}"
        contracted = self.contracted

        # ---------- write ----------
        with open(path, "w") as f:
            # metadata (always written, fixed order)
            f.write(f"ATOM: {atom}\n")
            f.write(f"ENERGY: {energy}\n")
            f.write(f"CONTRACTED: {'TRUE' if contracted else 'FALSE'}\n")

            # -------- method (optional) --------
            if (method != "Unknown"):
                f.write("<METHOD>\n")
                f.write(method)
                f.write("\n</METHOD>\n")

            # -------- exponents --------
            f.write("<EXPONENTS>\n")
            f.write(f"{len(self.exponents)}\n")

            for exp in self.exponents:
                f.write(f"{exp.shape[0]}\n")
                f.write(" ".join(f"{v:.16e}" for v in exp) + "\n")

            f.write("</EXPONENTS>")

            # -------- contractions (optional) --------
            if contracted:
                f.write("\n<CONTRACTION>\n")
                f.write(f"{len(self.contractions)}\n")

                for cont in self.contractions:
                    n_cont = cont.shape[1]
                    f.write(f"{n_cont}\n")
                    for row in cont:
                        f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

                f.write("</CONTRACTION>\n")

        return path
    
    @classmethod
    def load(cls, path):
        path = Path(path)  # ensure it's a Path object

        if path.suffix.lower() != ".expo":
            raise ValueError(f"Expected .expo file, got {path}")

        if not path.exists():
            raise FileNotFoundError(path)

        # ---------------- helpers ----------------
        def clean_lines(lines):
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line

        # ---------------- defaults ----------------
        atom       = None
        method     = None
        energy     = None
        contracted = None

        exponents_raw     = []
        contractions_raw  = []
        found_exponents   = False
        found_contraction = False
        found_method      = False

        # ---------------- read ----------------
        with open(path, "r") as f:
            lines = list(clean_lines(f))

        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]

            # ---------- metadata ----------
            if line.startswith("ATOM:"):
                atom = line.split(":", 1)[1].strip()
                i += 1
                continue

            if line.startswith("METHOD:"):
                method = line.split(":", 1)[1].strip()
                i += 1
                continue

            if line.startswith("ENERGY:"):
                val = line.split(":", 1)[1].strip()
                energy = None if val == "NONE" else float(val)
                i += 1
                continue

            if line.startswith("CONTRACTED:"):
                val = line.split(":", 1)[1].strip().upper()
                if val not in ("TRUE", "FALSE"):
                    raise ValueError("CONTRACTED must be TRUE or FALSE")
                contracted = (val == "TRUE")
                i += 1
                continue
            

            # ---------- METHOD ----------
            if line == "<METHOD>":
                found_method = True
                i += 1

                method = ""

                while (lines[i] != "</METHOD>"):
                    if (i >= len(lines)):
                        raise ValueError("Missing </METHOD>")

                    method += lines[i]
                    method += "\n"
                    i      += 1
                i += 1
                continue


            # ---------- EXPONENTS ----------
            if line == "<EXPONENTS>":
                found_exponents = True
                i += 1

                n_shells = int(lines[i])
                i += 1

                for q in range(n_shells):
                    n_prim = int(lines[i])
                    i += 1

                    vals = list(map(float, lines[i].split()))
                    if len(vals) != n_prim:
                        raise ValueError("Exponent count mismatch")
                    exponents_raw.append(vals)
                    i += 1

                if lines[i] != "</EXPONENTS>":
                    raise ValueError("Missing </EXPONENTS>")
                i += 1
                continue

            # ---------- CONTRACTION ----------
            if line == "<CONTRACTION>":
                found_contraction = True
                i += 1

                n_shells = int(lines[i])
                i += 1

                for l in range(n_shells):
                    n_cont = int(lines[i])
                    i += 1

                    rows = []
                    for q in range(len(exponents_raw[l])):
                        row = list(map(float, lines[i].split()))
                        if len(row) != n_cont:
                            raise ValueError("Contraction row size mismatch")
                        rows.append(row)
                        i += 1

                    contractions_raw.append(rows)

                if lines[i] != "</CONTRACTION>":
                    raise ValueError("Missing </CONTRACTION>")
                i += 1
                continue

            # ---------- unknown ----------
            raise ValueError(f"Unrecognized line: {line}")

        # ---------------- validation ----------------
        if not found_exponents:
            raise RuntimeError("Missing <EXPONENTS> block")

        if contracted is True and not found_contraction:
            raise RuntimeError(
                "CONTRACTED is TRUE but no <CONTRACTION> block found"
            )

        if contracted is False:
            contractions_raw = None

        if contracted is None:
            contracted = found_contraction

        # ---------------- construct ----------------
        return cls(
            label=None,
            atom_name=atom,
            exponents=exponents_raw,
            contractions=contractions_raw,
            method=method,
            energy=energy,
            contracted=contracted,
        )

    @classmethod
    def from_file(cls, path: str):
        return cls.load(path)
