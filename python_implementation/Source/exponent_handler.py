from typing import Optional, List, Sequence, Union
from numpy import allclose, delete, ndarray, float64, eye, array
from pathlib import Path


class Exponent_Set:
    def __init__(
        self,
        label: Optional[int]                                              = None,
        atom_name: Optional[str]                                          = None,
        exponents: Optional[List[Sequence[float] | ndarray]]              = None,
        contractions: Optional[List[Sequence[Sequence[float]] | ndarray]] = None,
        method: Optional[str]                                             = None,
        *,
        contracted: Optional[bool]                                        = None,
        energy: Optional[float]                                           = None
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

    def _initialize(self) -> None:
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
 
        if  self._raw_contractions is not None and len(self._raw_contractions) != len(self.exponents):
            raise ValueError("Number of contraction shells must match exponent shells")

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

    def _is_identity(self, mat: ndarray, *, rtol=1e-12, atol=1e-14) -> bool:
        n, m = mat.shape
        if n != m:
            return False
        return allclose(mat, eye(n, dtype=mat.dtype), rtol=rtol, atol=atol)

    @staticmethod
    def _validate_exponents(exp: ndarray, idx: int) -> None:
        if exp.ndim != 1:
            raise ValueError(f"exponents[{idx}] must be 1D")

    @staticmethod
    def _validate_contractions(cont: ndarray, exp: ndarray, idx: int) -> None:
        if cont.ndim != 2:
            raise ValueError(f"contractions[{idx}] must be 2D")
        if cont.shape[0] != exp.shape[0]:
            raise ValueError(
                f"contractions[{idx}] rows must match number of exponents"
            )


    # ---------------- core behavior ----------------

    def copy_without_energy(self) -> "Exponent_Set":
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


    def assign_results(self, *, energy: Optional[float] = None) -> None:
        if energy is not None:
            self.energy = float(energy)
        self.used = True

    # ---------------- presentation ----------------

    def __str__(self) -> str:
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

    def print_exponents(self) -> None:
        print(f"Exponents for set with label {self.label}:")
        for i in range(len(self.exponents)):
            print(f"    l = {i}")
            print(f"    {self.exponents[i]}")


    def save(
        self,
        directory: Union[str, Path] = ".",
        filename: Optional[str] = None,
        *,
        overwrite: bool = False,
    ) -> Path:

        # ---------- normalize path ----------
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # ---------- filename ----------
        if filename is None:
            atom = self.atom_name or "X"
            label = self.label if self.label is not None else "nolabel"
            filename = f"{atom}_{label}.expo"

        if not filename.endswith(".expo"):
            filename += ".expo"

        path = directory / filename

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")

        # ---------- defaults ----------
        atom = self.atom_name or "X"
        method = self.method or "Unknown"
        energy = "NONE" if self.energy is None else f"{self.energy:.16e}"
        contracted = self.contracted

        # ---------- write ----------
        with path.open("w") as f:
            # metadata
            f.write(f"ATOM: {atom}\n")
            f.write(f"ENERGY: {energy}\n")
            f.write(f"CONTRACTED: {'TRUE' if contracted else 'FALSE'}\n")

            # method (optional)
            if method != "Unknown":
                f.write("<METHOD>\n")
                f.write(method.rstrip() + "\n")
                f.write("</METHOD>\n")

            # exponents
            f.write("<EXPONENTS>\n")
            f.write(f"{len(self.exponents)}\n")

            for exp in self.exponents:
                f.write(f"{exp.shape[0]}\n")
                f.write(" ".join(f"{v:.16e}" for v in exp) + "\n")

            f.write("</EXPONENTS>")

            # contractions (optional)
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
    def load(cls, path: str | Path) -> "Exponent_Set":
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

                method_lines = []
                while True:
                    if i >= n:
                        raise ValueError("Unexpected end of file inside <METHOD> block")
                    if lines[i] == "</METHOD>":
                        break
                    method_lines.append(lines[i])
                    i += 1

                method = "\n".join(method_lines)
                i += 1
                continue


            # ---------- EXPONENTS ----------
            if line == "<EXPONENTS>":
                found_exponents = True
                i += 1

                if i >= n:
                    raise ValueError("Unexpected end of file while reading number of exponent shells")
                n_shells = int(lines[i])
                i += 1

                for q in range(n_shells):
                    if i >= n:
                        raise ValueError(f"Unexpected end of file while reading number of primitives for shell {q}")
                    n_prim = int(lines[i])
                    i += 1

                    if i >= n:
                        raise ValueError(f"Unexpected end of file while reading exponents for shell {q}")
                    vals = list(map(float, lines[i].split()))
                    if len(vals) != n_prim:
                        raise ValueError(f"Exponent count mismatch in shell {q}")
                    exponents_raw.append(vals)
                    i += 1

                if i >= n or lines[i] != "</EXPONENTS>":
                    raise ValueError("Missing </EXPONENTS> block")
                i += 1
                continue


            # ---------- CONTRACTION ----------
            if line == "<CONTRACTION>":
                found_contraction = True
                i += 1

                if i >= n:
                    raise ValueError("Unexpected end of file while reading <CONTRACTION> block header")
                n_shells = int(lines[i])
                i += 1

                if n_shells != len(exponents_raw):
                    raise ValueError(
                        "Number of contraction shells does not match number of exponent shells"
                    )

                for l in range(n_shells):
                    if i >= n:
                        raise ValueError(f"Unexpected end of file while reading number of contracted functions for shell {l}")
                    n_cont = int(lines[i])
                    i += 1

                    rows = []
                    for q in range(len(exponents_raw[l])):
                        if i >= n:
                            raise ValueError(f"Unexpected end of file while reading contraction row {q} for shell {l}")
                        row = list(map(float, lines[i].split()))
                        if len(row) != n_cont:
                            raise ValueError(f"Contraction row size mismatch in shell {l}, row {q}")
                        rows.append(row)
                        i += 1

                    contractions_raw.append(rows)

                if i >= n or lines[i] != "</CONTRACTION>":
                    raise ValueError("Missing </CONTRACTION> block")
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
    def from_file(cls, path: str) -> "Exponent_Set":
        return cls.load(path)

    def remove_exponent_uncontracted(self, l: int, q: int) -> None:
        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")
        if q < 0 or q >= self.lengths[l]:
            raise IndexError(f"Invalid exponent index q={q} for shell l={l}")
        
        # Update lengths
        self.lengths[l]  -= 1
        self.exponents[l] = delete(self.exponents[l], q)

        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot remove exponent without updating contraction matrix. Use remove_exponent_contracted() instead.")
        
        else:
            # If not contracted, we need to maintain the invariant that the contraction matrix is identity
            n = self.lengths[l]
            self.n_contracted[l] = n
            self.contractions[l] = eye(n, dtype=float64)

    def add_exponent_uncontracted(self, l: int, value: float) -> None:
        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")
        
        # Update lengths
        self.lengths[l]  += 1
        self.exponents[l] = array(list(self.exponents[l]) + [value], dtype=float64)

        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot add exponent without updating contraction matrix. Use add_exponent_contracted() instead.")
        
        else:
            # If not contracted, we need to maintain the invariant that the contraction matrix is identity
            n = self.lengths[l]
            self.n_contracted[l] = n
            self.contractions[l] = eye(n, dtype=float64)

    def change_exponent_uncontracted(self, l: int, q: int, value: float) -> None:
        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")
        if q < 0 or q >= self.lengths[l]:
            raise IndexError(f"Invalid exponent index q={q} for shell l={l}")
        
        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot change exponent with this method. Use change_exponent_uncontracted() instead.")
        
        self.exponents[l][q] = value

        