from typing import Optional, List, Sequence, Union
from numpy import allclose, argsort, delete, ndarray, float64, eye, array, hstack
from pathlib import Path


class Exponent_Set:
    def __init__(
        self,
        label: Optional[int]                                              = None,
        atom_name: Optional[str]                                          = None,
        exponents: Optional[List[Sequence[float] | ndarray]]              = None,
        contractions: Optional[List[Sequence[Sequence[float]] | ndarray]] = None,
        contracted_shells: Optional[List[bool]]                           = None,
        method: Optional[str]                                             = None,
        *,
        contracted: Optional[bool]                                                  = None,
        energy: Optional[float]                                                     = None,
        resulting_contraction: Optional[List[Sequence[Sequence[float]] | ndarray]]  = None
    ):
        # ---- metadata ----
        self.label: Optional[int] = label
        self.atom_name: str       = atom_name if atom_name is not None else "X"
        self.method: str          = method if method is not None else "Unknown"

        # ---- raw inputs ----
        self._raw_exponents              = exponents
        self._raw_contractions           = contractions
        self._raw_contracted_shells      = contracted_shells
        self._contracted_flag            = contracted
        self._raw_resulting_contraction  = resulting_contraction

        # ---- core data (declared, filled later) ----
        self.exponents: List[ndarray]                   = []
        self.contractions: List[ndarray]                = []
        self.contracted_shells: List[bool]              = []
        self.lengths: List[int]                         = []
        self.n_contracted: List[int]                    = []
        self.contracted: bool                           = False
        self.resulting_contraction: Optional[List[ndarray]] = None

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
        self.contracted_shells.clear()
        self.lengths.clear()
        self.n_contracted.clear()
        self.resulting_contraction = None

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

        # resolve per-shell contraction activation
        if self._raw_contracted_shells is not None:
            cs = list(self._raw_contracted_shells)
            if len(cs) != len(self.exponents):
                raise ValueError("contracted_shells length must match number of shells")
            self.contracted_shells = [bool(v) for v in cs]
        elif self._contracted_flag is not None:
            self.contracted_shells = [self._contracted_flag] * len(self.exponents)
        else:
            self.contracted_shells = [not self._is_identity(cont) for cont in self.contractions]

        self.contracted = any(self.contracted_shells)

        # derived dimensions
        for exp, cont in zip(self.exponents, self.contractions):
            self.lengths.append(exp.shape[0])
            self.n_contracted.append(cont.shape[0])

        if self._raw_resulting_contraction is not None:
            if len(self._raw_resulting_contraction) != len(self.exponents):
                raise ValueError("Number of resulting_contraction shells must match exponent shells")
            res_conts = []
            for i, cont in enumerate(self._raw_resulting_contraction):
                cont_arr = array(cont, dtype=float64)
                self._validate_contractions(cont_arr, self.exponents[i], i)
                res_conts.append(cont_arr)
            self.resulting_contraction = res_conts

        self._ensure_descending_all()

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
        if cont.shape[1] != exp.shape[0]:
            raise ValueError(
                f"contractions[{idx}] columns ({cont.shape[1]}) must match number of primitives ({exp.shape[0]})"
            )

    def _ensure_descending_all(self) -> None:
        """
        Ensure all shells have exponents sorted descending.
        If any shell is not sorted, reorder exponents and, if contracted, the contraction matrix columns.
        """
        for l, exp in enumerate(self.exponents):
            if exp.size <= 1:
                continue  # nothing to do

            # Check if already sorted (fast path)
            if (exp[:-1] >= exp[1:]).all():
                continue

            # Compute descending order indices
            order = argsort(exp)[::-1]

            # Reorder exponents
            self.exponents[l] = exp[order]

            # Reorder contraction matrix columns (primitive index) whenever the matrix
            # is non-trivial, regardless of per-shell activation — an inactive shell may
            # still have stored coefficients that must stay aligned with exponent ordering
            if not self._is_identity(self.contractions[l]):
                self.contractions[l] = self.contractions[l][:, order]

            if self.resulting_contraction is not None:
                self.resulting_contraction[l] = self.resulting_contraction[l][:, order]

    # ---------------- core behavior ----------------

    def copy(
        self,
        *,
        no_energy: bool             = False,
        no_contractions: bool       = False,
        no_contracted_shells: bool  = False,
    ) -> "Exponent_Set":
        exponents_copy    = [exp.copy() for exp in self.exponents]
        contractions_copy = None if no_contractions else [cont.copy() for cont in self.contractions]
        # contracted_shells can only be meaningful when contractions are present
        shells_copy       = None if (no_contracted_shells or no_contractions) else list(self.contracted_shells)
        res_cont_copy     = (
            [cont.copy() for cont in self.resulting_contraction]
            if self.resulting_contraction is not None else None
        )

        new = Exponent_Set(
            label=self.label,
            atom_name=self.atom_name,
            exponents=exponents_copy,
            contractions=contractions_copy,
            contracted_shells=shells_copy,
            method=self.method,
            energy=None if no_energy else self.energy,
            resulting_contraction=res_cont_copy,
        )

        new.is_copy = True
        new.used    = False

        return new


    def assign_results(
        self,
        *,
        energy: Optional[float]          = None,
        resulting_contraction: Optional[List[ndarray]] = None,
    ) -> None:
        if energy is not None:
            self.energy = float(energy)
        if resulting_contraction is not None:
            if len(resulting_contraction) != len(self.exponents):
                raise ValueError(
                    f"Number of result contraction matrices ({len(resulting_contraction)}) must match "
                    f"number of shells ({len(self.exponents)})"
                )
            new_conts = []
            for i, cont in enumerate(resulting_contraction):
                cont_arr = array(cont, dtype=float64)
                if cont_arr.ndim != 2:
                    raise ValueError(f"resulting_contraction[{i}] must be 2D")
                if cont_arr.shape[1] != self.lengths[i]:
                    raise ValueError(
                        f"resulting_contraction[{i}] has {cont_arr.shape[1]} columns but "
                        f"shell {i} has {self.lengths[i]} primitives"
                    )
                new_conts.append(cont_arr)
            self.resulting_contraction = new_conts
        self.used = True

    # ---------------- presentation ----------------

    def __str__(self) -> str:
        lines = []

        l_max = len(self.exponents) - 1 if self.exponents else -1

        lines.append(f"Exponent_set(label={self.label}, atom={self.atom_name})")
        lines.append( f"method:\n{self.method}")
        lines.append(
            f"  (l_max)={l_max}, "
            f"contracted={self.contracted}, contracted_shells={self.contracted_shells}, copy={self.is_copy}, used={self.used}"
        )

        for l, exp in enumerate(self.exponents):
            lines.append(f"  l = {l}")
            lines.append(
                "    exponents: "
                + " ".join(f"{v:.6f}" for v in exp)
            )

            if self.contracted_shells[l]:
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
        atom   = self.atom_name or "X"
        method = self.method or "Unknown"
        energy = "NONE" if self.energy is None else f"{self.energy:.16e}"

        # Write a CONTRACTION block whenever any shell is active OR has a non-trivial
        # matrix stored, so that CONTRACTED: TRUE always co-occurs with the block.
        needs_contraction = self.contracted or any(
            not self._is_identity(cont) for cont in self.contractions
        )

        # ---------- write ----------
        with path.open("w") as f:
            # metadata
            f.write(f"ATOM: {atom}\n")
            f.write(f"ENERGY: {energy}\n")
            f.write(f"CONTRACTED: {'TRUE' if needs_contraction else 'FALSE'}\n")
            if needs_contraction:
                shells_str = " ".join("1" if s else "0" for s in self.contracted_shells)
                f.write(f"CONTRACTED_SHELLS: {shells_str}\n")

            # method
            if self.method != "Unknown":
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

            # contractions — written whenever any shell is active or has a stored
            # non-trivial matrix, keeping CONTRACTED: TRUE always paired with this block
            if needs_contraction:
                f.write("\n<CONTRACTION>\n")
                f.write(f"{len(self.contractions)}\n")

                for cont in self.contractions:
                    n_cont = cont.shape[0]  # rows = MOs
                    f.write(f"{n_cont}\n")
                    for row in cont:        # each row = one MO over all primitives
                        f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

                f.write("</CONTRACTION>\n")

            if self.resulting_contraction is not None:
                f.write("\n<RES_CONTRACTION>\n")
                f.write(f"{len(self.resulting_contraction)}\n")

                for cont in self.resulting_contraction:
                    f.write(f"{cont.shape[0]}\n")
                    for row in cont:
                        f.write(" ".join(f"{v:.16e}" for v in row) + "\n")

                f.write("</RES_CONTRACTION>\n")

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

        exponents_raw              = []
        contractions_raw           = []
        contracted_shells_raw      = None
        resulting_contraction_raw  = None
        found_exponents            = False
        found_contraction          = False
        found_method               = False

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

            if line.startswith("CONTRACTED_SHELLS:"):
                vals = line.split(":", 1)[1].strip().split()
                contracted_shells_raw = []
                for v in vals:
                    v_up = v.upper()
                    if v_up in ("1", "TRUE"):
                        contracted_shells_raw.append(True)
                    elif v_up in ("0", "FALSE"):
                        contracted_shells_raw.append(False)
                    else:
                        raise ValueError(f"CONTRACTED_SHELLS values must be 1/0 (or TRUE/FALSE), got '{v}'")
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
                    n_prim = len(exponents_raw[l])
                    for q in range(n_cont):  # n_cont rows, each = one MO over all primitives
                        if i >= n:
                            raise ValueError(f"Unexpected end of file while reading contraction row {q} for shell {l}")
                        row = list(map(float, lines[i].split()))
                        if len(row) != n_prim:
                            raise ValueError(f"Contraction row size mismatch in shell {l}, row {q}: expected {n_prim} primitives, got {len(row)}")
                        rows.append(row)
                        i += 1

                    contractions_raw.append(rows)

                if i >= n or lines[i] != "</CONTRACTION>":
                    raise ValueError("Missing </CONTRACTION> block")
                i += 1
                continue


            # ---------- RES_CONTRACTION ----------
            if line == "<RES_CONTRACTION>":
                i += 1

                if i >= n:
                    raise ValueError("Unexpected end of file while reading <RES_CONTRACTION> block header")
                n_shells = int(lines[i])
                i += 1

                if n_shells != len(exponents_raw):
                    raise ValueError(
                        "Number of RES_CONTRACTION shells does not match number of exponent shells"
                    )

                resulting_contraction_raw = []
                for l in range(n_shells):
                    if i >= n:
                        raise ValueError(f"Unexpected end of file while reading number of contracted functions for RES_CONTRACTION shell {l}")
                    n_cont = int(lines[i])
                    i += 1

                    rows = []
                    n_prim = len(exponents_raw[l])
                    for q in range(n_cont):
                        if i >= n:
                            raise ValueError(f"Unexpected end of file while reading RES_CONTRACTION row {q} for shell {l}")
                        row = list(map(float, lines[i].split()))
                        if len(row) != n_prim:
                            raise ValueError(f"RES_CONTRACTION row size mismatch in shell {l}, row {q}: expected {n_prim} primitives, got {len(row)}")
                        rows.append(row)
                        i += 1

                    resulting_contraction_raw.append(rows)

                if i >= n or lines[i] != "</RES_CONTRACTION>":
                    raise ValueError("Missing </RES_CONTRACTION> block")
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

        if not found_contraction:
            contractions_raw = None

        if contracted is None:
            contracted = found_contraction

        if not found_method:
            method = "Unknown"

        # ---------------- construct ----------------
        return cls(
            label=None,
            atom_name=atom,
            exponents=exponents_raw,
            contractions=contractions_raw,
            contracted_shells=contracted_shells_raw,
            method=method,
            energy=energy,
            contracted=contracted,
            resulting_contraction=resulting_contraction_raw,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Exponent_Set":
        return cls.load(Path(path))


#TO DO: FIX THE LACKING CHECK FOR ORDER/REMOVE UNNECESSARY METHODS
    def remove_exponent_uncontracted(self, l: int, q: int) -> None:
        if self.contracted:
            raise ValueError(
                "Exponent set is contracted; cannot remove exponent without updating "
                "contraction matrix. Use remove_exponent_contracted() instead."
            )

        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")
        if q < 0 or q >= self.lengths[l]:
            raise IndexError(f"Invalid exponent index q={q} for shell l={l}")

        self.exponents[l] = delete(self.exponents[l], q)
        self.lengths[l]  -= 1

        n = self.lengths[l]
        self.n_contracted[l] = n
        self.contractions[l] = eye(n, dtype=float64)
        self.resulting_contraction = None

    def add_exponent_uncontracted(self, l: int, value: float) -> None:
        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")

        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot add exponent without updating contraction matrix. Use add_exponent_contracted() instead.")

        self.lengths[l]  += 1
        self.exponents[l] = array(list(self.exponents[l]) + [value], dtype=float64)

        n = self.lengths[l]
        self.n_contracted[l] = n
        self.contractions[l] = eye(n, dtype=float64)
        self.resulting_contraction = None

    def change_exponent_uncontracted(self, l: int, q: int, value: float) -> None:
        if l < 0 or l >= len(self.exponents):
            raise IndexError(f"Invalid shell index l={l}")
        if q < 0 or q >= self.lengths[l]:
            raise IndexError(f"Invalid exponent index q={q} for shell l={l}")
        
        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot change exponent with this method. Use change_exponent_contracted() instead.")
        
        self.exponents[l][q] = value
#/TO DO: FIX THE LACKING CHECK FOR ORDER/REMOVE UNNECESSARY METHODS

    def update_exponent_uncontracted_from_flat_same_shape(self, new_exponents: ndarray) -> None:

        if self.contracted:
            raise ValueError("Exponent set is contracted; cannot update exponents with this method. Use update_exponent_contracted_from_flat() instead.")
        
        if len(new_exponents) != sum(self.lengths):
            raise ValueError("Length of new_exponents does not match total number of exponents defined by exponents_shape")

        idx = 0
        for l, n in enumerate(self.lengths):
            self.exponents[l] = array(new_exponents[idx:idx+n], dtype=float64)
            idx += n

    def flatten_exps(self) -> ndarray:
        nonempty = [exp for exp in self.exponents if len(exp) > 0]
        if not nonempty:
            return array([], dtype=float64)
        return hstack(nonempty)

    def same_shape_as(self, other_set: "Exponent_Set") -> bool:
        return self.lengths == other_set.lengths and self.n_contracted == other_set.n_contracted and self.contracted == other_set.contracted
    

    # ---------------- contraction helpers ----------------

    def add_contraction_shells(self, contracted_shells: List[bool]) -> None:
        if len(contracted_shells) != len(self.exponents):
            raise ValueError(
                f"contracted_shells length ({len(contracted_shells)}) must match "
                f"number of shells ({len(self.exponents)})"
            )
        self.contracted_shells = [bool(v) for v in contracted_shells]
        self.contracted        = any(self.contracted_shells)

    def change_contraction(
        self,
        contractions: List[ndarray],
        contracted_shells: Optional[List[bool]] = None,
    ) -> None:
        if len(contractions) != len(self.exponents):
            raise ValueError(
                f"Number of contraction matrices ({len(contractions)}) must match "
                f"number of shells ({len(self.exponents)})"
            )
        new_conts = []
        for i, cont in enumerate(contractions):
            cont_arr = array(cont, dtype=float64)
            if cont_arr.ndim != 2:
                raise ValueError(f"contractions[{i}] must be 2D")
            if cont_arr.shape[1] != self.lengths[i]:
                raise ValueError(
                    f"contractions[{i}] has {cont_arr.shape[1]} columns but shell {i} "
                    f"has {self.lengths[i]} primitives"
                )
            new_conts.append(cont_arr)

        self.contractions = new_conts
        self.n_contracted = [c.shape[0] for c in self.contractions]
        self.add_contraction_shells(
            contracted_shells if contracted_shells is not None
            else [True] * len(self.exponents)
        )
