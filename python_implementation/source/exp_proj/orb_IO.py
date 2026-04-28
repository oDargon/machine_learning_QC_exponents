from numpy import float64, str_, zeros, array
from numpy.typing import NDArray
from typing import List, Tuple


class MO_input:
    def __init__(
            self,
            title: str,
            info_vals: NDArray[float64],
            orbital_coeffs: List[NDArray[float64]],
            occupation: None | List[NDArray[float64]]         = None,
            one_electron_energy: None | List[NDArray[float64]] = None,
            index_labels: None | List[NDArray[str_]]           = None,
            two_el_energy: float | None                        = None
    ):
        self.title               = title
        self.info_vals           = info_vals
        self.orbital_coeffs      = orbital_coeffs
        self.occupation          = occupation
        self.one_electron_energy = one_electron_energy
        self.index_labels        = index_labels
        self.two_el_energy       = two_el_energy

        self.mo_mode           = self.info_vals[0]
        self.nsym              = int(self.info_vals[1])
        self.wavefunction_type = int(self.info_vals[2])

        if self.mo_mode != 0:
            raise ValueError(
                f"Undefined mode: {self.info_vals[0]}. "
                "Expected 0 (MO input); UHF (mode 1) is not yet supported."
            )
        if self.nsym < 0 or self.nsym > 8:
            raise ValueError("nsym must be an integer in [0, 8].")

        self.nbas = self.info_vals[3:3 + self.nsym].astype(int)
        self.norb = self.info_vals[3 + self.nsym:3 + 2 * self.nsym].astype(int)

        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.info_vals.ndim != 1:
            raise ValueError("info_vals must be a 1D array.")

        if len(self.orbital_coeffs) != self.nsym:
            raise ValueError(
                f"orbital_coeffs must have {self.nsym} entries (one per irrep), "
                f"got {len(self.orbital_coeffs)}."
            )
        for i in range(self.nsym):
            if self.orbital_coeffs[i].ndim != 2:
                raise ValueError(f"orbital_coeffs[{i}] must be a 2D array.")
            expected = (self.norb[i], self.nbas[i])
            if self.orbital_coeffs[i].shape != expected:
                raise ValueError(
                    f"orbital_coeffs[{i}] has shape {self.orbital_coeffs[i].shape}, "
                    f"expected {expected}."
                )

        if self.occupation is not None:
            if len(self.occupation) != self.nsym:
                raise ValueError(
                    f"occupation must have {self.nsym} entries, got {len(self.occupation)}."
                )
            for i in range(self.nsym):
                if self.occupation[i].shape != (self.norb[i],):
                    raise ValueError(
                        f"occupation[{i}] has shape {self.occupation[i].shape}, "
                        f"expected ({self.norb[i]},)."
                    )

        if self.one_electron_energy is not None:
            if len(self.one_electron_energy) != self.nsym:
                raise ValueError(
                    f"one_electron_energy must have {self.nsym} entries, "
                    f"got {len(self.one_electron_energy)}."
                )
            for i in range(self.nsym):
                if self.one_electron_energy[i].shape != (self.norb[i],):
                    raise ValueError(
                        f"one_electron_energy[{i}] has shape "
                        f"{self.one_electron_energy[i].shape}, expected ({self.norb[i]},)."
                    )

        if self.index_labels is not None:
            if len(self.index_labels) != self.nsym:
                raise ValueError(
                    f"index_labels must have {self.nsym} entries, got {len(self.index_labels)}."
                )
            for i in range(self.nsym):
                if self.index_labels[i].shape != (self.norb[i],):
                    raise ValueError(
                        f"index_labels[{i}] has shape {self.index_labels[i].shape}, "
                        f"expected ({self.norb[i]},)."
                    )

        if self.two_el_energy is not None and not isinstance(self.two_el_energy, (int, float)):
            raise TypeError("two_el_energy must be a number if provided.")

    def copy_full(self) -> 'MO_input':
        return MO_input(
            title               = self.title,
            info_vals           = self.info_vals.copy(),
            orbital_coeffs      = [b.copy() for b in self.orbital_coeffs],
            occupation          = [o.copy() for o in self.occupation] if self.occupation is not None else None,
            one_electron_energy = [e.copy() for e in self.one_electron_energy] if self.one_electron_energy is not None else None,
            index_labels        = [l.copy() for l in self.index_labels] if self.index_labels is not None else None,
            two_el_energy       = self.two_el_energy,
        )

    def copy_essential(self) -> 'MO_input':
        return MO_input(
            title               = self.title,
            info_vals           = self.info_vals.copy(),
            orbital_coeffs      = [b.copy() for b in self.orbital_coeffs],
            occupation          = [o.copy() for o in self.occupation] if self.occupation is not None else None,
            one_electron_energy = None,
            index_labels        = [l.copy() for l in self.index_labels] if self.index_labels is not None else None,
            two_el_energy       = None,
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INP_ORB_VERSION_MAGIC = "#INPORB 2.2"

_SECTION_HEADERS = frozenset({
    "#INFO", "#EXTRAS", "#ORB", "#OCC", "#OCHR", "#ONE", "#INDEX"
})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_flat_floats(
        lines: list, start: int, n: int
) -> Tuple[NDArray[float64], int]:
    """Read exactly *n* floats from consecutive lines starting at *start*.

    Returns the values as a 1-D float64 array and the index of the next
    unconsumed line.  Lines may contain any number of whitespace-separated
    values; partial last lines are handled correctly.
    """
    values: List[float] = []
    counter = start
    while len(values) < n:
        values.extend(float(x) for x in lines[counter].split())
        counter += 1
    return array(values[:n], dtype=float64), counter


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_mo_input(file_path: str) -> MO_input:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != INP_ORB_VERSION_MAGIC:
        raise ValueError(
            f"Unsupported version: {lines[0].strip()!r}. "
            f"Expected {INP_ORB_VERSION_MAGIC!r}."
        )

    # If the file contains multiple MO sets, read only the first one.
    end = len(lines)
    for i in range(1, len(lines)):
        if lines[i].strip() == INP_ORB_VERSION_MAGIC:
            end = i
            break

    # Map each section header to its line index (first occurrence only).
    section_pos: dict = {}
    for i in range(1, end):
        s = lines[i].strip()
        if s in _SECTION_HEADERS and s not in section_pos:
            section_pos[s] = i

    # --- #INFO ---
    ip        = section_pos["#INFO"]
    title     = lines[ip + 1].strip()
    row0      = [int(x) for x in lines[ip + 2].split()]   # mo_mode, nsym, wf_type
    row1      = [int(x) for x in lines[ip + 3].split()]   # nbas per irrep
    row2      = [int(x) for x in lines[ip + 4].split()]   # norb per irrep
    info_vals = array(row0 + row1 + row2, dtype=float64)

    nsym = int(info_vals[1])
    nbas = [int(info_vals[3 + k]) for k in range(nsym)]
    norb = [int(info_vals[3 + nsym + k]) for k in range(nsym)]

    # --- #EXTRAS ---
    two_el_energy = None
    if "#EXTRAS" in section_pos:
        ep = section_pos["#EXTRAS"]
        two_el_energy = float(lines[ep + 2].strip())

    # --- #ORB ---
    if "#ORB" not in section_pos:
        raise ValueError("No #ORB section found in file.")

    counter = section_pos["#ORB"] + 1
    orbital_coeffs: List[NDArray[float64]] = []
    for i in range(nsym):
        coef_block   = zeros([norb[i], nbas[i]])
        rows_per_orb = (nbas[i] + 4) // 5  # ceil(nbas[i] / 5)
        for j in range(norb[i]):
            counter += 1          # skip "* ORBITAL  <irrep>  <orb>" header
            pos = 0
            for _ in range(rows_per_orb):
                if counter >= end or lines[counter].strip().startswith('*'):
                    break
                nums = [float(x) for x in lines[counter].split()]
                n    = len(nums)
                coef_block[j, pos:pos + n] = nums
                pos     += n
                counter += 1
        orbital_coeffs.append(coef_block)

    # --- #OCC ---
    occupation = None
    if "#OCC" in section_pos:
        # skip "#OCC" line and "* OCCUPATION NUMBERS" comment
        counter = section_pos["#OCC"] + 2
        occupation = []
        for i in range(nsym):
            arr, counter = _read_flat_floats(lines, counter, norb[i])
            occupation.append(arr)

    # #OCHR is skipped on read — it is redundant with #OCC and derived on write.

    # --- #ONE ---
    one_electron_energy = None
    if "#ONE" in section_pos:
        # skip "#ONE" line and "* ONE ELECTRON ENERGIES" comment
        counter = section_pos["#ONE"] + 2
        one_electron_energy = []
        for i in range(nsym):
            arr, counter = _read_flat_floats(lines, counter, norb[i])
            one_electron_energy.append(arr)

    # --- #INDEX ---
    index_labels = None
    if "#INDEX" in section_pos:
        counter = section_pos["#INDEX"] + 1
        index_labels = []
        for i in range(nsym):
            counter += 1          # skip "* 1234567890" column-ruler header
            chars: List[str] = []
            while len(chars) < norb[i]:
                row_str = lines[counter].strip()
                # Format: "D CCCCCC..." where D is the tens-digit prefix
                chars.extend(list(row_str[2:]))
                counter += 1
            index_labels.append(array(chars[:norb[i]], dtype=str_))

    return MO_input(
        title               = title,
        info_vals           = info_vals,
        orbital_coeffs      = orbital_coeffs,
        occupation          = occupation,
        one_electron_energy = one_electron_energy,
        index_labels        = index_labels,
        two_el_energy       = two_el_energy,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_mo_input(mo: MO_input, file_path: str) -> None:
    """Write *mo* to an INPORB 2.2 file at *file_path*.

    Sections present in *mo* are written; absent optional sections are omitted.
    The output is suitable for a round-trip: read → write → read preserves all
    data within the precision of each section's fixed-point format.
    """
    with open(file_path, 'w') as f:

        # Version header
        f.write(f"{INP_ORB_VERSION_MAGIC}\n")

        # #INFO
        f.write("#INFO\n")
        f.write(f"{mo.title}\n")
        f.write(f"{int(mo.mo_mode):8d}{mo.nsym:8d}{mo.wavefunction_type:8d}\n")
        f.write(''.join(f"{int(b):8d}" for b in mo.nbas) + '\n')
        f.write(''.join(f"{int(n):8d}" for n in mo.norb) + '\n')
        f.write("*BC: written by orb_IO\n")

        # #EXTRAS
        if mo.two_el_energy is not None:
            f.write("#EXTRAS\n")
            f.write("* ACTIVE TWO-EL ENERGY\n")
            f.write(f" {mo.two_el_energy:.12E}\n")

        # #ORB — orbital coefficients, 5 per line, 22.14E format
        f.write("#ORB\n")
        for i in range(mo.nsym):
            for j in range(mo.norb[i]):
                f.write(f"* ORBITAL{i + 1:5d}{j + 1:5d}\n")
                row = mo.orbital_coeffs[i][j]
                for k, v in enumerate(row):
                    f.write(f"{v:22.14E}")
                    if (k + 1) % 5 == 0:
                        f.write('\n')
                if mo.nbas[i] % 5 != 0:
                    f.write('\n')

        # #OCC — occupation numbers, 5 per line, 22.14E format
        if mo.occupation is not None:
            f.write("#OCC\n")
            f.write("* OCCUPATION NUMBERS\n")
            for i in range(mo.nsym):
                for k, v in enumerate(mo.occupation[i]):
                    f.write(f"{v:22.14E}")
                    if (k + 1) % 5 == 0:
                        f.write('\n')
                if mo.norb[i] % 5 != 0:
                    f.write('\n')

            # #OCHR — human-readable occupation numbers, derived from occupation
            f.write("#OCHR\n")
            f.write("* OCCUPATION NUMBERS (HUMAN-READABLE)\n")
            for i in range(mo.nsym):
                for k, v in enumerate(mo.occupation[i]):
                    f.write(f"{v:8.4f}")
                    if (k + 1) % 10 == 0:
                        f.write('\n')
                if mo.norb[i] % 10 != 0:
                    f.write('\n')

        # #ONE — one-electron energies, 10 per line, 12.4E format
        if mo.one_electron_energy is not None:
            f.write("#ONE\n")
            f.write("* ONE ELECTRON ENERGIES\n")
            for i in range(mo.nsym):
                for k, v in enumerate(mo.one_electron_energy[i]):
                    f.write(f"{v:12.4E}")
                    if (k + 1) % 10 == 0:
                        f.write('\n')
                if mo.norb[i] % 10 != 0:
                    f.write('\n')

        # #INDEX — orbital type labels, 10 per row, "D CCCCCCCCCC" format
        if mo.index_labels is not None:
            f.write("#INDEX\n")
            for i in range(mo.nsym):
                f.write("* 1234567890\n")
                labels = mo.index_labels[i]
                pos    = 0
                row    = 0
                while pos < mo.norb[i]:
                    chunk = labels[pos:pos + 10]
                    f.write(f"{row % 10} {''.join(str(c) for c in chunk)}\n")
                    pos += 10
                    row += 1
