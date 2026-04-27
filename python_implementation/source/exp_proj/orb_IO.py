from numpy import ndarray, float64, str_, zeros, array, fromstring
from numpy.typing import NDArray
from typing import List


class MO_input:
    def __init__(
            self,
            title: str,
            info_vals: NDArray[float64],
            orbital_coeffs: List[NDArray[float64]],
            occupation: None | List[NDArray[float64]]    = None,
            one_electron_energy: None | List[NDArray[float64]] = None,
            index_labels: None | List[NDArray[str_]]     = None,
            two_el_energy: float | None                  = None
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
            raise ValueError(f"Undefined mode encountered: {self.info_vals[0]}. Expected 0 for MO input or 1 for UHF (currently unsupported).")

        if self.nsym < 0 or self.nsym > 8:
            raise ValueError("Number of symmetries (nsym) must be a non-negative integer <= 8.")

        self.nbas = self.info_vals[3:3+self.nsym].astype(int)
        self.norb = self.info_vals[3+self.nsym:3+2*self.nsym].astype(int)

        self.validate_shapes()

    def validate_shapes(self) -> None:
        if self.info_vals.ndim != 1:
            raise ValueError("info_vals must be a 1D array.")

        if len(self.orbital_coeffs) != self.nsym:
            raise ValueError(f"orbital_coeffs must have one entry per irrep, expected {self.nsym} got {len(self.orbital_coeffs)}.")
        for i in range(self.nsym):
            if self.orbital_coeffs[i].ndim != 2:
                raise ValueError(f"orbital_coeffs[{i}] must be a 2D array.")
            if self.orbital_coeffs[i].shape != (self.norb[i], self.nbas[i]):
                raise ValueError(f"orbital_coeffs[{i}] has shape {self.orbital_coeffs[i].shape}, expected ({self.norb[i]}, {self.nbas[i]}).")

        if self.occupation is not None:
            if len(self.occupation) != self.nsym:
                raise ValueError(f"occupation must have one entry per irrep, expected {self.nsym} got {len(self.occupation)}.")
            for i in range(self.nsym):
                if self.occupation[i].shape != (self.norb[i],):
                    raise ValueError(f"occupation[{i}] has shape {self.occupation[i].shape}, expected ({self.norb[i]},).")

        if self.one_electron_energy is not None:
            if len(self.one_electron_energy) != self.nsym:
                raise ValueError(f"one_electron_energy must have one entry per irrep, expected {self.nsym} got {len(self.one_electron_energy)}.")
            for i in range(self.nsym):
                if self.one_electron_energy[i].shape != (self.norb[i],):
                    raise ValueError(f"one_electron_energy[{i}] has shape {self.one_electron_energy[i].shape}, expected ({self.norb[i]},).")

        if self.index_labels is not None:
            if len(self.index_labels) != self.nsym:
                raise ValueError(f"index_labels must have one entry per irrep, expected {self.nsym} got {len(self.index_labels)}.")
            for i in range(self.nsym):
                if self.index_labels[i].shape != (self.norb[i],):
                    raise ValueError(f"index_labels[{i}] has shape {self.index_labels[i].shape}, expected ({self.norb[i]},).")

        if self.two_el_energy is not None and not isinstance(self.two_el_energy, (int, float)):
            raise TypeError("two_el_energy must be a number if provided.")

    def copy_full(self) -> 'MO_input':
        return MO_input(
            title               = self.title,
            info_vals           = self.info_vals.copy(),
            orbital_coeffs      = [block.copy() for block in self.orbital_coeffs],
            occupation          = [occ.copy() for occ in self.occupation] if self.occupation is not None else None,
            one_electron_energy = [e.copy() for e in self.one_electron_energy] if self.one_electron_energy is not None else None,
            index_labels        = [l.copy() for l in self.index_labels] if self.index_labels is not None else None,
            two_el_energy       = self.two_el_energy
        )

    def copy_essential(self) -> 'MO_input':
        return MO_input(
            title               = self.title,
            info_vals           = self.info_vals.copy(),
            orbital_coeffs      = [block.copy() for block in self.orbital_coeffs],
            occupation          = [occ.copy() for occ in self.occupation] if self.occupation is not None else None,
            one_electron_energy = None,
            index_labels        = [l.copy() for l in self.index_labels] if self.index_labels is not None else None,
            two_el_energy       = None
        )
    

INP_ORB_VERSION_MAGIC  = "#INPORB 2.2"
UNIQUE_FIELDS_MAGIC    = ["#EXTRAS", "#ORB", "#OCC", "ONE", "INDEX"]
NUMBERS_PER_LINE_MAGIC = 5

def read_mo_input(file_path: str) -> MO_input:

    title               = None
    info_vals           = None
    orbital_coeffs      = None
    occupation          = None
    one_electron_energy = None
    index_labels        = None
    two_el_energy       = None

    with open(file=file_path, mode='r') as f:
        lines = f.readlines()


        counter = 0 
        version = lines[counter].strip()

        if version != INP_ORB_VERSION_MAGIC:
            raise ValueError(f"Unsupported MO input version: {version}. Expected {INP_ORB_VERSION_MAGIC}.")

        for i in range(1, len(lines)):
            if lines[i].strip() == "#INFO":
                title     = lines[i+1].strip()
                info_vals = []
                for j in range(3):
                    info_vals += [int(x) for x in lines[i+2+j].strip().split()]
                
                counter = i + 5
                break
        
        found = [0,0,0,0,0]
        while counter < len(lines):

            if lines[counter].strip() == "#EXTRAS" and not found[0]:
                counter += 2
                two_el_energy = float64(lines[counter].strip())
                counter += 1
                found[0] = 1

            if lines[counter].strip() == "#ORB" and not found[1]:
                num_irreps = info_vals[1]
                nbas       = info_vals[3:3+num_irreps]
                norb       = info_vals[3+num_irreps:3+2*num_irreps]
                counter   += 1
                    
                orbital_coeffs = []
                for i in range(num_irreps):
                    coef_block = zeros([norb[i],nbas[i]])
                    irrep_rows = (nbas[i] + NUMBERS_PER_LINE_MAGIC -1)//NUMBERS_PER_LINE_MAGIC
                    for j in range(norb[i]):
                        pos      = 0
                        counter += 1
                        for z in range(irrep_rows):
                            if lines[counter].strip().startswith('*'):
                                break
                            elif counter > len(lines):
                                break
                            nums                    = array( [float64(x) for x in lines[counter].strip().split()] )
                            n                       = nums.size
                            coef_block[j,pos:pos+n] = nums
                            pos                    += n
                            counter                += 1
                    
                    orbital_coeffs.append(coef_block)    
                found[1] = 1

                print(orbital_coeffs[-1][0])
                exit()
            counter += 1

            


    return MO_input

path = "/home/dzemail/Desktop/Code_Projects/PHD_work/ML_EXP/python_implementation/source/exp_proj/Fe.RasOrb"

read_mo_input(path)