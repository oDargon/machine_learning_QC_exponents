from exponent_handler import *
from pathlib import Path
from datetime import datetime
from numpy import zeros
from job_manager import *
import copy

class Gauntlet:
    def __init__(self, input_files: List[str | Path], cfg: Job_Manager_Config, working_dir: Optional[str | Path] = None) -> None:

        self.input_files = [Path(f) for f in input_files]
        self.cfg         = cfg
        self.working_dir = Path(working_dir) if working_dir else self.cfg.run_script.parent 
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.methods_present: list[dict[str, str]] = []
        self.molecules_info: list[str]             = []
        self.molecules: list[str]                  = []
        self._initialized                          = False

        self.initialize()

    def initialize(self) -> None:

        if not self._initialized:   
            for f in self.input_files:
                if not f.exists():
                    raise FileNotFoundError(f"Input file does not exist: {f}")

            self.extract_methods()
            self.extract_molecules()

            for idx, methods in enumerate(self.methods_present):
                if not methods:
                    raise ValueError(f"No methods found in input file {self.input_files[idx]}")

            for idx, info in enumerate(self.molecules_info):
                if not info.strip():
                    raise ValueError(f"No molecule info found in input file {self.input_files[idx]}")
                
            self._initialized = True

        else:
            raise RuntimeError("Gauntlet is already initialized, cannot initialize again.")

    def extract_method(self, idx: int) -> None:

        method_info = {}

        i = 0
        with self.input_files[idx].open("r") as f:
            lines = f.readlines()
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith("<Method>"):
                    i += 1
                    method_str  = ""
                    method_name = lines[i].strip()
                    i += 1
                    while True:
                        if i >= len(lines):
                            raise ValueError(f"Unexpected end of file while parsing method {method_name} in {self.input_files[idx]}, all methods should be properly closed with </Method> tags.")
                        if lines[i].startswith("</Method>"):
                            break
                        method_str += lines[i]
                        i += 1

                    method_info[method_name] = method_str
                i += 1
        
        self.methods_present.append(method_info)

    def extract_methods(self) -> None:
        
        for idx in range(len(self.input_files)):
            self.extract_method(idx)

    def extract_molecule(self, idx: int) -> None:
        """
        Extracts the <Molecule> block from the input file at index idx.
        Saves:
        - molecules_info: the full block *excluding* the first line (coordinates only)
        - molecules: the first line after <Molecule> (the molecule name)
        """
        with self.input_files[idx].open("r") as f:
            lines = f.readlines()

        molecule_block = ""
        molecule_name = ""
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("<Molecule>"):
                i += 1  # move to first line inside block

                if i >= len(lines):
                    raise ValueError(
                        f"No molecule name found after <Molecule> in {self.input_files[idx]}"
                    )

                # first line is the molecule name
                molecule_name = lines[i].strip()
                i += 1

                # remaining lines until </Molecule> are molecule info
                while True:
                    if i >= len(lines):
                        raise ValueError(
                            f"Unexpected end of file while parsing molecule in {self.input_files[idx]}"
                        )
                    if lines[i].startswith("</Molecule>"):
                        break
                    molecule_block += lines[i]
                    i += 1

                break  # stop after first molecule block
            i += 1

        if not molecule_name:
            raise ValueError(f"No molecule name found in {self.input_files[idx]}")

        self.molecules_info.append(molecule_block)
        self.molecules.append(molecule_name)
    
    def extract_molecules(self):

        for idx in range(len(self.input_files)):
            self.extract_molecule(idx)
    
    def build_and_store_input(self, molecule_name: str, method_name: str, output_dir: str | Path) -> int:

        output_dir = Path(output_dir)

        if not output_dir.exists() or not output_dir.is_dir():
            raise ValueError(f"Output directory does not exist: {output_dir}")

        try:
            mol_idx = self.molecules.index(molecule_name)
        except ValueError:
            raise ValueError(f"Molecule '{molecule_name}' not found.")

        molecule_info = self.molecules_info[mol_idx]
        methods_dict  = self.methods_present[mol_idx]
        
        if method_name not in methods_dict:
            return 0  # method not present for this molecule

        method_block = methods_dict[method_name]
        content      = molecule_info + "\n" + method_block
        input_file   = output_dir / f"{molecule_name}_{method_name}.input"

        with input_file.open("w") as f:
            f.write(content)

        return 1



    def run_gauntlet(self, exponents: Exponent_Set, mols: List[str], methods: List[str], name: Optional[str] = None) -> None:
        
        
        for mol in mols:
            if mol not in self.molecules:
                raise ValueError(f"Molecule '{mol}' not available.")

        new_gauntlet_dir = self.working_dir / f"gauntlet_{name}" if name else self.working_dir / f"gauntlet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_gauntlet_dir.mkdir(parents=True, exist_ok=False)
        

        log_file = new_gauntlet_dir / "gauntlet_log.txt"
        with log_file.open("w") as log:
            log.write(f"Gauntlet run started at {datetime.now()}\n")
            log.write(f"Input files: {', '.join(str(f) for f in self.input_files)}\n")
            log.write(f"Run script: {self.cfg.run_script}\n")
            log.write(f"Exponents: {exponents.__str__()}\n")
            log.write(f"Molecules: {', '.join(mols)}\n")
            log.write(f"Methods: {', '.join(methods)}\n")

        existence_matrix = zeros((len(mols), len(methods)), dtype=bool)
        names_matrix     = [[None for i in methods] for j in mols]

        input_dir        = new_gauntlet_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=False)
        for i in range(len(mols)):
            for j in range(len(methods)):
                existence_matrix[i, j] = self.build_and_store_input(mols[i], methods[j], input_dir)
                names_matrix[i, j]     = f"{mols[i]}_{methods[j]}"

        molcas_work_dir = new_gauntlet_dir / "molcas_work"
        molcas_work_dir.mkdir(parents=True, exist_ok=False)

        new_cfg                = copy(self.cfg)
        new_cfg.group_dir_path = molcas_work_dir
        M                      = Job_Manager.from_config(new_cfg)
        for i in range(len(mols)):
            for j in range(len(methods)):
                if existence_matrix[i, j]:
                    input_file = input_dir / f"{mols[i]}_{methods[j]}.input"
                    M.add_job(exponents, input_file, names_matrix[i, j])
        M.run_all_jobs()


        with log_file.open("a") as log:
            log.write(f"Gauntlet run completed at {datetime.now()}.\n")