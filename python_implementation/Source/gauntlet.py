from exponent_handler import *
from pathlib import Path
from datetime import datetime

class Gauntlet:
    def __init__(self, input_files: List[str | Path], run_script: str | Path):

        self.input_files = [Path(f) for f in input_files]
        self.run_script  = Path(run_script)
        self.base_dir    = self.run_script.parent

        self.methods_present = List[dict]

    def extract_method(self, idx: int) -> None:

        method_info = {}

        i = 0
        with self.input_files[idx].open("r") as f:
            lines = f.readlines()
            while i < len(lines):
                line = lines[i]
                if line.startswith("<Method>"):
                    i += 1

                    method_str  = ""
                    method_name = lines[i].strip()
                    
                    i += 1
                    while not lines[i].startswith("</Method>"):

                        if (i >= len(lines)):
                            raise ValueError(f"Unexpected end of file while parsing method {method_name} in {self.input_files[idx]}, all methods should be properly closed with </Method> tags.")

                        method_str += lines[i]
                        i += 1
                    
                    method_info[method_name]= method_str
                    
                i += 1
        
        self.methods_present.append(method_info)
    def extract_methods(self) -> None:
        
        
    


    def run_gauntlet(self, exonents: Exponent_Set, mols: List[str], methods: List[str], name: Optional[str] = None) -> None:

        new_gauntlet_dir = self.base_dir / f"gauntlet_{name}" if name else self.base_dir / f"gauntlet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_gauntlet_dir.mkdir(parents=True, exist_ok=False)

        log_file = new_gauntlet_dir / "gauntlet_log.txt"
        with log_file.open("w") as log:
            log.write(f"Gauntlet run started at {datetime.now()}\n")
            log.write(f"Input files: {', '.join(str(f) for f in self.input_files)}\n")
            log.write(f"Run script: {self.run_script}\n")
            log.write(f"Exponents: {exonents.__str__()}\n")
            log.write(f"Molecules: {', '.join(mols)}\n")
            log.write(f"Methods: {', '.join(methods)}\n")







        with log_file.open("a") as log:
            log.write(f"Gauntlet run completed at {datetime.now()}.\n")