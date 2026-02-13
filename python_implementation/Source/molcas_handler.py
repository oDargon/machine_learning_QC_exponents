from pathlib import Path
from enum import Enum
import re
from exponent_handler import *
from handles import *

class Job_Status(Enum):
    CREATED   = "created"
    PREPARED  = "prepared"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED    = "failed"


class Molcas_Job:

    def __init__(
        self,
        job_id: int,
        job_dir: Path,
        template_path: Path,
        exponent_set: Exponent_Set,
        overwrite: bool = False,
        *,
        input_name: str = None,
        logging: bool   = False,
        name: str       = None,
        
    ):
        
        self.job_id        = job_id
        self.job_dir       = Path(job_dir)
        self.template_path = Path(template_path)
        self.exponent_set  = exponent_set
        self.input_name    = input_name if input_name is not None else "input"
        self.logging       = logging
        self.expo_name     = name if name is not None else "None"
        self.handle        = None #Handle used by the manager to asses job completion

        # Freeze exponent_set inside job
        self.exponent_set = exponent_set.copy()
        self.overwrite    = overwrite

        # Runtime state
        self.status          = Job_Status.CREATED
        self.external_job_id = None
        self.results         = None

        # Derived paths
        self.input_file  = self.job_dir / (self.input_name +".input")
        self.output_file = self.job_dir / (self.input_name+".log")

    
    def prepare_job(self):
        print("Sexy")
        if (self.logging): print(f"[MolcasJob] Preparing job '{self.job_id}' in {self.job_dir}")

        self.exponent_set.label = self.job_id

        if self.job_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"Job directory {self.job_dir} already exists")
            else:
                import shutil
                shutil.rmtree(self.job_dir)

        self.job_dir.mkdir(parents=True)

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file {self.template_path} does not exist")


        self.method = self.extract_method_from_template()
        self.exponent_set.method = self.method

        self.make_input_from_template()
        self.status = Job_Status.PREPARED
        # self.exponent_set.save(self.job_dir)


    def replacer(self, match):
        kind, num_str = match.groups()
        index = int(num_str)

        try:
            if kind == "NUMS":
                return "    " + str(self.exponent_set.lengths[index]) + " " + str(self.exponent_set.n_contracted[index])
            elif kind == "EXPS":
                values = self.exponent_set.exponents[index]
                return " ".join(f"{v:.10f}" for v in values)
            elif kind == "CONT":
                matrix = self.exponent_set.contractions[index]
                return "\n".join(
                    " ".join(f"{value:.10f}" for value in row)
                    for row in matrix
                )
        except IndexError:
            raise IndexError(f"Placeholder {kind}{num_str} exceeds Exponent_Set size {len(self.exponent_set.exponents)}")


    def make_input_from_template(self):

        if (self.logging): print(f"[MolcasJob] Writing input to {self.input_file}")

        with open(self.template_path) as f:
            text = f.read()

        pattern  = re.compile(r"(NUMS|EXPS|CONT)(\d+)")
        new_text = pattern.sub(self.replacer, text)

        with open(self.input_file, "w") as f:
            f.write(new_text)


    def update_from_output(self):
        if not self.output_file.exists():
            self.status = Job_Status.FAILED
            if self.logging:
                print(f"[MolcasJob] Output file not found for job '{self.job_id}'. Marked as FAILED.")
            self._save_exponent_file()
            return

        # Extract energy
        energy = self.extract_energy_from_output()

        if energy is None:
            self.status = Job_Status.FAILED
            if self.logging:
                print(f"[MolcasJob] Could not extract energy for job '{self.job_id}'. Marked as FAILED.")
            self.save_exponent_file()
            return

        # Energy found, mark complete
        self.results = {"energy": energy}
        self.status  = Job_Status.COMPLETED

        # Update exponent set
        self.exponent_set.assign_results(energy=energy)
        self.save_exponent_file()

        if self.logging:
            print(f"[MolcasJob] Job '{self.job_id}' completed. Energy: {energy}")

    def save_exponent_file(self):
        if self.expo_name == "None":
            self.exponent_set.save(self.job_dir)
        else:
            self.exponent_set.save(self.job_dir, self.expo_name)


    def extract_method_from_template(self):
        """
        Extracts the method block from the template file.
        Method is defined as everything after &SEWARD,
        excluding comment (*) and blank lines.
        Line breaks are preserved between kept lines.
        """
        method_lines    = []
        in_seward_block = False

        with open(self.template_path) as f:
            for line in f:
                stripped = line.strip()

                if not in_seward_block:
                    if stripped.upper().startswith("&SEWARD"):
                        in_seward_block = True
                    continue

                # After &SEWARD
                if not stripped:           # skip blank lines
                    continue
                if stripped.startswith("*"):  # skip Molcas comments
                    continue

                method_lines.append(stripped)

        if not in_seward_block:
            raise ValueError(
                f"&SEWARD block not found in template {self.template_path}"
            )

        method_block = "\n".join(method_lines)

        if not method_block and self.logging:
            print(f"[MolcasJob] Warning: SEWARD found but no method content detected.")

        return method_block

    def extract_energy_from_output(self):
        energy = None

        if not self.output_file.exists():
            return None

        print(self.output_file)

        with open(self.output_file) as f:
            for line in f:
                if "::" in line:
                    parts = line.strip().split()
                    for token in reversed(parts):
                        try:
                            energy = float(token)
                            return energy
                        except ValueError:
                            continue

        return None

    

    def mark_submitted(self):
        self.status = Job_Status.SUBMITTED
