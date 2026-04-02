from pathlib import Path
from enum import Enum
import re
import shutil
from .exponent_handler import Exponent_Set
from .parsing import make_input_from_template
import subprocess


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
        extract_path: Path,
        exponent_set: Exponent_Set,
        overwrite: bool = False,
        *,
        input_name: str = None,
        logging: bool   = False,
        name: str       = None,
        
    ):
        
        self.job_id         = job_id
        self.job_dir        = Path(job_dir)
        self.template_path  = Path(template_path)
        self.extract_path   = Path(extract_path)
        self.input_name     = input_name if input_name is not None else "input"
        self.logging        = logging
        self.expo_name      = name if name is not None else "None"
        self.handle         = None #Handle used by the manager to asses job completion

        # Freeze exponent_set inside job
        self.exponent_set = exponent_set.copy_without_energy()
        self.overwrite    = overwrite

        # Runtime state
        self.status          = Job_Status.CREATED
        self.external_job_id = None
        self.results         = None

        # Derived paths
        self.input_file   = self.job_dir / (self.input_name +".input")
        self.output_file  = self.job_dir / (self.input_name+".log")
        self.extract_file = self.job_dir / ("extractor.sh")

    
    def prepare_job(self):
        if (self.logging): print(f"[MolcasJob] Preparing job '{self.job_id}' in {self.job_dir}")


        if self.status != Job_Status.CREATED:
            raise RuntimeError("Job already prepared or processed")

        self.exponent_set.label = self.job_id

        if self.job_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"Job directory {self.job_dir} already exists")
            else:
                # ---- SAFETY GUARD ----
                if len(self.job_dir.resolve().parts) < 3:
                    raise RuntimeError(
                        f"Refusing to delete shallow directory: {self.job_dir}"
                    )

                shutil.rmtree(self.job_dir)

        self.job_dir.mkdir(parents=True)

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file {self.template_path} does not exist")


        self.method              = self.extract_method_from_template()
        self.exponent_set.method = self.method

        make_input_from_template(self.input_file, self.template_path, self.exponent_set, job_id=self.job_id)
        self.make_extractor()
        self.status = Job_Status.PREPARED
        # self.exponent_set.save(self.job_dir)

    def make_extractor(self):
        if not self.extract_path.exists():
            raise FileNotFoundError(f"Extractor file {self.extract_path} does not exist")

        shutil.copy(self.extract_path, self.extract_file)
        self.extract_file.chmod(0o755)

    def update_from_output(self):
        if self.status != Job_Status.SUBMITTED:
            raise RuntimeError("Cannot update job that wasn't submitted")
        
        if not self.output_file.exists():
            self.status = Job_Status.FAILED
            if self.logging:
                print(f"[MolcasJob] Output file not found for job '{self.job_id}'. Marked as FAILED.")
            self.save_exponent_file()
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

    def save_exponent_file(self, custom_location: Path | str = None):

        if custom_location is not None:
            if self.expo_name == "None":
                self.exponent_set.save(custom_location)
            else:
                self.exponent_set.save(custom_location, self.expo_name)
        else:
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

    # def extract_energy_from_output(self):
    #     energy = None

    #     if not self.output_file.exists():
    #         return None

    #     with open(self.output_file) as f:
    #         for line in f:
    #             if "::" in line:
    #                 parts = line.strip().split()
    #                 for token in reversed(parts):
    #                     try:
    #                         energy = float(token)
    #                         break   # stop scanning tokens in this line
    #                     except ValueError:
    #                         continue

    #     return energy
    
    # def extract_energy_from_output(self):
    #     energies = []

    #     if not self.output_file.exists():
    #         return None

    #     with open(self.output_file) as f:
    #         for line in f:
    #             if "::" in line:
    #                 if "CASPT2 Root" in line:
    #                     parts = line.strip().split()
    #                     for token in reversed(parts):
    #                         try:
    #                             energy = float(token)
    #                             energies.append(energy)
    #                             break
    #                         except ValueError:
    #                             continue

    #     if not energies:
    #         return None

    #     return sum(energies) / len(energies)

    
    def extract_energy_from_output(self):
        if not self.output_file.exists():
            return None

        if not self.extract_file.exists():
            return None

        try:
            result = subprocess.run(
                [str(self.extract_file), str(self.output_file)],
                cwd=self.job_dir,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if self.logging:
                print(f"[MolcasJob] Extractor failed for job '{self.job_id}': {e.stderr.strip()}")
            return None

        output = result.stdout.strip()

        if not output:
            return None

        try:
            return float(output)
        except ValueError:
            if self.logging:
                print(f"[MolcasJob] Invalid extractor output for job '{self.job_id}': {output}")
            return None

    def mark_submitted(self):

        if self.status != Job_Status.PREPARED:
            raise RuntimeError("Cannot submit unprepared job")

        self.status = Job_Status.SUBMITTED
