import subprocess
from abc import ABC, abstractmethod

class Handle(ABC):
    """
    Base class for all job handles.
    Provides a unified interface for the manager regardless of the underlying process type.
    """
    @abstractmethod
    def is_finished(self) -> bool:
        """Return True if the job is done, False otherwise."""
        pass

    @abstractmethod
    def return_code(self):
        """Return the job's return code if finished, else None."""
        pass


class Bash_Handle(Handle):
    """
    Handle for local bash/subprocess jobs.
    Wraps a subprocess.Popen object.
    """
    def __init__(self, process: subprocess.Popen, output_handle):
        self.process       = process
        self.output_handle = output_handle

    def is_finished(self) -> bool:
        finished = self.process.poll() is not None
        if finished and not self.output_handle.closed:
            self.output_handle.close()
        return finished

    def return_code(self):
        return self.process.poll()
    

class Slurm_Handle(Handle):
    """
    Handle for SLURM submitted jobs.
    Uses the SLURM job id to track job status.
    """
    def __init__(self, job_id: str):
        self.job_id: str  = job_id
        self._return_code = None

    def is_finished(self) -> bool:
        result = subprocess.run(
            ["squeue", "-j", self.job_id],
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split("\n")

        # squeue returns header + job line if running
        if len(lines) <= 1:
            # Job no longer in queue → finished
            self._update_return_code()
            return True

        return False

    def _update_return_code(self):
        if self._return_code is not None:
            return

        result = subprocess.run(
            ["sacct", "-j", self.job_id, "--format=State", "--noheader"],
            capture_output=True,
            text=True
        )

        state = result.stdout.strip().split()[0]

        if state == "COMPLETED":
            self._return_code = 0
        else:
            self._return_code = 1

    def return_code(self):
        return self._return_code