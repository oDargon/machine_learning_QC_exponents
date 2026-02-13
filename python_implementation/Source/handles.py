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
    def __init__(self, process: subprocess.Popen):
        self.process = process

    def is_finished(self) -> bool:
        return self.process.poll() is not None

    def return_code(self):
        return self.process.poll()