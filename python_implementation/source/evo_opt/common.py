from enum import Enum


class Job_Status(Enum):
    CREATED   = "created"
    PREPARED  = "prepared"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED    = "failed"

class Remote_Pullback_Policy(Enum):
    MINIMAL  = "minimal"   # e.g. only energy from log file at remote
    STANDARD = "standard"  # log file and optionaly RASORB file
    FULL     = "full"      # everything in remote job dir


class Executor_Type(Enum):
    LOCAL_BASH           = "local_bash"
    LOCAL_SLURM          = "local_slurm"
    REMOTE_BASH_SERIAL   = "remote_bash_serial"
    REMOTE_SLURM_SERIAL  = "remote_slurm_serial"
    REMOTE_BASH_BATCHED  = "remote_bash_batched"
    REMOTE_SLURM_BATCHED = "remote_slurm_batched"

SERIAL_EXECUTORS = {
    Executor_Type.LOCAL_BASH,
    Executor_Type.LOCAL_SLURM,
    Executor_Type.REMOTE_BASH_SERIAL,
    Executor_Type.REMOTE_SLURM_SERIAL,
}

BATCHED_EXECUTORS = {
    Executor_Type.REMOTE_BASH_BATCHED,
    Executor_Type.REMOTE_SLURM_BATCHED,
}

REMOTE_EXECUTORS = {
    Executor_Type.REMOTE_BASH_SERIAL,
    Executor_Type.REMOTE_SLURM_SERIAL,
    Executor_Type.REMOTE_BASH_BATCHED,
    Executor_Type.REMOTE_SLURM_BATCHED,
}


