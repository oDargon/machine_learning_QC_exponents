from enum import Enum


# Chemical accuracy: 1 kcal/mol expressed in Hartree (1 Ha = 627.509474 kcal/mol).
CHEMICAL_ACCURACY = 1.0 / 627.509474   # ~= 1.5936e-3 Eh


# Spectroscopic angular-momentum labels, l = 0 .. 15 (the max MOLCAS handles).
# Alphabetical after 'f', omitting 'j', plus 'p'/'s' which are already used for
# lower l. shell_label() callers should fall back to str(l) beyond this range.
L_LABELS = ["s", "p", "d", "f", "g", "h", "i", "k", "l", "m", "n", "o", "q", "r", "t", "u"]


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


