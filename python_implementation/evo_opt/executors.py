import shlex
import subprocess
import tempfile
import shutil
import tarfile
from pathlib import Path
from typing import List 
from enum import Enum
from .molcas_handler import Molcas_Job
from .handles import Bash_Handle, Remote_Bash_Batch_Handle, Slurm_Handle, Remote_Bash_Handle, Remote_Slurm_Handle



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


# def prepare_script_local(job: Molcas_Job, template: Path | str):
#     with open(template) as f:
#         content = f.read()

#     content = content.replace("{{JOB_NAME}}", str(job.input_file))

#     script = job.job_dir / "run.sh"
#     with open(script, "w") as f:
#         f.write(content)

#     script.chmod(0o755)
#     return script

def prepare_script(job: Molcas_Job, template: Path | str):
    with open(template) as f:
        content = f.read()

    content = content.replace("{{JOB_NAME}}", job.input_file.name)

    script = job.job_dir / "run.sh"
    with open(script, "w") as f:
        f.write(content)

    script.chmod(0o755)
    return script



def local_bash_executor(job: Molcas_Job, script_template_path: Path | str, **kwargs):
    """
    Submits a local bash job for the given Molcas_Job.
    Returns a Bash_Handle.
    """
    script_path   = prepare_script(job, script_template_path)
    # Launch process
    output_handle = open(job.output_file, "w")

    process = subprocess.Popen(
        ["bash", script_path.name],
        cwd=job.job_dir,
        stdout=output_handle,
        stderr=subprocess.STDOUT,
    )

    return Bash_Handle(process, output_handle)

def slurm_executor(job: Molcas_Job, script_template_path, **kwargs):
    """
    Submits a local bash job for the given Molcas_Job.
    Returns a Slurm_Handle.
    """
    script_path = prepare_script(job, script_template_path)

    result = subprocess.run(
        ["sbatch", "--parsable", script_path.name],
        cwd=job.job_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    job_id = result.stdout.strip()

    return Slurm_Handle(job_id)

def remote_bash_executor_serial(
    job: Molcas_Job,
    script_template_path: Path | str,
    *,
    ssh_target: str,
    remote_work_root: str | Path,
    remote_pullback_policy,
    pull_rasorb: bool,
    cleanup_remote: bool,
    **kwargs,
):  

    script_path    = prepare_script(job, script_template_path)
    remote_job_dir = Path(remote_work_root) / job.job_dir.name

    result = subprocess.run(
        ["ssh", ssh_target, f"mkdir -p {shlex.quote(str(remote_job_dir))}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create remote job directory '{remote_job_dir}': {result.stderr}")

    files_to_upload = [script_path, job.input_file]

    for path in files_to_upload:
        result = subprocess.run(
            ["scp", str(path), f"{ssh_target}:{str(remote_job_dir)}/"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload '{path.name}' to '{remote_job_dir}': {result.stderr}"
            )

    script_text = Path(script_path).read_text()

    remote_cmd = (
        f"cd {shlex.quote(str(remote_job_dir))}; "
        "{ "
        f"{script_text} "
        "} </dev/null >/dev/null 2>/dev/null & "
        "echo $!"
    )

    result = subprocess.run(
        ["ssh", ssh_target, remote_cmd],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Remote bash execution failed: {result.stderr}")

    if result.stderr.strip():
        err_file = job.job_dir / "execution.err"
        err_file.write_text(result.stderr)

    pid = result.stdout.strip().splitlines()[-1]
    if not pid:
        raise RuntimeError("Failed to retrieve remote PID")

    return Remote_Bash_Handle(
        pid             = pid,
        output_name     = job.output_file.name,
        ssh_target      = ssh_target,
        remote_job_dir  = remote_job_dir,
        local_job_dir   = job.job_dir,
        pullback_policy = remote_pullback_policy,
        pull_rasorb     = pull_rasorb,
        cleanup_remote  = cleanup_remote,
    )

def remote_slurm_executor_serial(
    job: Molcas_Job,
    script_template_path: Path | str,
    *,
    ssh_target: str,
    remote_work_root: str | Path,
    remote_pullback_policy,
    pull_rasorb: bool,
    cleanup_remote: bool,
    **kwargs,
):
    
    script_path    = prepare_script(job, script_template_path)
    remote_job_dir = Path(remote_work_root) / job.job_dir.name

    result = subprocess.run(
        ["ssh", ssh_target, f"mkdir -p {shlex.quote(str(remote_job_dir))}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create remote job directory '{remote_job_dir}': {result.stderr}")

    files_to_upload = [script_path,job.input_file]

    for path in files_to_upload:
        result = subprocess.run(
            ["scp", str(path), f"{ssh_target}:{str(remote_job_dir)}/"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload '{path.name}' to '{remote_job_dir}': {result.stderr}"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to upload '{path.name}' to '{remote_job_dir}': {result.stderr}"
            )

    result = subprocess.run(
        ["ssh", ssh_target,
            (   f"cd {shlex.quote(str(remote_job_dir))} && "
                f"sbatch --parsable {shlex.quote(script_path.name)}")
        ],
        capture_output=True,
        text=True
    )

    if result.stderr.strip():
        err_file = job.job_dir / "execution.err"
        err_file.write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Remote sbatch failed in '{remote_job_dir}': {result.stderr}")

    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError(f"Remote sbatch returned no job id for local job '{job.job_id}'.")

    return Remote_Slurm_Handle(
        job_id          = job_id,
        output_name     = job.output_file.name,
        ssh_target      = ssh_target,
        remote_job_dir  = remote_job_dir,
        local_job_dir   = job.job_dir,
        pullback_policy = remote_pullback_policy,
        pull_rasorb     = pull_rasorb,
        cleanup_remote  = cleanup_remote,
    )

def remote_bash_executor_batched(
    jobs: List[Molcas_Job],
    script_template_path: Path | str,
    *,
    ssh_target: str,
    remote_work_root: str | Path,
    remote_pullback_policy,
    pull_rasorb: bool,
    cleanup_remote: bool,
    **kwargs,
):
    remote_work_root = Path(remote_work_root)

    with tempfile.TemporaryDirectory() as tmpdir:
        staging_root = Path(tmpdir) / "batch_payload"
        staging_root.mkdir(parents=True, exist_ok=True)

        # Build staging tree
        for job in jobs:
            script_path = prepare_script(job, script_template_path)

            staged_job_dir = staging_root / job.job_dir.name
            staged_job_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(script_path, staged_job_dir / script_path.name)
            shutil.copy2(job.input_file, staged_job_dir / job.input_file.name)

        # Create tarball
        tar_path = Path(tmpdir) / "batch_payload.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(staging_root, arcname=".")

        remote_tar = remote_work_root / "batch_payload.tar.gz"

        # Upload tarball
        result = subprocess.run(
            ["scp", str(tar_path), f"{ssh_target}:{str(remote_tar)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload tarball: {result.stderr}")

        # Extract remotely and delete tarball
        remote_cmd = (
            f"cd {shlex.quote(str(remote_work_root))} && "
            f"tar -xzf {shlex.quote(remote_tar.name)} && "
            f"rm -f {shlex.quote(remote_tar.name)}"
        )

        result = subprocess.run(
            ["ssh", ssh_target, remote_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to extract remote tarball: {result.stderr}")


    remote_cmd_lines = [
    f"cd {shlex.quote(str(remote_work_root))}",
    'PID_FILE=".batch_pids"',
    '> "$PID_FILE"',
    ]

    for job in jobs:
        remote_job_dir = Path(remote_work_root) / job.job_dir.name
        script_text    = (job.job_dir / "run.sh").read_text()

        remote_cmd_lines.append(
            f'cd {shlex.quote(str(remote_job_dir))}; '
            '{ '
            f'{script_text} '
            '} </dev/null >/dev/null 2>/dev/null & '
            f'echo "{job.job_dir.name}:$!" >> "$PID_FILE"; '
            f'cd {shlex.quote(str(remote_work_root))}'
        )

    remote_cmd_lines.append('cat "$PID_FILE"')

    remote_cmd = "\n".join(remote_cmd_lines)

    result = subprocess.run(
        ["ssh", ssh_target, remote_cmd],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Remote batch launch failed:\n{result.stderr}")

    expected_names = {job.job_dir.name for job in jobs}
    pid_map        = {}

    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        job_name, pid = line.split(":", 1)
        pid_map[job_name] = pid

    received_names = set(pid_map)
    missing = expected_names - received_names

    if missing:
        raise RuntimeError(
            "Some remote jobs failed to launch.\n"
            f"Expected: {sorted(expected_names)}\n"
            f"Started:  {sorted(received_names)}\n"
            f"Missing:  {sorted(missing)}"
        )


    output_name_map    = {job.job_dir.name: job.output_file.name for job in jobs}
    remote_job_dir_map = {job.job_dir.name: remote_work_root / job.job_dir.name for job in jobs}
    local_job_dir_map  = {job.job_dir.name: job.job_dir for job in jobs}

    return Remote_Bash_Batch_Handle(
        pid_map            = pid_map,
        output_name_map    = output_name_map,
        ssh_target         = ssh_target,
        remote_work_root   = remote_work_root,
        remote_job_dir_map = remote_job_dir_map,
        local_job_dir_map  = local_job_dir_map,
        pullback_policy    = remote_pullback_policy,
        pull_rasorb        = pull_rasorb,
        cleanup_remote     = cleanup_remote,
    )

def remote_slurm_executor_batched(
    jobs: List[Molcas_Job],
    script_template_path: Path | str,
    *,
    ssh_target: str,
    remote_work_root: str | Path,
    remote_pullback_policy,
    pull_rasorb: bool,
    cleanup_remote: bool,
    **kwargs,
): 
    return

executor_map = {
    Executor_Type.LOCAL_BASH:          local_bash_executor,
    Executor_Type.LOCAL_SLURM:         slurm_executor,
    Executor_Type.REMOTE_BASH_SERIAL:  remote_bash_executor_serial,
    Executor_Type.REMOTE_SLURM_SERIAL: remote_slurm_executor_serial,
    Executor_Type.REMOTE_BASH_BATCHED:  remote_bash_executor_batched,  
    Executor_Type.REMOTE_SLURM_BATCHED: remote_slurm_executor_batched,  
}

