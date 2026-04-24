import subprocess
from abc import ABC, abstractmethod
import shlex
from pathlib import Path
import tarfile
import tempfile
from time import sleep

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
    

class Remote_Slurm_Handle(Handle):
    def __init__(
        self,
        job_id: str,
        output_name: str,
        ssh_target: str,
        remote_job_dir: str | Path,
        local_job_dir: str | Path,
        pullback_policy,
        pull_rasorb: bool = False,
        cleanup_remote: bool = True,
    ):
        self.job_id          = job_id
        self.output_name     = output_name
        self.ssh_target      = ssh_target
        self.remote_job_dir  = str(remote_job_dir)
        self.local_job_dir   = Path(local_job_dir)
        self.pullback_policy = pullback_policy
        self.pull_rasorb     = pull_rasorb
        self.cleanup_remote  = cleanup_remote

        self._return_code      = None
        self._artifacts_synced = False
        self._cleanup_done     = False

    def _run_ssh(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", self.ssh_target, command],
            capture_output=True,
            text=True
        )

    def is_finished(self) -> bool:
        result = self._run_ssh(f"squeue -j {shlex.quote(self.job_id)}")
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to query remote squeue for job {self.job_id}: {result.stderr}"
            )

        lines = result.stdout.strip().split("\n")

        if len(lines) <= 1:
            self._update_return_code()

            if not self._artifacts_synced:
                self._pull_back_results()
                self._artifacts_synced = True

            if self.cleanup_remote and not self._cleanup_done:
                self._cleanup_remote_dir()
                self._cleanup_done = True

            return True

        return False

    def _update_return_code(self):
        if self._return_code is not None:
            return

        result = self._run_ssh(f"sacct -j {shlex.quote(self.job_id)} --format=State --noheader")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to query remote sacct for job {self.job_id}: {result.stderr}")

        stdout = result.stdout.strip()
        if not stdout:
            self._return_code = 1
            return

        state = stdout.split()[0]

        if state == "COMPLETED":
            self._return_code = 0
        else:
            self._return_code = 1

    def _pull_back_results(self):

        files_to_copy = []

        if self.pullback_policy.name == "MINIMAL":
            files_to_copy.append(self.output_name + ".log")
        elif self.pullback_policy.name == "STANDARD":
            files_to_copy.append(self.output_name + ".log")
        elif self.pullback_policy.name == "FULL":
            subprocess.run(
                [
                    "scp", "-r",
                    f"{self.ssh_target}:{self.remote_job_dir}/.",
                    str(self.local_job_dir)
                ],
                check=False
            )
            return

        if self.pull_rasorb:
            files_to_copy.append(self.output_name + ".RasOrb")

        for fname in files_to_copy:
            subprocess.run(
                [
                    "scp",
                    f"{self.ssh_target}:{self.remote_job_dir}/{fname}",
                    str(self.local_job_dir / fname)
                ],
                check=False
            )

    def _cleanup_remote_dir(self):

        remote_path = Path(self.remote_job_dir)

        if len(remote_path.parts) < 4:
            raise RuntimeError(
                f"Refusing to delete shallow remote path: {self.remote_job_dir}"
            )

        unsafe = {"", "/", ".", "~"}
        if self.remote_job_dir.strip() in unsafe:
            raise RuntimeError(
                f"Refusing to delete unsafe remote path: {self.remote_job_dir}"
            )

        result = self._run_ssh(f"rm -rf {shlex.quote(self.remote_job_dir)}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to delete remote job directory '{self.remote_job_dir}': {result.stderr}")

    def return_code(self):
        return self._return_code
    
class Remote_Bash_Handle(Handle):

    def __init__(
        self,
        pid: str,
        output_name: str,
        ssh_target: str,
        remote_job_dir: str | Path,
        local_job_dir: str | Path,
        pullback_policy,
        pull_rasorb: bool    = False,
        cleanup_remote: bool = True,
    ):
        self.pid              = pid
        self.output_name      = output_name
        self.ssh_target       = ssh_target
        self.remote_job_dir   = str(remote_job_dir)
        self.local_job_dir    = Path(local_job_dir)
        self.pullback_policy  = pullback_policy
        self.pull_rasorb      = pull_rasorb
        self.cleanup_remote   = cleanup_remote

        self._return_code      = None
        self._artifacts_synced = False
        self._cleanup_done     = False

    def _run_ssh(self, command: str):
        return subprocess.run(
            ["ssh", self.ssh_target, command],
            capture_output=True,
            text=True
        )

    def is_finished(self) -> bool:
        result = self._run_ssh(f"kill -0 {shlex.quote(self.pid)} 2>/dev/null")
        if result.returncode != 0:
            if not self._artifacts_synced:
                self._pull_back_results()
                self._artifacts_synced = True

            if self.cleanup_remote and not self._cleanup_done:
                self._cleanup_remote_dir()
                self._cleanup_done = True

            return True

        return False

    def _pull_back_results(self):

        files_to_copy = []

        if self.pullback_policy.name in ("MINIMAL", "STANDARD"):
            files_to_copy.append(self.output_name)

        elif self.pullback_policy.name == "FULL":
            subprocess.run(
                [
                    "scp", "-r",
                    f"{self.ssh_target}:{self.remote_job_dir}/.",
                    str(self.local_job_dir)
                ],
                check=False
            )
            return

        if self.pull_rasorb:
            files_to_copy.append(self.output_name + ".RasOrb")

        for fname in files_to_copy:
            subprocess.run(
            [
                "scp",
                "-q",  # quiet
                f"{self.ssh_target}:{self.remote_job_dir}/{fname}",
                str(self.local_job_dir / fname)
            ]
        )

    def _cleanup_remote_dir(self):
        remote_path = Path(self.remote_job_dir)

        if len(remote_path.parts) < 4:
            raise RuntimeError(f"Refusing to delete shallow remote path: {self.remote_job_dir}")

        result = self._run_ssh(f"rm -rf {shlex.quote(self.remote_job_dir)}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to delete remote job directory '{self.remote_job_dir}': {result.stderr}")

    def return_code(self):
        return self._return_code
    
class Remote_Bash_Batch_Handle(Handle):
    def __init__(
        self,
        pid_map: dict[str, str],
        output_name_map: dict[str, str],
        ssh_target: str,
        remote_work_root: str | Path,
        remote_job_dir_map: dict[str, str | Path],
        local_job_dir_map: dict[str, str | Path],
        pullback_policy,
        pull_rasorb: bool = False,
        cleanup_remote: bool = True,
    ):
        self.pid_map            = pid_map
        self.output_name_map    = output_name_map
        self.ssh_target         = ssh_target
        self.remote_work_root   = str(remote_work_root)
        self.remote_job_dir_map = {k: str(v) for k, v in remote_job_dir_map.items()}
        self.local_job_dir_map  = {k: Path(v) for k, v in local_job_dir_map.items()}
        self.pullback_policy    = pullback_policy
        self.pull_rasorb        = pull_rasorb
        self.cleanup_remote     = cleanup_remote

        self._return_code       = None
        self._artifacts_synced  = False
        self._cleanup_done      = False

        cmd_lines = []

        for job_name, pid in self.pid_map.items():
            cmd_lines.append(f"kill -0 {shlex.quote(pid)} 2>/dev/null || echo dead:{job_name}")

        self._status_cmd = "\n".join(cmd_lines)

    def _run_ssh(self, command: str):
        return subprocess.run(
            ["ssh", self.ssh_target, command],
            capture_output=True,
            text=True
        )
    
    def is_finished(self) -> bool:
        result = self._run_ssh(self._status_cmd)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to poll remote jobs: {result.stderr}")

        dead_jobs = set()

        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("dead:"):
                dead_jobs.add(line.split(":", 1)[1])

        # if any job still alive → not finished
        if len(dead_jobs) != len(self.pid_map):
            return False

        # all dead → finished → trigger pullback + cleanup
        if not self._artifacts_synced:
            self._pull_back_results()
            self._artifacts_synced = True

        if self.cleanup_remote and not self._cleanup_done:
            self._cleanup_remote_dirs()
            self._cleanup_done = True

        return True
    
    def _pull_back_results(self):

        remote_stage_dir = Path(self.remote_work_root) / ".batch_pullback"
        remote_tar       = Path(self.remote_work_root) / "batch_results.tar.gz"

        cmd_lines = [
            f'cd {shlex.quote(str(self.remote_work_root))}',
            f'rm -rf {shlex.quote(remote_stage_dir.name)}',
            f'mkdir -p {shlex.quote(remote_stage_dir.name)}',
        ]

        for job_name, remote_job_dir in self.remote_job_dir_map.items():
            remote_job_dir = Path(remote_job_dir)
            output_name    = self.output_name_map[job_name]

            cmd_lines.append(f'mkdir -p {shlex.quote(str(Path(remote_stage_dir.name) / job_name))}')

            if self.pullback_policy.name in ("MINIMAL", "STANDARD"):
                cmd_lines.append(
                    f'cp {shlex.quote(str(remote_job_dir / output_name))} '
                    f'{shlex.quote(str(remote_stage_dir / job_name / output_name))}'
                )

                if self.pull_rasorb:
                    cmd_lines.append(
                        f'cp {shlex.quote(str(remote_job_dir / (output_name + ".RasOrb")))} '
                        f'{shlex.quote(str(remote_stage_dir / job_name / (output_name + ".RasOrb")))}'
                    )

            elif self.pullback_policy.name == "FULL":
                cmd_lines.append(
                f'cp -r {shlex.quote(str(remote_job_dir))}/. '
                f'{shlex.quote(str(remote_stage_dir / job_name))}'
            )

        cmd_lines.append(f'tar -C {shlex.quote(str(remote_stage_dir))} -czf {shlex.quote(remote_tar.name)} .')

        remote_cmd = "\n".join(cmd_lines)

        result = self._run_ssh(remote_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stage remote results: {result.stderr}")

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir       = Path(tmpdir)
            local_tar    = tmpdir / "batch_results.tar.gz"
            extract_root = tmpdir / "extracted"
            extract_root.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["scp", f"{self.ssh_target}:{str(remote_tar)}", str(local_tar)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to copy remote tarball back: {result.stderr}")

            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=extract_root)
                            
            for job_name, local_job_dir in self.local_job_dir_map.items():
                extracted_job_dir = extract_root / job_name
                local_job_dir     = Path(local_job_dir)

                if not extracted_job_dir.exists():
                    raise RuntimeError(f"Missing extracted results for job '{job_name}'")

                subprocess.run(
                ["cp", "-r", f"{str(extracted_job_dir)}/.", str(local_job_dir)],
                check=True,
            )

    def _cleanup_remote_dirs(self):
        remote_paths = [Path(p) for p in self.remote_job_dir_map.values()]
        remote_paths.append(Path(self.remote_work_root) / ".batch_pullback")
        remote_paths.append(Path(self.remote_work_root) / "batch_results.tar.gz")

        unsafe = {"", "/", ".", "~"}

        for path in remote_paths:
            if str(path).strip() in unsafe:
                raise RuntimeError(f"Refusing to delete unsafe remote path: {path}")
            if len(path.parts) < 4:
                raise RuntimeError(f"Refusing to delete shallow remote path: {path}")

        joined_paths = " ".join(shlex.quote(str(p)) for p in remote_paths)

        result = self._run_ssh(f"rm -rf {joined_paths}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to delete remote batch paths: {result.stderr}")

    def return_code(self):
        return self._return_code
    
class Remote_Slurm_Batch_Handle(Handle):
    def __init__(
        self,
        jobid_map: dict[str, str],
        output_name_map: dict[str, str],
        ssh_target: str,
        remote_work_root: str | Path,
        remote_job_dir_map: dict[str, str | Path],
        local_job_dir_map: dict[str, str | Path],
        pullback_policy,
        pull_rasorb: bool    = False,
        cleanup_remote: bool = True,
    ):
        self.jobid_map          = jobid_map
        self.output_name_map    = output_name_map
        self.ssh_target         = ssh_target
        self.remote_work_root   = str(remote_work_root)
        self.remote_job_dir_map = {k: str(v) for k, v in remote_job_dir_map.items()}
        self.local_job_dir_map  = {k: Path(v) for k, v in local_job_dir_map.items()}
        self.pullback_policy    = pullback_policy
        self.pull_rasorb        = pull_rasorb
        self.cleanup_remote     = cleanup_remote

        self._return_code       = None
        self._artifacts_synced  = False
        self._cleanup_done      = False

        job_ids = ",".join(str(job_id) for job_id in self.jobid_map.values())
        self._status_cmd = f"squeue -h -j {job_ids}"

    def _run_ssh(self, command: str):
        return subprocess.run(
            ["ssh", self.ssh_target, command],
            capture_output=True,
            text=True
        )
    
    def is_finished(self) -> bool:
        result = self._run_ssh(self._status_cmd)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to poll remote jobs: {result.stderr}")

        active_lines = [line for line in result.stdout.splitlines() if line.strip()]

        if active_lines:
            return False

        if not self._artifacts_synced:
            self._pull_back_results()
            self._artifacts_synced = True

        if self.cleanup_remote and not self._cleanup_done:
            self._cleanup_remote_dirs()
            self._cleanup_done = True

        return True
    
    def _pull_back_results(self):

        remote_stage_dir = Path(self.remote_work_root) / ".batch_pullback"
        remote_tar       = Path(self.remote_work_root) / "batch_results.tar.gz"

        cmd_lines = [
            f'cd {shlex.quote(str(self.remote_work_root))}',
            f'rm -rf {shlex.quote(remote_stage_dir.name)}',
            f'mkdir -p {shlex.quote(remote_stage_dir.name)}',
        ]

        for job_name, remote_job_dir in self.remote_job_dir_map.items():
            remote_job_dir = Path(remote_job_dir)
            output_name    = self.output_name_map[job_name]

            cmd_lines.append(f'mkdir -p {shlex.quote(str(Path(remote_stage_dir.name) / job_name))}')

            if self.pullback_policy.name in ("MINIMAL", "STANDARD"):
                cmd_lines.append(
                    f'cp {shlex.quote(str(remote_job_dir / output_name))} '
                    f'{shlex.quote(str(remote_stage_dir / job_name / output_name))}'
                )

                if self.pull_rasorb:
                    cmd_lines.append(
                        f'cp {shlex.quote(str(remote_job_dir / (output_name + ".RasOrb")))} '
                        f'{shlex.quote(str(remote_stage_dir / job_name / (output_name + ".RasOrb")))}'
                    )

            elif self.pullback_policy.name == "FULL":
                cmd_lines.append(
                f'cp -r {shlex.quote(str(remote_job_dir))}/. '
                f'{shlex.quote(str(remote_stage_dir / job_name))}'
            )

        cmd_lines.append(f'tar -C {shlex.quote(str(remote_stage_dir))} -czf {shlex.quote(remote_tar.name)} .')

        remote_cmd = "\n".join(cmd_lines)

        result = self._run_ssh(remote_cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stage remote results: {result.stderr}")

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir       = Path(tmpdir)
            local_tar    = tmpdir / "batch_results.tar.gz"
            extract_root = tmpdir / "extracted"
            extract_root.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["scp", f"{self.ssh_target}:{str(remote_tar)}", str(local_tar)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to copy remote tarball back: {result.stderr}")

            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=extract_root)
                            
            for job_name, local_job_dir in self.local_job_dir_map.items():
                extracted_job_dir = extract_root / job_name
                local_job_dir     = Path(local_job_dir)

                if not extracted_job_dir.exists():
                    raise RuntimeError(f"Missing extracted results for job '{job_name}'")

                subprocess.run(
                ["cp", "-r", f"{str(extracted_job_dir)}/.", str(local_job_dir)],
                check=True,
            )

    def _cleanup_remote_dirs(self):
        remote_paths = [Path(p) for p in self.remote_job_dir_map.values()]
        remote_paths.append(Path(self.remote_work_root) / ".batch_pullback")
        remote_paths.append(Path(self.remote_work_root) / "batch_results.tar.gz")

        unsafe = {"", "/", ".", "~"}

        for path in remote_paths:
            if str(path).strip() in unsafe:
                raise RuntimeError(f"Refusing to delete unsafe remote path: {path}")
            if len(path.parts) < 4:
                raise RuntimeError(f"Refusing to delete shallow remote path: {path}")

        joined_paths = " ".join(shlex.quote(str(p)) for p in remote_paths)

        result = self._run_ssh(f"rm -rf {joined_paths}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to delete remote batch paths: {result.stderr}")

    def return_code(self):
        return self._return_code