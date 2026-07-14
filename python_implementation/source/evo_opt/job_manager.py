from pathlib import Path
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from .molcas_handler import Molcas_Job, Job_Status
from .handles import Handle
from .executors import executor_map
from .common import Remote_Pullback_Policy, Executor_Type, BATCHED_EXECUTORS, REMOTE_EXECUTORS
import time
import shutil
from uuid import uuid4





@dataclass
class Job_Manager_Config:
    executor_type: Optional[Executor_Type] = None
    execution_script: Optional[Path]       = None
    extraction_script: Optional[Path]      = None
    group_dir_name: Optional[str]          = None
    group_dir_path: Optional[Path]         = None
    auto_run: bool                         = False
    full_logging: bool                     = False
    manager_logging: bool                  = False
    overwrite_existing: bool               = False
    custom_poll_interval: float            = None

    # SSH / remote execution options
    over_ssh: bool                         = False
    ssh_target: Optional[str]              = None
    remote_work_root: Optional[Path]       = None
    remote_pullback_policy: Remote_Pullback_Policy = Remote_Pullback_Policy.STANDARD
    pull_rasorb: bool                      = False
    cleanup_remote: bool                   = True 

class Job_Manager:

    def __init__(
        self,
        executor_type: Executor_Type,
        execution_script: str | Path,
        extraction_script: str | Path,
        group_dir_name: Optional[str] = None,   
        *,
        group_dir_path: str | Path  = None,  # keyword-only full path override
        auto_run: bool              = False,
        full_logging: bool          = False,
        manager_logging: bool       = False,
        overwrite_existing: bool    = False,
        custom_poll_interval: float = None,

        # SSH / remote execution options
        over_ssh: bool                         = False,
        ssh_target: Optional[str]              = None,
        remote_work_root: Optional[str | Path] = None,
        remote_pullback_policy: Remote_Pullback_Policy = Remote_Pullback_Policy.STANDARD,
        pull_rasorb: bool                      = False,
        cleanup_remote: bool                   = True,
    ):

        self.execution_script = Path(execution_script).resolve()
        if not self.execution_script.exists():
            raise FileNotFoundError(
                f"Execution script not found: {self.execution_script}"
            )
        
        self.extraction_script = Path(extraction_script).resolve()
        if not self.extraction_script.exists():
            raise FileNotFoundError(
                f"Extraction script not found: {self.extraction_script}"
            )
        self.base_dir                     = self.execution_script.parent

        self.auto_run: bool               = auto_run
        self.jobs: List[Molcas_Job]       = []
        self.job_counter: int             = 0
        # A job's name doubles as the MOLCAS Project, which keys its scratch dir
        # ($MOLCAS_WORKDIR/$Project). Concurrent managers (one per shell, plus
        # global evals) would otherwise all mint "job_0" and collide in the
        # shared scratch root. This per-manager uuid prefix makes every job name
        # unique across all managers, processes, and machines.
        self._job_token: str              = uuid4().hex
        self.full_logging: bool           = full_logging
        self.manager_logging: bool        = manager_logging
        self.overwrite_existing: bool     = overwrite_existing
        self.executor_type: Executor_Type = executor_type
        self.batched_execution: bool      = executor_type in BATCHED_EXECUTORS
        self.global_poll_interval: float  = custom_poll_interval if custom_poll_interval is not None else 5.0
        self.all_jobs_ran: bool           = False

        self.over_ssh               = over_ssh
        self.ssh_target             = ssh_target
        self.remote_work_root       = Path(remote_work_root) if remote_work_root is not None else None
        self.remote_pullback_policy = remote_pullback_policy
        self.pull_rasorb            = pull_rasorb
        self.cleanup_remote         = cleanup_remote

        self._validate_remote_settings()

        # Set executor
        self.executor = self.get_builtin_executor(executor_type)
        self.group_dir = self._resolve_group_dir(group_dir_name, group_dir_path)

        if self.group_dir.exists():
            if self.overwrite_existing:
                resolved = self.group_dir.resolve()

                # ---- SAFETY GUARD ----
                if len(resolved.parts) < 3:
                    raise RuntimeError(
                        f"Refusing to delete shallow directory: {resolved}"
                    )

                shutil.rmtree(resolved)
            else:
                raise FileExistsError(
                    f"Run directory '{self.group_dir}' already exists. "
                    "Set overwrite_existing=True to overwrite it."
            )

        # Create the directory
        self.group_dir.mkdir(parents=True)

    def _resolve_group_dir(self, group_dir_name: Optional[str], group_dir_path: Optional[str | Path]) -> Path:
        if group_dir_path is not None:
            return Path(group_dir_path).resolve()

        base_name = group_dir_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        return (self.base_dir / base_name).resolve()
    
    def get_builtin_executor(self, executor_type: Executor_Type ):
        try:
            return executor_map[executor_type]
        except KeyError:
            raise ValueError(f"Unsupported executor type: {executor_type}")

    def _validate_remote_settings(self):
        if self.executor_type not in REMOTE_EXECUTORS:
            return

        if not self.over_ssh:
            raise ValueError(
                f"Executor type {self.executor_type.value} requires over_ssh=True."
            )

        if not self.ssh_target:
            raise ValueError("When using a remote executor, ssh_target must be provided.")

        if self.remote_work_root is None:
            raise ValueError("When using a remote executor, remote_work_root must be provided.")

    def add_job(self, exponent_set, template_path, *, name: Optional[str] = None, this_log: bool = False, rasorbs=None):

        log_flag = True if (this_log or self.full_logging) else False

        if name:
            job_name = name
        else:
            job_name = f"job_{self._job_token}_{self.job_counter}"

        job_dir = self.group_dir / job_name

        if job_dir.exists():
            raise ValueError(f"Job directory '{job_dir}' already exists.")

        job = Molcas_Job(
            job_id        = self.job_counter,
            job_dir       = job_dir,
            template_path = template_path,
            extract_path  = self.extraction_script,
            exponent_set  = exponent_set,
            name          = name,
            input_name    = job_name,
            logging       = log_flag,
            rasorbs       = rasorbs,
        )

        job.prepare_job()
        self.jobs.append(job)
        self.job_counter += 1

        return job
    
    def submit_single(self, job: Molcas_Job):
        handle = self.executor(
            job,
            self.execution_script,
            over_ssh               = self.over_ssh,
            ssh_target             = self.ssh_target,
            remote_work_root       = self.remote_work_root,
            remote_pullback_policy = self.remote_pullback_policy,
            pull_rasorb            = self.pull_rasorb,
            cleanup_remote         = self.cleanup_remote,
        )

        if not isinstance(handle, Handle):
            raise RuntimeError(
                f"Executor did not return a valid Handle for job {job.job_id}"
            )

        job.handle = handle
        job.mark_submitted()
        return handle
    
    def submit_batch(self, jobs: List[Molcas_Job]):
        handle = self.executor(
            jobs,
            self.execution_script,
            over_ssh               = self.over_ssh,
            ssh_target             = self.ssh_target,
            remote_work_root       = self.remote_work_root,
            remote_pullback_policy = self.remote_pullback_policy,
            pull_rasorb            = self.pull_rasorb,
            cleanup_remote         = self.cleanup_remote,
        )

        if not isinstance(handle, Handle):
            raise RuntimeError(
                f"Executor did not return a valid Handle for batch job"
            )
        
        for job in jobs:
            job.mark_submitted()

        return handle

    def _finalize_run_summary(self):
        failed_jobs = [job for job in self.jobs if job.status == Job_Status.FAILED]

        if failed_jobs:
            if self.manager_logging:
                print(f"[JobManager] {len(failed_jobs)} job(s) failed.")
        else:
            if self.manager_logging:
                print("[JobManager] All jobs completed successfully.")

        self.all_jobs_ran = True

    def _run_all_jobs_serial(self, max_jobs: int = 1, *, poll_interval_override: float | None = None):
        poll_interval = poll_interval_override if poll_interval_override is not None else self.global_poll_interval if self.global_poll_interval is not None else 5.0 
        running_jobs  = []

        while True:
            for job in self.jobs:
                if len(running_jobs) >= max_jobs:
                    break

                if job.status == Job_Status.PREPARED:
                    self.submit_single(job)
                    running_jobs.append(job)
                    if job.logging or self.manager_logging:
                        print(f"[JobManager] Submitted job '{job.job_id}'")

            for job in running_jobs[:]:  # iterate over copy since we may remove
                if job.handle.is_finished():
                    if job.logging or self.manager_logging:
                        print(f"[JobManager] Submitted job '{job.job_id}' is finished")
                    job.update_from_output()
                    running_jobs.remove(job)

            all_done = all(job.status in (Job_Status.COMPLETED, Job_Status.FAILED) for job in self.jobs)
            if all_done:
                break

            time.sleep(poll_interval)

    def _run_all_jobs_batched(self, max_jobs: int = 1, *, poll_interval_override: float | None = None):
        poll_interval = poll_interval_override if poll_interval_override is not None else self.global_poll_interval if self.global_poll_interval is not None else 5.0 
        batch_running = False
        batch_handle  = None

        current_batch_jobs = []

        while True:
            if not batch_running:
                pending_jobs = [job for job in self.jobs if job.status == Job_Status.PREPARED]

                if pending_jobs:
                    current_batch_jobs = pending_jobs[:max_jobs]
                    batch_handle       = self.submit_batch(current_batch_jobs)

                    if self.manager_logging:
                        print(f"[JobManager] Submitted batch of {len(current_batch_jobs)} jobs.")

                    batch_running = True
            
            else:
                if batch_handle.is_finished():
                    if self.manager_logging:
                        print(f"[JobManager] Batch of {len(current_batch_jobs)} jobs finished.")

                    for job in current_batch_jobs:
                        job.update_from_output()

                    current_batch_jobs = []
                    batch_handle       = None
                    batch_running      = False

            all_done = all(job.status in (Job_Status.COMPLETED, Job_Status.FAILED) for job in self.jobs)
            if all_done:
                break

            time.sleep(poll_interval)

    def run_all_jobs(self, max_jobs: int = 1, *, poll_interval_override: float | None = None):

        if max_jobs < 1:
            raise ValueError("max_jobs must be at least 1.")

        if self.batched_execution:
            self._run_all_jobs_batched(max_jobs, poll_interval_override=poll_interval_override)
        else:
            self._run_all_jobs_serial(max_jobs, poll_interval_override=poll_interval_override)
        
        self._finalize_run_summary()

    def collect_successful_results(self, custom_path_for_expo: Optional[str | Path] = None):
        if not getattr(self, "all_jobs_ran", False):
            raise RuntimeError("Cannot collect results before all jobs have been run.")
        
        # Determine the results directory
        if custom_path_for_expo is not None:
            results_dir = Path(custom_path_for_expo).resolve()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = self.group_dir / f"Run_results_{timestamp}"
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        collected_files = 0

        for job in self.jobs:
            if job.status == Job_Status.COMPLETED:
                job.save_exponent_file(results_dir)
                collected_files += 1
        
        if self.full_logging:
            print(f"Collected {collected_files} results into {results_dir}")

    def get_successful_exponents(self) -> list:

        if not getattr(self, "all_jobs_ran", False):
            raise RuntimeError("Cannot collect exponents before all jobs have been run.")

        successful_exponents = [
            job.exponent_set
            for job in self.jobs
            if job.status == Job_Status.COMPLETED
        ]

        if self.full_logging:
            print(f"Collected {len(successful_exponents)} successful Exponent_Set objects.")

        return successful_exponents
    
    def copy_without_jobs(
    self,
    new_group_dir_name: Optional[str] = None,
    *,
    overwrite_existing: bool = False
) -> "Job_Manager":
        """
        Returns a copy of this Job_Manager with identical configuration
        but without jobs.

        Behavior:
        - If new_group_dir_name is None and overwrite_existing is False:
            → Creates a new timestamped directory.
        - If new_group_dir_name is None and overwrite_existing is True:
            → Reuses current directory name and overwrites it.
        - If new_group_dir_name is provided:
            → Uses that directory name (overwrite behavior respected).
        """
        if new_group_dir_name is None:
            if overwrite_existing:
                # Reuse current directory name
                group_dir_name_to_use = self.group_dir.name
            else:
                # Let _resolve_group_dir auto-generate timestamp
                group_dir_name_to_use = None
        else:
            group_dir_name_to_use = new_group_dir_name

        # Create new Job_Manager instance
        copy_manager = Job_Manager(
            executor_type        = self.executor_type,
            execution_script     = self.execution_script,
            extraction_script    = self.extraction_script,
            group_dir_name       = group_dir_name_to_use,
            group_dir_path       = None,
            auto_run             = self.auto_run,
            full_logging         = self.full_logging,
            manager_logging      = self.manager_logging,
            overwrite_existing   = overwrite_existing,
            custom_poll_interval = self.global_poll_interval,

            over_ssh                = self.over_ssh,
            ssh_target              = self.ssh_target,
            remote_work_root        = self.remote_work_root,
            remote_pullback_policy  = self.remote_pullback_policy,
            pull_rasorb             = self.pull_rasorb,
            cleanup_remote          = self.cleanup_remote,
        )

        # Clear jobs and reset counter
        copy_manager.jobs         = []
        copy_manager.job_counter  = 0
        copy_manager.all_jobs_ran = False

        return copy_manager
    
    @classmethod
    def from_config(cls, config: Job_Manager_Config) -> "Job_Manager":
        return cls(
            executor_type        = config.executor_type,
            execution_script     = config.execution_script,
            extraction_script    = config.extraction_script,
            group_dir_name       = config.group_dir_name,
            group_dir_path       = config.group_dir_path,
            auto_run             = config.auto_run,
            full_logging         = config.full_logging,
            manager_logging      = config.manager_logging,
            overwrite_existing   = config.overwrite_existing,
            custom_poll_interval = config.custom_poll_interval,

            over_ssh                = config.over_ssh,
            ssh_target              = config.ssh_target,
            remote_work_root        = config.remote_work_root,
            remote_pullback_policy  = config.remote_pullback_policy,
            pull_rasorb             = config.pull_rasorb,
            cleanup_remote          = config.cleanup_remote,
        )
        
        
     



