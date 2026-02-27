from pathlib import Path
from typing import Optional, Callable, List
from enum import Enum
import subprocess
from dataclasses import dataclass
from datetime import datetime
from molcas_handler import *
from handles import *
import time
import shutil







class Executor_Type(Enum):
    LOCAL_BASH = "local_bash"
    SLURM      = "slurm"


@dataclass
class Job_Manager_Config:
    executor_type: Optional[Executor_Type] = None
    execution_script: Optional[Path]       = None
    group_dir_name: Optional[str]          = None
    group_dir_path: Optional[Path]         = None
    auto_run: bool                         = False
    custom_executor: Optional[Callable]    = None
    full_logging: bool                     = False
    manager_logging: bool                  = False
    overwrite_existing: bool               = False

class Job_Manager:

    def __init__(
        self,
        executor_type: Executor_Type,
        execution_script: str | Path,
        group_dir_name: Optional[str] = None,   
        *,
        group_dir_path: Optional[str | Path] = None,  # keyword-only full path override
        auto_run: bool                       = False,
        custom_executor: Optional[Callable]  = None,
        full_logging: bool                   = False,
        manager_logging: bool                = False,
        overwrite_existing: bool             = False
    ):

        self.execution_script = Path(execution_script).resolve()
        if not self.execution_script.exists():
            raise FileNotFoundError(
                f"Execution script not found: {self.execution_script}"
            )
        self.base_dir                 = self.execution_script.parent

        self.auto_run: bool           = auto_run
        self.jobs: List[Molcas_Job]   = []
        self.job_counter: int         = 0
        self.full_logging: bool       = full_logging
        self.manager_logging: bool    = manager_logging
        self.overwrite_existing: bool = overwrite_existing
        self.all_jobs_ran: bool       = False

        # Set executor
        if custom_executor:
            self.executor = custom_executor
        else:
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
    
    def get_builtin_executor(self, type: Executor_Type ):
        try:
            return executor_map[type]
        except KeyError:
            raise ValueError(f"Unsupported executor type: {type}")


    def add_job(self, exponent_set, template_path, *, name: Optional[str] = None, this_log: bool = False):

        log_flag = True if (this_log or self.full_logging) else False

        if name:
            job_name = name
        else:
            job_name = f"job_{self.job_counter}"

        job_dir = self.group_dir / job_name

        if job_dir.exists():
            raise ValueError(f"Job directory '{job_dir}' already exists.")

        job = Molcas_Job(
            job_id        = self.job_counter,
            job_dir       = job_dir,
            template_path = template_path,
            exponent_set  = exponent_set,
            name          = name,
            logging       = log_flag   
        )

        job.prepare_job()
        self.jobs.append(job)
        self.job_counter += 1

        return job
    
    def run_job(self, job: Molcas_Job):
        handle = self.executor(job, self.execution_script)

        if not isinstance(handle, Handle):
            raise RuntimeError(
                f"Executor did not return a valid Handle for job {job.job_id}"
            )

        job.handle = handle
        job.mark_submitted()
        return handle
    
    def run_all_jobs(self, max_jobs: int = 1, poll_interval: float = 1.0):
        """
        Run jobs managed by this Job_Manager.

        max_jobs: maximum number of jobs to run concurrently.
        poll_interval: seconds to wait between polling cycles.
        """
        running_jobs = []

        while True:
            # Check for jobs that can be submitted
            for job in self.jobs:
                if len(running_jobs) >= max_jobs:
                    break

                # print(job.status)
                if job.status == Job_Status.PREPARED:
                    self.run_job(job)
                    running_jobs.append(job)
                    if job.logging or self.manager_logging:
                        print(f"[JobManager] Submitted job '{job.job_id}'")

            # Poll running jobs
            for job in running_jobs[:]:  # iterate over copy since we may remove
                if job.handle.is_finished():
                    if job.logging or self.manager_logging:
                        print(f"[JobManager] Submitted job '{job.job_id}' is finished")
                    job.update_from_output()
                    running_jobs.remove(job)

            # Break if no more jobs pending or running
            all_done = all(job.status in (Job_Status.COMPLETED, Job_Status.FAILED) for job in self.jobs)
            if all_done:
                break

            # Sleep between polls
            time.sleep(poll_interval)

        # Summary
        failed_jobs = [job for job in self.jobs if job.status == Job_Status.FAILED]
        if failed_jobs:
            if self.manager_logging:
                print(f"[JobManager] {len(failed_jobs)} job(s) failed.")
        else:
            if self.manager_logging:
                print("[JobManager] All jobs completed successfully.")

        self.all_jobs_ran = True

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
            executor_type      =None,  # We'll override executor manually
            execution_script   =self.execution_script,
            group_dir_name     =group_dir_name_to_use,
            group_dir_path     =None,
            auto_run           =self.auto_run,
            custom_executor    =self.executor,
            full_logging       =self.full_logging,
            manager_logging    =self.manager_logging,
            overwrite_existing =overwrite_existing
        )

        # Clear jobs and reset counter
        copy_manager.jobs         = []
        copy_manager.job_counter  = 0
        copy_manager.all_jobs_ran = False

        return copy_manager
    
    @classmethod
    def from_config(cls, config: Job_Manager_Config) -> "Job_Manager":
        return cls(
            executor_type      = config.executor_type,
            execution_script   = config.execution_script,
            group_dir_name     = config.group_dir_name,
            group_dir_path     = config.group_dir_path,
            auto_run           = config.auto_run,
            custom_executor    = config.custom_executor,
            full_logging       = config.full_logging,
            manager_logging    = config.manager_logging,
            overwrite_existing = config.overwrite_existing
        )
        
        
        



    






def local_bash_executor(job: Molcas_Job, script_template_path):
    """
    Submits a local bash job for the given Molcas_Job.
    Returns a BashHandle.
    """
    template_path = Path(script_template_path)

    # Read template
    with open(template_path, "r") as f:
        script_content = f.read()

    # Replace placeholder with full input file path
    script_content = script_content.replace(
        "{{JOB_NAME}}",
        str(job.input_file)
    )

    # Write job-specific script
    script_path = job.job_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)

    # Launch process
    output_handle = open(job.output_file, "w")

    process = subprocess.Popen(
        ["bash", script_path.name],
        cwd=job.job_dir,
        stdout=output_handle,
        stderr=subprocess.STDOUT,
    )

    return Bash_Handle(process, output_handle)

def slurm_executor(job: Molcas_Job, script_template_path):
    return None



executor_map = {
    Executor_Type.LOCAL_BASH: local_bash_executor,
    Executor_Type.SLURM:      slurm_executor,
}