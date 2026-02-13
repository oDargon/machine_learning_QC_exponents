from pathlib import Path
from typing import Optional, Callable, List
from enum import Enum
import subprocess
from datetime import datetime
from molcas_handler import *
from handles import *
import time


class ExecutorType(Enum):
    LOCAL_BASH = "local_bash"
    SLURM      = "slurm"





class Job_Manager:

    def __init__(
        self,
        executor_type: ExecutorType,
        execution_script: str,
        group_dir: Optional[str] = None,
        *,
        auto_run: bool                      = False,
        custom_executor: Optional[Callable] = None,
        full_logging: bool                  = False
    ):
        
        self.execution_script       = execution_script
        self.auto_run               = auto_run
        self.jobs: List[Molcas_Job] = []
        self.job_counter            = 0
        self.executor               = None
        self.full_logging: bool     = full_logging

        if custom_executor:
            self.executor = custom_executor
        else:
            self.executor = self.get_builtin_executor(executor_type)

        if group_dir:
            self.group_dir = Path(group_dir)
        else:
            self.group_dir = self.create_group_dir()

        self.group_dir.mkdir(parents=True, exist_ok=True)

        

    def create_group_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"group_{timestamp}")
    
    def get_builtin_executor(self, type: ExecutorType ):
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

        self.job_counter += 1

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

        return job
    
    def run_job(self, job: Molcas_Job):
        handle     = self.executor(job, self.execution_script)
        job.handle = handle
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
                # print(job.status)
                if job.status == Job_Status.PREPARED and len(running_jobs) < max_jobs:
                    job.handle = self.executor(job, self.execution_script)
                    job.mark_submitted()
                    running_jobs.append(job)
                    if job.logging:
                        print(f"[JobManager] Submitted job '{job.job_id}'")

            # Poll running jobs
            for job in running_jobs[:]:  # iterate over copy since we may remove
                if job.handle.is_finished():
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
            print(f"[JobManager] {len(failed_jobs)} job(s) failed.")
        else:
            print("[JobManager] All jobs completed successfully.")



    






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
    process = subprocess.Popen(
        ["bash", script_path.name],
        cwd=job.job_dir,
        stdout=open(job.output_file, "w"),
        stderr=subprocess.STDOUT,
    )

    # Wrap in BashHandle before returning
    return Bash_Handle(process)

def slurm_executor(job: Molcas_Job, script_template_path):
    return None



executor_map = {
    ExecutorType.LOCAL_BASH: local_bash_executor,
    ExecutorType.SLURM:      slurm_executor,
}