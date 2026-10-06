"""Mechanics shared by the Paper_tests scripts.

Only things that were byte-identical across several of them live here: input staging, the
run log, the local job config, the padded progress line, the bootstrap contraction, and the
CMA-run-with-live-trace loop. Each test keeps its own USER CONFIGURATION block, its own
statistics and its own reporting, because that is the part a reader needs to see in one
place to judge the test.
"""

import shutil
import time
from pathlib import Path

from ..exponent_handler import Exponent_Set
from ..job_manager import Job_Manager_Config
from ..objectives import Ground_Energy_Objective
from ..common import Executor_Type
from ..cma_opt_2 import evaluate_initial
from ..cma_shell_opt import Shell_Optimization


# In-place progress lines are padded so a shorter line fully overwrites a longer one, and
# truncated so one line never exceeds a terminal row — a wrapped line makes \r return to
# the start of the wrapped remainder rather than the start of the line, which is what makes
# the updates pile up instead of overwriting.
PROGRESS_W = max(40, shutil.get_terminal_size((100, 24)).columns - 1)


def progress(msg):
    print(msg[:PROGRESS_W].ljust(PROGRESS_W), end="\r", flush=True)


def clear_progress():
    print(" " * PROGRESS_W, end="\r", flush=True)


class Counting_Objective(Ground_Energy_Objective):
    # exact job count: every MOLCAS job goes through evaluate_batch, so counting the batch
    # sizes here beats deriving it from generation counts or trusting scipy's nfev (which
    # counts calls, not the jobs each call costs)
    def __init__(self, template_file, manager_cfg=None):
        super().__init__(template_file, manager_cfg)
        self.n_jobs = 0

    def evaluate_batch(self, exps, **kwargs):
        self.n_jobs += len(exps)
        return super().evaluate_batch(exps, **kwargs)


class Log:
    """Mirrors every line to stdout and to a file, flushed per line so a crashed run keeps
    everything it had already printed."""

    def __init__(self, path):
        self._f = open(path, "w")

    def emit(self, msg=""):
        print(msg, flush=True)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def stage_inputs(submit_dir, start_dir, names):
    """Copy each named input from the submit dir into the run's Start dir, so the run reads
    from scratch and never from shared storage. Returns the staged paths in order."""
    start_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(len(names)):
        dst = start_dir / names[i]
        shutil.copy(submit_dir / names[i], dst)
        out.append(dst)
    return out


def local_job_config(run_scr, extract_scr, poll_sec=0.1):
    """Local-bash job config. poll_sec matters: the Job_Manager default is 5.0s, which
    quantises every batch to a multiple of it and then dominates all measured wall clock,
    hitting multi-round batches hardest (the loop costs 2*rounds-1 poll cycles)."""
    return Job_Manager_Config(
        executor_type        = Executor_Type.LOCAL_BASH,
        execution_script     = run_scr,
        extraction_script    = extract_scr,
        overwrite_existing   = True,
        custom_poll_interval = poll_sec,
    )


def bootstrap_contraction(basis, full_objective, work_dir, threads=1, subdir_name="bootstrap"):
    """One fully uncontracted run at the starting basis, which is where the GENANO
    contraction under test comes from. Returns (base_contracted, base_uncontracted, energy).
    Raises if the template produced no contraction, since every caller needs one."""
    boot = evaluate_initial(basis, full_objective, work_dir, threads=threads,
                            subdir_name=subdir_name)
    if boot.resulting_contraction is None:
        raise RuntimeError(f"bootstrap produced no contraction — does the template run "
                           f"GENANO? (work_dir {work_dir})")
    base_contracted = boot.copy(no_energy=True)
    base_contracted.change_contraction(boot.resulting_contraction)
    base_uncontracted = boot.copy(no_energy=True)
    base_uncontracted.uncontract_all()
    return base_contracted, base_uncontracted, float(boot.energy)


def run_cma_shell(start_exp, start_energy, objective, work_dir, shell, *,
                  generation_size, sigma, max_generations, contract_frozen_shells,
                  use_tempering, n_tempering_params, use_stopping, stop_tol, seed,
                  threads, poll_sec=0.05, on_generation=None, crash_label=""):
    """One CMA-ES run on one shell, draining the optimizer's history live.

    on_generation(h, elapsed) is called once per generation with that generation's history
    dict, so the caller can write its own trace rows and progress line without this helper
    having to know their shape.

    poll_sec only gates how often the history is drained — the optimizer never waits on it.
    Keep it small: each drained row is stamped when the loop NOTICES the generation, so a
    slow poll inflates the recorded times, and only ever in one direction.

    Returns {best_exp, best_energy, gens, jobs, wall, hit_cap}.
    """
    opt = Shell_Optimization(
        start_exp, start_energy, objective,
        work_dir               = work_dir,
        generation_size        = generation_size,
        sigma                  = sigma,
        max_generations        = max_generations,
        active_shell           = shell,
        overwrite              = True,
        logging                = False,
        contract_frozen_shells = contract_frozen_shells,
        use_tempering          = use_tempering,
        n_tempering_params     = n_tempering_params,
        use_stopping           = use_stopping,
        stop_tol               = stop_tol,
        seed                   = seed,
    )

    t0      = time.time()
    drained = 0
    opt.start(threads=threads)
    while True:
        running = opt.is_running
        history = opt.history
        for i in range(drained, len(history)):
            if on_generation is not None:
                on_generation(history[i], time.time() - t0)
        drained = len(history)
        if not running:
            break
        time.sleep(poll_sec)
    opt.wait()
    clear_progress()

    if opt.exception is not None:
        raise RuntimeError(f"CMA crashed{(' (' + crash_label + ')') if crash_label else ''}") \
            from opt.exception

    state = opt.get_state()
    gens  = max(state["generation"] + 1, 0)
    best  = float(state["best_energy"]) if state["best_energy"] is not None else float(start_energy)
    return {
        "best_exp":    state["best_exp"],
        "best_energy": best,
        "gens":        gens,
        "jobs":        gens * generation_size,
        "wall":        time.time() - t0,
        "hit_cap":     gens >= max_generations,
    }
