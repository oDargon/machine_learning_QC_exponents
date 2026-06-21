import sys
import csv
import time
import yaml
import shutil
import subprocess
from pathlib import Path
from numpy import array, eye, ones, trace
from numpy.linalg import solve, LinAlgError

WORK_DIR   = Path.cwd() / "Optimization"
SUBMIT_DIR = Path.cwd()
sys.path.insert(0, str(SUBMIT_DIR))

from evo_opt.exponent_handler import Exponent_Set
from evo_opt.objectives import Ground_Energy_Objective
from evo_opt.job_manager import Job_Manager, Job_Manager_Config
from evo_opt.common import Executor_Type, Job_Status
from evo_opt.cma_opt_2 import evaluate_initial


def anderson_extrapolate(history, depth):
    """
    Type-I Anderson extrapolation over a shell's own raw (unaccelerated)
    per-cycle result history. Returns None if there isn't enough history yet.

    history: list of ndarray, oldest first - history[k] is that shell's best
             exponents as of the end of cycle k (or the bootstrap value at
             index 0).
    depth:   number of residual vectors (steps back) to mix.
    """
    if depth < 1 or len(history) < depth + 1:
        return None

    X = array(history[-(depth + 1):])   # depth+1 points
    F = X[1:] - X[:-1]                  # depth residuals f_i = x_{i+1} - x_i

    # minimize ||F^T @ alpha||^2 s.t. sum(alpha) = 1, via Lagrange multipliers:
    # alpha = G^-1 1 / (1^T G^-1 1), G = F F^T, regularized for stability when
    # recent residuals are nearly collinear (common with a stochastic inner
    # optimizer like CMA-ES).
    G   = F @ F.T
    reg = 1e-10 * (trace(G) / depth if trace(G) > 0 else 1.0)
    try:
        w = solve(G + reg * eye(depth), ones(depth))
    except LinAlgError:
        return None

    if not w.any():
        return None

    alpha = w / w.sum()
    return alpha @ X[1:]

exp_path      = SUBMIT_DIR / "Si.expo"
template      = SUBMIT_DIR / "template.inp"
template_full = SUBMIT_DIR / "template_full.inp"   # uncontracted + GENANO
submit_scr    = SUBMIT_DIR / "run.sh"
extract_scr   = SUBMIT_DIR / "extract.sh"

CORES_PER_SHELL    = [12, 11, 7, 5, 1]
GEN_SIZE_PER_SHELL = [12, 11, 7, 5, 1]
SIGMA              = 0.01
MAX_GENERATIONS    = 10
CYCLES             = 5
SHELLS_TO_OPTIMIZE = [0, 1, 2, 3]
PROPAGATE_CMA      = True
USE_STOPPING       = False

PROPAGATE_FULL_CONTRACTION = True
USE_ANDERSON          = True
ANDERSON_DEPTH        = 2   # residual vectors (steps back) to mix
ANDERSON_START_CYCLE  = 3   # earliest cycle index (0-indexed) allowed to extrapolate

L_LABELS = ["s", "p", "d", "f", "g", "h"]

DATA_DIR = WORK_DIR / "Data"
FULL_DIR = WORK_DIR / "Full"   # owned by full_manager below, it creates this itself
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

for j in SHELLS_TO_OPTIMIZE:
    (DATA_DIR / f"shell_{j}").mkdir(parents=True, exist_ok=True)

exp = Exponent_Set.from_file(exp_path)
n_shells = len(exp.exponents)

cfg = Job_Manager_Config(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = WORK_DIR,
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)
objective      = Ground_Energy_Objective(template,      cfg)
full_objective = Ground_Energy_Objective(template_full, cfg)

# Bootstrap: two jobs before any cycle runs.
#   1) fully uncontracted, on the GENANO template -> the very first contraction.
#   2) the same exponents, now contracted, on the cheap template -> the real
#      starting energy in the regime every subsequent per-shell run lives in.
init_uncontracted = evaluate_initial(exp, full_objective, WORK_DIR / "initial_uncontracted", threads=1)

if init_uncontracted.resulting_contraction is None:
    raise RuntimeError("Initial uncontracted run produced no contraction (no .ANO file found).")

global_state = init_uncontracted.copy(no_energy=True)
global_state.change_contraction(init_uncontracted.resulting_contraction)

init_contracted = evaluate_initial(
    global_state, objective, WORK_DIR / "initial_contracted",
    threads=1, contract_frozen_shells=True,
)
start_energy = init_contracted.energy

# Each shell's own freshest optimized primitives. Always inherited forward
# from that shell's own previous optimization step, independent of the
# (lagging) global contraction snapshot below.
latest_own = [shell.copy() for shell in exp.exponents]

# Each shell's own previous result (full-atom contracted energy from the last
# time that specific shell was optimized). This is the meaningful "before"
# baseline per shell now - there's no single coherent whole-atom "current
# energy" thread anymore, since other shells sit at a lagging snapshot during
# any one shell's run.
last_shell_energy = [start_energy] * n_shells

# Each optimized shell's own raw result history, oldest first, seeded with
# the bootstrap value. Anderson extrapolates from this directly - it never
# looks at global_state/contraction at all.
shell_history = {j: [latest_own[j].copy()] for j in SHELLS_TO_OPTIMIZE}

# Background full-run pipeline. A new full run is launched at the end of
# every cycle unconditionally, so more than one can be in flight if a single
# full run takes longer than a cycle. Each is polled (never blocked on) at
# the top of every cycle; if several have landed by the time we check, only
# the newest is applied to global_state and the older (superseded) ones are
# dropped without being acted on.
#
# Raw Job_Manager is used here instead of Ground_Energy_Objective, since
# evaluate_batch()/run_all_jobs() block until completion - we need to fire a
# job and poll it ourselves. One manager is reused for every full job; it
# gives each job its own subdirectory under FULL_DIR automatically.
full_manager = Job_Manager(
    Executor_Type.LOCAL_BASH,
    submit_scr,
    extract_scr,
    group_dir_path       = FULL_DIR,
    overwrite_existing   = True,
    custom_poll_interval = 0.1,
)

full_jobs         = []
full_run_energies = [float(init_uncontracted.energy)]   # energy of every completed full run, free byproduct

SEP           = "=" * 72
LOG_FILE      = WORK_DIR / "cyclic_log.txt"
CSV_FILE      = WORK_DIR / "cyclic_log.csv"
FULL_LOG_FILE = WORK_DIR / "full_log.txt"
FULL_CSV_FILE = WORK_DIR / "full_log.csv"

print(f"Log      : {LOG_FILE}")
print(f"CSV      : {CSV_FILE}")
print(f"Full log : {FULL_LOG_FILE}")
print(f"Full CSV : {FULL_CSV_FILE}")

# Per-shell log: each row compares a shell's result against that same shell's
# own previous run, not against any whole-atom running total.
CSV_HEADER = [
    "cycle", "shell", "shell_label", "n_exponents", "popsize", "max_generations",
    "E_before", "E_after", "dE", "exp_change_pct",
]

# Full-run log: this is the one series safe to read as true cumulative
# whole-atom progress - same (uncontracted + GENANO) method every time, built
# from every shell's freshest value, no staleness.
FULL_CSV_HEADER = ["order", "launch_cycle", "landed_cycle", "energy", "dE", "dE_total", "wall_time_sec", "cumulative_wall_time_sec"]

with open(LOG_FILE, "w") as log, open(CSV_FILE, "w", newline="") as csv_f, \
     open(FULL_LOG_FILE, "w") as full_log, open(FULL_CSV_FILE, "w", newline="") as full_csv_f:

    csv_writer      = csv.writer(csv_f)
    full_csv_writer = csv.writer(full_csv_f)
    csv_writer.writerow(CSV_HEADER)
    full_csv_writer.writerow(FULL_CSV_HEADER)
    csv_f.flush()

    cumulative_wall_time = 0.0

    full_csv_writer.writerow([
        0, "bootstrap", "bootstrap", f"{full_run_energies[0]:.10f}", f"{0.0:.10f}", f"{0.0:.10f}", "NA", f"{cumulative_wall_time:.2f}",
    ])
    full_csv_f.flush()

    full_log.write(f"{SEP}\n")
    full_log.write("Bootstrap full run (uncontracted, order 0)\n")
    full_log.write(f"  E        = {full_run_energies[0]:20.10f} Eh\n")
    full_log.write(f"{SEP}\n\n")
    full_log.flush()

    for i in range(CYCLES):

        # ---- consume any landed full runs: scan oldest -> newest, apply only the newest one that's done ----
        applied_idx = None
        for idx in range(len(full_jobs)):
            entry_scan = full_jobs[idx]
            full_job   = entry_scan["job"]
            if not full_job.handle.is_finished():
                continue

            full_job.update_from_output()
            if full_job.status == Job_Status.COMPLETED:
                energy = float(full_job.exponent_set.energy)
                full_run_energies.append(energy)

                order     = len(full_run_energies) - 1
                dE        = energy - full_run_energies[order - 1]
                dE_total  = energy - full_run_energies[0]
                wall_time = time.time() - entry_scan["launch_time"]

                cumulative_wall_time += wall_time

                full_csv_writer.writerow([
                    order, entry_scan["launch_cycle"] + 1, i + 1,
                    f"{energy:.10f}", f"{dE:.10f}", f"{dE_total:.10f}", f"{wall_time:.2f}", f"{cumulative_wall_time:.2f}",
                ])
                full_csv_f.flush()

                full_log.write(f"{SEP}\n")
                full_log.write(f"Full run order {order}  |  launched at cycle {entry_scan['launch_cycle'] + 1}  |  landed at cycle {i + 1}\n")
                full_log.write(f"  E         = {energy:20.10f} Eh\n")
                full_log.write(f"  dE        = {dE:+20.10f} Eh\n")
                full_log.write(f"  dE_total  = {dE_total:+20.10f} Eh\n")
                full_log.write(f"  wall_time = {wall_time:19.2f} s\n")
                full_log.write(f"  cum_time  = {cumulative_wall_time:19.2f} s\n")
                full_log.write(f"{SEP}\n\n")
                full_log.flush()

            applied_idx = idx

        if applied_idx is not None:
            entry    = full_jobs[applied_idx]
            full_job = entry["job"]
            full_jobs = full_jobs[applied_idx + 1:]   # older (and this) entries are superseded, drop them

            if not PROPAGATE_FULL_CONTRACTION:
                msg = f"[Full] run launched at cycle {entry['launch_cycle'] + 1} landed -> contraction propagation disabled, global state unchanged"
            elif full_job.status == Job_Status.COMPLETED and full_job.exponent_set.resulting_contraction is not None:
                new_state = full_job.exponent_set.copy(no_energy=True)
                new_state.change_contraction(full_job.exponent_set.resulting_contraction)
                global_state = new_state

                msg = f"[Full] run launched at cycle {entry['launch_cycle'] + 1} landed -> global contraction refreshed, E = {full_job.exponent_set.energy:.10f} Eh"
            else:
                msg = f"[Full] run launched at cycle {entry['launch_cycle'] + 1} FAILED or produced no contraction -> global state unchanged"

            print(msg)
            full_log.write(msg + "\n\n")
            full_log.flush()

        # ---- Anderson: compute every optimized shell's extrapolated guess up front ----
        extrapolated = {}
        for j in SHELLS_TO_OPTIMIZE:
            extrapolated[j] = (
                anderson_extrapolate(shell_history[j], ANDERSON_DEPTH)
                if USE_ANDERSON and i >= ANDERSON_START_CYCLE else None
            )

        for j in SHELLS_TO_OPTIMIZE:
            lbl      = L_LABELS[j] if j < len(L_LABELS) else str(j)
            n_exp    = len(latest_own[j])
            gen_size = GEN_SIZE_PER_SHELL[j]

            print(f"{SEP}")
            print(f"  Cycle {i + 1}/{CYCLES}  |  Shell {j} ({lbl})  |  {n_exp} exponents  |  popsize={gen_size}  |  sigma={SIGMA}")
            print(f"{SEP}")

            CURRENT_OPT = WORK_DIR / "current_shell_opt"
            INIT_DIR    = CURRENT_OPT / "init"
            MEM_DIR     = CURRENT_OPT / "memory"

            INIT_DIR.mkdir(parents=True, exist_ok=True)
            MEM_DIR.mkdir(parents=True, exist_ok=True)

            shutil.copy(template,    INIT_DIR / "template.inp")
            shutil.copy(submit_scr,  INIT_DIR / "run.sh")
            shutil.copy(extract_scr, INIT_DIR / "extract.sh")

            # Other shells stay at the (possibly lagging) global snapshot;
            # only the active shell is swapped in - with its Anderson-
            # extrapolated guess if one's available this cycle, otherwise its
            # own raw latest value, same as before.
            seed_j = extrapolated[j] if extrapolated[j] is not None else latest_own[j]
            if extrapolated[j] is not None:
                print(f"  [Anderson] shell {j} ({lbl}) seeded from depth-{ANDERSON_DEPTH} extrapolation")

            base              = global_state.copy(no_energy=True)
            base.exponents[j] = seed_j.copy()
            base.save(INIT_DIR, "current", overwrite=True)

            if PROPAGATE_CMA:
                prev_state = DATA_DIR / f"shell_{j}" / f"cycle_{i - 1}" / "cma_state.pkl"
                if prev_state.exists():
                    shutil.copy(prev_state, INIT_DIR / "cma_state.pkl")

            pre_energy = last_shell_energy[j]

            spec = {
                "active_shell":           j,
                "generation_size":        GEN_SIZE_PER_SHELL[j],
                "threads":                CORES_PER_SHELL[j],
                "sigma":                  SIGMA,
                "max_generations":        MAX_GENERATIONS,
                "use_stopping":           USE_STOPPING,
                "start_energy":           float(pre_energy),
                "contract_frozen_shells": True,
            }
            if extrapolated[j] is not None:
                spec["mean_override"] = seed_j.tolist()

            CONFIG_PATH = INIT_DIR / "config.yaml"
            with open(CONFIG_PATH, "w") as cfg_f:
                yaml.safe_dump(spec, cfg_f)

            # Logged/used for delta_pct below as the honest "before" value -
            # the raw previous result, not whatever Anderson seeded this run
            # with, so the delta reflects this cycle's real net movement.
            pre_exp_j = latest_own[j].copy()

            subprocess.run(
                ["cmafex", str(CONFIG_PATH), str(INIT_DIR), str(MEM_DIR)],
                cwd=CURRENT_OPT,
                check=True,
            )

            CYCLE_DATA = DATA_DIR / f"shell_{j}" / f"cycle_{i}"
            CYCLE_DATA.mkdir(parents=True, exist_ok=True)

            OUT_DIR = CURRENT_OPT / "OUT"
            shutil.copy(OUT_DIR / "cma.log",       CYCLE_DATA / "cma.log")
            shutil.copy(OUT_DIR / "cma_trace.csv", CYCLE_DATA / "cma_trace.csv")
            shutil.copy(MEM_DIR / "current.expo",  CYCLE_DATA / "current.expo")
            shutil.copy(MEM_DIR / "cma_state.pkl", CYCLE_DATA / "cma_state.pkl")

            new_exp   = Exponent_Set.from_file(MEM_DIR / "current.expo")
            delta_e   = new_exp.energy - pre_energy
            delta_pct = float((abs(new_exp.exponents[j] - pre_exp_j) / pre_exp_j).mean()) * 100

            shutil.rmtree(CURRENT_OPT)

            log.write(f"{SEP}\n")
            log.write(f"Cycle {i + 1:2d}/{CYCLES}  |  Shell {j} ({lbl})  |  {n_exp} exponents  |  popsize={gen_size}  |  max_gen={MAX_GENERATIONS}\n")
            log.write(f"  E_before  = {pre_energy:20.10f} Eh\n")
            log.write(f"  E_after   = {new_exp.energy:20.10f} Eh\n")
            log.write(f"  dE        = {delta_e:+20.10f} Eh\n")
            log.write(f"  exp_chg   = {delta_pct:19.4f} %\n")
            log.write(f"{SEP}\n\n")
            log.flush()

            csv_writer.writerow([
                i + 1, j, lbl, n_exp, gen_size, MAX_GENERATIONS,
                f"{pre_energy:.10f}", f"{new_exp.energy:.10f}",
                f"{delta_e:.10f}", f"{delta_pct:.4f}",
            ])
            csv_f.flush()

            latest_own[j]         = new_exp.exponents[j]
            last_shell_energy[j]  = new_exp.energy
            shell_history[j].append(new_exp.exponents[j])

        # ---- launch this cycle's full run unconditionally ----
        full_exp = global_state.copy(no_energy=True)
        for k in range(n_shells):
            full_exp.exponents[k] = latest_own[k].copy()
        full_exp.uncontract_all()

        full_job    = full_manager.add_job(full_exp, template_full, name=f"full_{i}")
        launch_time = time.time()
        full_manager.submit_single(full_job)

        full_jobs.append({
            "job":          full_job,
            "launch_cycle": i,
            "launch_time":  launch_time,
        })

        msg = f"[Full] launched run from cycle {i + 1} state -> {full_job.job_dir}"
        print(msg)
        full_log.write(msg + "\n\n")
        full_log.flush()
