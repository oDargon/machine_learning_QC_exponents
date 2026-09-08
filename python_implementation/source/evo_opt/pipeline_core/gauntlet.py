import csv
import shutil
from pathlib import Path
from dataclasses import dataclass

from numpy import zeros, float64

from ..exponent_handler import Exponent_Set
from ..objectives import Ground_Energy_Objective
from ..job_manager import Job_Manager_Config
from ..common import Executor_Type


@dataclass
class Gauntlet_Config:
    submit_dir: Path
    work_dir:   Path
    expo_dir:   Path         # directory of .expo basis files (rows of the matrix)
    input_dir:  Path         # directory of MOLCAS input templates (columns of the matrix)

    run_script:     str = "run.sh"
    extract_script: str = "extract.sh"
    total_cores:    int = 1          # max MOLCAS jobs run concurrently (1 core per job assumed)
    expo_glob:      str = "*.expo"   # which files in expo_dir count as bases
    input_glob:     str = "*.inp"    # which files in input_dir count as templates

    subtract_start: bool = False     # also emit a delta matrix: each cell minus the reference basis on the same input
    start_name:     str  = "start"   # reference basis for the delta (expects <start_name>.expo among the bases)


def run_gauntlet(cfg: Gauntlet_Config) -> Path:
    SUBMIT_DIR = Path(cfg.submit_dir).resolve()
    WORK_DIR   = (Path(cfg.work_dir) / "gauntlet").resolve()

    # the two matrix-source dirs: absolute paths as-is, else relative to the submit dir
    expo_dir  = Path(cfg.expo_dir)
    input_dir = Path(cfg.input_dir)
    EXPO_DIR  = expo_dir.resolve()  if expo_dir.is_absolute()  else (SUBMIT_DIR / cfg.expo_dir).resolve()
    INPUT_DIR = input_dir.resolve() if input_dir.is_absolute() else (SUBMIT_DIR / cfg.input_dir).resolve()

    START_DIR   = WORK_DIR / "Start"
    STAGE_EXPO  = START_DIR / "expos"
    STAGE_INPUT = START_DIR / "inputs"
    RESULTS_DIR = SUBMIT_DIR / "results"
    LOGS_DIR    = RESULTS_DIR / "logs_dir"
    for d in (START_DIR, STAGE_EXPO, STAGE_INPUT, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    def stage(name: str) -> Path:
        dst = START_DIR / name
        shutil.copy(SUBMIT_DIR / name, dst)
        return dst

    run_scr     = stage(cfg.run_script)
    extract_scr = stage(cfg.extract_script)

    src_expo  = sorted(EXPO_DIR.glob(cfg.expo_glob))
    src_input = sorted(INPUT_DIR.glob(cfg.input_glob))
    if not src_expo:
        raise SystemExit(f"gauntlet: no basis files matching {cfg.expo_glob!r} in {EXPO_DIR}")
    if not src_input:
        raise SystemExit(f"gauntlet: no input files matching {cfg.input_glob!r} in {INPUT_DIR}")

    # stage the matrix sources off shared/home storage into the work dir, then draw from there
    expo_files = []
    for i in range(len(src_expo)):
        dst = STAGE_EXPO / src_expo[i].name
        shutil.copy(src_expo[i], dst)
        expo_files.append(dst)
    input_files = []
    for i in range(len(src_input)):
        dst = STAGE_INPUT / src_input[i].name
        shutil.copy(src_input[i], dst)
        input_files.append(dst)

    expos      = [Exponent_Set.from_file(f) for f in expo_files]
    expo_names = [f.stem for f in expo_files]
    inp_names  = [f.stem for f in input_files]

    n_expo = len(expos)
    n_inp  = len(input_files)
    print(f"gauntlet: {n_expo} basis set(s) x {n_inp} input(s) = {n_expo * n_inp} jobs, "
          f"up to {cfg.total_cores} concurrent")

    job_cfg = Job_Manager_Config(
        executor_type      = Executor_Type.LOCAL_BASH,
        execution_script   = run_scr,
        extraction_script  = extract_scr,
        overwrite_existing = True,
    )

    # failure sentinel: the objective stamps 1e6 on any job whose energy could not be read
    energies = zeros((n_expo, n_inp), dtype=float64)

    for j in range(n_inp):
        objective = Ground_Energy_Objective(input_files[j], job_cfg)
        # fresh copies so a failed cell can't inherit the previous column's energy
        batch = [expos[i].copy(no_energy=True) for i in range(n_expo)]
        print(f"  [{j + 1}/{n_inp}] input '{inp_names[j]}': running {n_expo} basis set(s)...", flush=True)
        results = objective.evaluate_batch(
            batch,
            work_dir = WORK_DIR / inp_names[j],
            threads  = cfg.total_cores,
            names    = expo_names,
        )
        for i in range(n_expo):
            energies[i, j] = float(results[i].energy)

    # ─── pull every job log off scratch for manual inspection ──────────────────
    # cleared first so the logs always match the matrices emitted below, never a
    # previous run's basis/input set

    shutil.rmtree(LOGS_DIR, ignore_errors=True)
    n_logs = 0
    for j in range(n_inp):
        dst_dir = LOGS_DIR / inp_names[j]
        dst_dir.mkdir(parents=True, exist_ok=True)
        logs = sorted((WORK_DIR / inp_names[j]).rglob("*.log"))
        for k in range(len(logs)):
            shutil.copy(logs[k], dst_dir / logs[k].name)
            n_logs += 1

    print(f"\ncopied {n_logs} job log(s) to {LOGS_DIR}", flush=True)

    # ─── emit matrices (rows = basis .expo, cols = input template) ─────────────
    # FAILED marks the 1e6 sentinel (a job whose energy couldn't be read).
    corner = "basis \\ input"       # row label = basis (.expo); column label = input template
    row_w  = max(max(len(n) for n in expo_names), len(corner))
    col_w  = max(14, max(len(n) for n in inp_names) + 2)

    def emit_matrix(mat, failed_mask, title, legend, csv_name, signed):
        print(f"\n=== {title} ===")
        print(legend)
        print(f"{corner:<{row_w}}" + "".join(f"{inp_names[j]:>{col_w}}" for j in range(n_inp)))
        for i in range(n_expo):
            cells = ""
            for j in range(n_inp):
                if failed_mask[i][j]:
                    cells += f"{'FAILED':>{col_w}}"
                elif signed:
                    cells += f"{mat[i, j]:>+{col_w}.6f}"
                else:
                    cells += f"{mat[i, j]:>{col_w}.6f}"
            print(f"{expo_names[i]:<{row_w}}{cells}")

        path = RESULTS_DIR / csv_name
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([corner] + inp_names)
            for i in range(n_expo):
                row = [expo_names[i]]
                for j in range(n_inp):
                    if failed_mask[i][j]:
                        row.append("FAILED")
                    elif signed:
                        row.append(f"{mat[i, j]:+.10f}")
                    else:
                        row.append(f"{mat[i, j]:.10f}")
                w.writerow(row)
        print(f"saved {path}")
        return path

    raw_failed = [[energies[i, j] >= 1e6 for j in range(n_inp)] for i in range(n_expo)]
    out_path = emit_matrix(
        energies, raw_failed,
        "gauntlet energy matrix (Eh)",
        "rows = basis set (.expo)   columns = input template   |   compare DOWN a column (same input): lowest = best basis",
        "gauntlet.csv", signed=False,
    )

    # ─── delta vs the reference (start) basis, per column ──────────────────────
    if cfg.subtract_start:
        start_idx = None
        for i in range(n_expo):
            if expo_names[i] == cfg.start_name:
                start_idx = i
                break
        if start_idx is None:
            raise SystemExit(f"gauntlet: subtract_start is on but no reference basis "
                             f"'{cfg.start_name}.expo' found among {expo_names}")

        ref          = energies[start_idx, :]        # the start basis's energy per input (column)
        delta        = energies - ref                # broadcast: subtract start row from every row, per column
        delta_failed = [[energies[i, j] >= 1e6 or ref[j] >= 1e6 for j in range(n_inp)]
                        for i in range(n_expo)]
        emit_matrix(
            delta, delta_failed,
            f"gauntlet delta vs '{cfg.start_name}' basis (Eh)",
            f"each cell = energy(basis) - energy({cfg.start_name}) on the SAME input   |   "
            f"negative = better than {cfg.start_name}   (the {cfg.start_name} row is 0)",
            "gauntlet_vs_start.csv", signed=True,
        )

    return out_path
