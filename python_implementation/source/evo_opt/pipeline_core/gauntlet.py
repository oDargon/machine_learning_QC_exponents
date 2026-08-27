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

    # matrix CSV: rows = basis sets, cols = inputs
    out_path = RESULTS_DIR / "gauntlet.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basis\\input"] + inp_names)
        for i in range(n_expo):
            w.writerow([expo_names[i]] + [f"{energies[i, j]:.10f}" for j in range(n_inp)])

    # printed table (FAILED where the 1e6 sentinel came back)
    row_w = max(len(n) for n in expo_names)
    col_w = max(14, max(len(n) for n in inp_names) + 2)
    print("\n=== gauntlet energy matrix (Eh) ===")
    print(" " * row_w + "".join(f"{inp_names[j]:>{col_w}}" for j in range(n_inp)))
    for i in range(n_expo):
        cells = "".join(
            (f"{'FAILED':>{col_w}}" if energies[i, j] >= 1e6 else f"{energies[i, j]:>{col_w}.6f}")
            for j in range(n_inp)
        )
        print(f"{expo_names[i]:<{row_w}}{cells}")

    print(f"\nsaved {out_path}")
    return out_path
