from pathlib import Path
import argparse
import yaml
import sys
import shutil
from numpy import array, float64
from .exponent_handler import Exponent_Set
from .objectives import Ground_Energy_Objective
from .job_manager import Job_Manager_Config
from .common import Executor_Type
from .cma_opt_2 import cma_fixed_exponent_count, evaluate_initial


REQUIRED_FIELDS = {
    "active_shell",
    "generation_size",
    "sigma",
    "max_generations",
    "threads",
}


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Config YAML must be a mapping")
    return data


def _validate_spec(spec: dict) -> None:
    missing = sorted(REQUIRED_FIELDS - set(spec))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def _find_expo(directory: Path) -> Path:
    expos = list(directory.glob("*.expo"))
    if not expos:
        raise FileNotFoundError(f"No .expo file found in {directory}")
    if len(expos) > 1:
        raise ValueError(f"Multiple .expo files in {directory}: {[p.name for p in expos]}")
    return expos[0]


def cli() -> None:
    parser = argparse.ArgumentParser(description="CMA-ES fixed exponent count optimizer for a single shells and optional contraction")
    parser.add_argument("config",    type=Path, help="YAML with optimization parameters")
    parser.add_argument("init_dir",  type=Path, help="Directory containing .expo, template.inp, run.sh, extract.sh")
    parser.add_argument("memory_dir", type=Path, help="Directory for persistent state (current.expo, cma_state.pkl)")
    parser.add_argument("--out_dir", type=Path, default=None, help="Existing directory to continuously mirror cma.log, cma_trace.csv and best.expo")
    args = parser.parse_args()

    config_path = args.config.resolve()
    init_dir    = args.init_dir.resolve()
    work_dir    = Path.cwd().resolve()
    memory_dir  = args.memory_dir.resolve()
    out_dir     = args.out_dir.resolve() if args.out_dir is not None else None

    for p, name in [(config_path, "config"), (init_dir, "init_dir")]:
        if not p.exists():
            print(f"{name} not found: {p}", file=sys.stderr)
            sys.exit(1)

    spec = _load_yaml(config_path)
    _validate_spec(spec)

    work_dir.mkdir(parents=True, exist_ok=True)
    for src in init_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, work_dir / src.name)

    init_state = work_dir / "cma_state.pkl"
    init_state = init_state if init_state.exists() else None

    expo_path  = _find_expo(work_dir)
    template    = work_dir / "template.inp"
    run_scr     = work_dir / "run.sh"
    extract_scr = work_dir / "extract.sh"

    active_shell    = int(spec["active_shell"])
    generation_size = int(spec["generation_size"])
    sigma           = float(spec["sigma"])
    max_generations = int(spec["max_generations"])
    threads         = int(spec["threads"])

    poll_interval   = float(spec.get("poll_interval",         0.5))
    overwrite       = bool(spec.get("overwrite",              True))
    logging_        = bool(spec.get("logging",                False))
    use_stopping    = bool(spec.get("use_stopping",           False))
    contract_frozen = bool(spec.get("contract_frozen_shells", False))
    update_cadence  = int(spec.get("update_cadence",          10))

    mean_override = spec.get("mean_override", None)
    if mean_override is not None:
        mean_override = array(mean_override, dtype=float64)

    cfg = Job_Manager_Config(
        executor_type        = Executor_Type.LOCAL_BASH,
        execution_script     = run_scr,
        extraction_script    = extract_scr,
        overwrite_existing   = True,
        custom_poll_interval = poll_interval,
    )

    exp       = Exponent_Set.from_file(expo_path)
    objective = Ground_Energy_Objective(template, cfg)

    if "start_energy" in spec:
        start_exp    = exp
        start_energy = float(spec["start_energy"])
    else:
        start_exp    = evaluate_initial(exp, objective, work_dir / "initial_eval", threads=threads, contract_frozen_shells=contract_frozen)
        start_energy = start_exp.energy

    cma_run_dir = work_dir / "cma_run"
    if out_dir is None:
        out_dir = memory_dir.parent / "OUT"
        out_dir.mkdir(parents=True, exist_ok=True)

    cma_fixed_exponent_count(
        start_exp,
        start_energy,
        objective,
        work_dir               = cma_run_dir,
        generation_size        = generation_size,
        sigma                  = sigma,
        max_generations        = max_generations,
        threads                = threads,
        active_shell           = active_shell,
        overwrite              = overwrite,
        logging                = logging_,
        use_stopping           = use_stopping,
        contract_frozen_shells = contract_frozen,
        init_state_path        = init_state,
        memory_dir             = memory_dir,
        update_cadence         = update_cadence,
        mean_override          = mean_override,
        out_dir                = out_dir,
    )


if __name__ == "__main__":
    cli()
