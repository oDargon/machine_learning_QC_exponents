from pathlib import Path
import argparse
import inspect
import yaml

from .exponent_handler import Exponent_Set
from .job_manager import Job_Manager_Config, parse_executor_type
from .cma_opt import cma_culling
from .objectives import Ground_Energy_Objective


SUPPLIED = {"start_exp", "objective", "work_dir"}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("exp_path", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("run_script", type=Path)
    parser.add_argument("extract_script", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("executor_type", type=str)

    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--poll-time", type=float, default=5.0)

    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError("YAML config must contain a top-level mapping/dictionary.")

    return data


def validate_params(params: dict):
    sig     = inspect.signature(cma_culling)
    allowed = set(sig.parameters.keys()) - SUPPLIED

    missing = [
        name
        for name, p in sig.parameters.items()
        if name not in SUPPLIED
        and p.default is inspect.Parameter.empty
        and name not in params
    ]
    if missing:
        raise TypeError(f"Missing required parameters: {missing}")

    unexpected = set(params.keys()) - allowed
    if unexpected:
        raise TypeError(f"Unexpected parameters: {unexpected}")


def resolve_run_dir(work_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        name = "run"
    else:
        if not isinstance(run_name, str):
            raise TypeError("run_name must be a string or None")
        name = run_name.strip()
        if not name:
            name = "run"

    return (work_dir / name).resolve()





def main(
    exp_path: Path,
    template_path: Path,
    run_script_path: Path,
    extract_script_path: Path,
    work_dir_path: Path,
    config_path: Path,
    executor_type: str,
    run_name: str | None = None,
    poll_time: float     = 5.0,
) -> int:

    params = load_yaml(config_path)
    validate_params(params)

    run_dir = resolve_run_dir(work_dir_path, run_name)

    executor = parse_executor_type(executor_type)

    C = Job_Manager_Config(
        executor,
        run_script_path,
        extract_script_path,
        manager_logging      = False,
        overwrite_existing   = False,
        custom_poll_interval = poll_time,
    )

    exp = Exponent_Set.from_file(exp_path)
    obj = Ground_Energy_Objective(template_path, C)
    
    cma_culling(exp, obj, run_dir, **params)

    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(
            exp_path            = args.exp_path,
            template_path       = args.template,
            run_script_path     = args.run_script,
            extract_script_path = args.extract_script,
            work_dir_path       = args.work_dir,
            config_path         = args.config,
            executor_type       = args.executor_type,
            run_name            = args.run_name,
            poll_time           = args.poll_time,
        )
    )