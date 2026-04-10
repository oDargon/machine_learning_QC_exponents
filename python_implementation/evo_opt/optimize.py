from pathlib import Path
import argparse
import inspect
import yaml
import subprocess
import shlex
from .exponent_handler import Exponent_Set
from .job_manager import Job_Manager_Config
from .executors import Executor_Type
from .cma_opt import cma_culling
from .objectives import Ground_Energy_Objective
from .parsing import parse_executor_type, parse_pullback_policy


SUPPLIED = {"start_exp", "objective", "work_dir"}


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1"}:
        return True
    if v in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("exp_path", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("run_script", type=Path)
    parser.add_argument("extract_script", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("executor_type", type=str)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--poll_time", type=float, default=5.0)

    parser.add_argument("--over_ssh", type=parse_bool, default=False)
    parser.add_argument("--ssh_target", type=str, default=None)
    parser.add_argument("--remote_work_root", type=Path, default=None)
    parser.add_argument("--remote_pullback_policy", type=str, default=None)
    parser.add_argument("--pull_rasorb", type=parse_bool, default=False)
    parser.add_argument("--cleanup_remote", type=parse_bool, default=True)

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


def prepare_remote_work_root(ssh_target: str, remote_work_root: str | Path, cleanup_remote: bool) -> None:
    remote_root = str(remote_work_root)
    remote_path = Path(remote_root)

    if len(remote_path.parts) < 4:
        raise RuntimeError(f"Refusing to clean shallow remote path: {remote_root}")

    if cleanup_remote:
        cmd = (
            f"mkdir -p {shlex.quote(remote_root)} && "
            f"find {shlex.quote(remote_root)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +"
        )
    else:
        cmd = f"mkdir -p {shlex.quote(remote_root)}"

    result = subprocess.run(
        ["ssh", ssh_target, cmd],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare remote work root '{remote_root}': {result.stderr}")


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
    over_ssh: bool                     = False,
    ssh_target: str | None             = None,
    remote_work_root: Path | None      = None,
    remote_pullback_policy: str | None = None,
    pull_rasorb: bool                  = False,
    cleanup_remote: bool               = True,
) -> int:

    params = load_yaml(config_path)
    validate_params(params)

    run_dir    = resolve_run_dir(work_dir_path, run_name)
    executor_t = parse_executor_type(executor_type)

    if over_ssh:
        if ssh_target is None:
            raise ValueError("ssh_target must be provided when over_ssh is true.")
        if remote_work_root is None:
            raise ValueError("remote_work_root must be provided when over_ssh is true.")

        prepare_remote_work_root(ssh_target=ssh_target, remote_work_root=remote_work_root, cleanup_remote=cleanup_remote)

    C = Job_Manager_Config(
        executor_type           = executor_t,
        execution_script        = run_script_path,
        extraction_script       = extract_script_path,
        manager_logging         = False,
        overwrite_existing      = False,
        custom_poll_interval    = poll_time,
        over_ssh                = over_ssh,
        ssh_target              = ssh_target,
        remote_work_root        = remote_work_root,
        remote_pullback_policy  = parse_pullback_policy(remote_pullback_policy),
        pull_rasorb             = pull_rasorb,
        cleanup_remote          = cleanup_remote,
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
            over_ssh               = args.over_ssh,
            ssh_target             = args.ssh_target,
            remote_work_root       = args.remote_work_root,
            remote_pullback_policy = args.remote_pullback_policy,
            pull_rasorb            = args.pull_rasorb,
            cleanup_remote         = args.cleanup_remote,
        )
    )