from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import yaml


REQUIRED_CHILD_FIELDS = {
    "exp_path",
    "template_path",
    "run_script_path",
    "extract_script_path",
    "config_path",
    "executor_type",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_spec", type=Path)
    parser.add_argument("--work_dir",       type=Path, default=None, help="Override work_dir from run spec")
    parser.add_argument("--output_dir",     type=Path, default=None, help="Override output_dir from run spec")
    parser.add_argument("--submission_dir", type=Path, default=None, help="Base dir for resolving relative input paths in run spec")
    parser.add_argument("--max_time",       type=float, default=None, help="Maximum run time in seconds; partial results are collected on timeout")
    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError("Run spec YAML must contain a top-level mapping/dictionary.")

    return data

def validate_run_spec(spec: dict) -> None:
    missing = sorted(REQUIRED_CHILD_FIELDS - set(spec))
    if missing:
        raise TypeError(f"Missing required run-spec fields: {missing}")

    path_fields = [
        "exp_path",
        "template_path",
        "run_script_path",
        "extract_script_path",
        "config_path",
    ]

    for key in path_fields:
        value = spec[key]
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{key} must be a path-like string.")

    if not isinstance(spec["executor_type"], str):
        raise TypeError("executor_type must be a string.")

    if "run_name" in spec and spec["run_name"] is not None and not isinstance(spec["run_name"], str):
        raise TypeError("run_name must be a string or null.")

    if "poll_time" in spec and not isinstance(spec["poll_time"], (int, float)):
        raise TypeError("poll_time must be a number.")

    if "work_dir" in spec and spec["work_dir"] is not None and not isinstance(spec["work_dir"], (str, Path)):
        raise TypeError("work_dir must be a path-like string if provided.")

    if "work_name" in spec and spec["work_name"] is not None:
        if not isinstance(spec["work_name"], str):
            raise TypeError("work_name must be a string if provided.")
        if not spec["work_name"].strip():
            raise ValueError("work_name cannot be empty if provided.")

    if "overwrite" in spec and not isinstance(spec["overwrite"], bool):
        raise TypeError("overwrite must be a boolean.")


    if "over_ssh" in spec and not isinstance(spec["over_ssh"], bool):
        raise TypeError("over_ssh must be a boolean if provided.")

    over_ssh = spec.get("over_ssh", False)

    if over_ssh:
        if "ssh_target" not in spec or spec["ssh_target"] is None:
            raise TypeError("ssh_target must be provided when over_ssh is true.")
        if not isinstance(spec["ssh_target"], str) or not spec["ssh_target"].strip():
            raise TypeError("ssh_target must be a non-empty string when over_ssh is true.")

        if "remote_work_root" not in spec or spec["remote_work_root"] is None:
            raise TypeError("remote_work_root must be provided when over_ssh is true.")
        if not isinstance(spec["remote_work_root"], (str, Path)):
            raise TypeError("remote_work_root must be a path-like string when over_ssh is true.")

        if "remote_pullback_policy" in spec and spec["remote_pullback_policy"] is not None:
            if not isinstance(spec["remote_pullback_policy"], str):
                raise TypeError("remote_pullback_policy must be a string if provided.")

        if "pull_rasorb" in spec and not isinstance(spec["pull_rasorb"], bool):
            raise TypeError("pull_rasorb must be a boolean if provided.")

        if "cleanup_remote" in spec and not isinstance(spec["cleanup_remote"], bool):
            raise TypeError("cleanup_remote must be a boolean if provided.")

def resolve_launcher_dir(spec: dict, work_dir_override: Path | None = None) -> Path:
    if work_dir_override is not None:
        base = work_dir_override.resolve()
    else:
        base = Path(spec["work_dir"]).resolve() if spec.get("work_dir") else Path.cwd().resolve()

    work_name = spec.get("work_name")

    if work_name is not None:
        if not isinstance(work_name, str):
            raise TypeError("work_name must be a string if provided.")
        work_name = work_name.strip()
        if not work_name:
            raise ValueError("work_name cannot be empty.")
        return (base / work_name).resolve()

    # auto-generate run_N
    i = 1
    while True:
        candidate = base / f"exp_evo_{i:03d}"
        if not candidate.exists():
            return candidate.resolve()
        i += 1

def prepare_launcher_dir(work_dir: Path, overwrite: bool) -> None:
    if work_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Work directory already exists: {work_dir}. "
                "Set overwrite: true in the run spec to replace it."
            )

        resolved = work_dir.resolve()
        if len(resolved.parts) < 3:
            raise RuntimeError(f"Refusing to delete shallow directory: {resolved}")

        shutil.rmtree(resolved)

    work_dir.mkdir(parents=True, exist_ok=False)

def resolve_output_dir(
    spec: dict,
    run_spec_path: Path,
    work_dir: Path,
    output_dir_override: Path | None = None,
    submission_dir: Path | None = None,
) -> Path:
    if output_dir_override is not None:
        return (output_dir_override / "output").resolve()

    if "output_dir" not in spec or spec["output_dir"] is None:
        return (work_dir / "output").resolve()

    base = submission_dir if submission_dir is not None else run_spec_path.parent
    p = Path(spec["output_dir"])

    if not p.is_absolute():
        p = base / p

    p = p / "output"

    return p.resolve()

def stage_inputs(
    work_dir: Path,
    spec: dict,
    run_spec_path: Path,
    submission_dir: Path | None = None,
) -> dict:
    input_dir = work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)

    base = submission_dir if submission_dir is not None else run_spec_path.parent

    staged = {}
    for key in [
        "exp_path",
        "template_path",
        "run_script_path",
        "extract_script_path",
        "config_path",
    ]:
        src = (base / spec[key]).resolve()

        if not src.exists():
            raise FileNotFoundError(f"{key} does not exist: {src}")

        dst = input_dir / src.name
        if dst.exists():
            raise FileExistsError(
                f"Staging collision for {key}: destination already exists: {dst}"
            )

        shutil.copy2(src, dst)
        staged[key] = dst

    spec_dst = input_dir / run_spec_path.name
    if spec_dst.exists():
        raise FileExistsError(f"Run spec staging collision: {spec_dst}")

    shutil.copy2(run_spec_path.resolve(), spec_dst)
    staged["run_spec"] = spec_dst

    return staged

def build_child_cmd(spec: dict, staged: dict, work_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "evo_opt.optimize",
        str(staged["exp_path"]),
        str(staged["template_path"]),
        str(staged["run_script_path"]),
        str(staged["extract_script_path"]),
        str(work_dir),
        str(staged["config_path"]),
        str(spec["executor_type"]),
    ]

    if spec.get("run_name") is not None:
        cmd.extend(["--run_name", spec["run_name"]])

    if "poll_time" in spec:
        cmd.extend(["--poll_time", str(spec["poll_time"])])

    if spec.get("over_ssh", False):
        cmd.extend(["--over_ssh", "true"])

        if spec.get("ssh_target") is not None:
            cmd.extend(["--ssh_target", str(spec["ssh_target"])])

        if spec.get("remote_work_root") is not None:
            cmd.extend(["--remote_work_root", str(spec["remote_work_root"])])

        if spec.get("remote_pullback_policy") is not None:
            cmd.extend(["--remote_pullback_policy", str(spec["remote_pullback_policy"])])

        if "pull_rasorb" in spec:
            cmd.extend(["--pull_rasorb", str(spec["pull_rasorb"]).lower()])

        if "cleanup_remote" in spec:
            cmd.extend(["--cleanup_remote", str(spec["cleanup_remote"]).lower()])

    return cmd

def run_child(cmd: list[str], timeout: float | None = None) -> tuple[int, bool]:
    try:
        result = subprocess.run(cmd, check=False, timeout=timeout)
        return result.returncode, False
    except subprocess.TimeoutExpired:
        return -1, True

def extract_outputs(work_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "culling.log",
        "culling_trace.csv",
    ]
    dirs = [
        "run_logs",
        "run_csvs",
        "initial_culled",
        "best_culled",
    ]
    for name in files:
        src = work_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    for name in dirs:
        src = work_dir / name
        dst = output_dir / name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

def write_status(work_dir: Path, returncode: int) -> None:
    status = "SUCCESS" if returncode == 0 else "FAILURE"

    with open(work_dir / "status.txt", "w") as f:
        f.write(f"{status}\n")
        f.write(f"returncode={returncode}\n")


def main(
    run_spec_path: Path,
    work_dir_override: Path | None = None,
    output_dir_override: Path | None = None,
    submission_dir: Path | None = None,
    max_time: float | None = None,
) -> int:
    try:
        run_spec_path = run_spec_path.resolve()

        if not run_spec_path.exists():
            raise FileNotFoundError(f"Run spec file not found: {run_spec_path}")

        if submission_dir is not None:
            submission_dir = submission_dir.resolve()
            if not submission_dir.is_dir():
                raise NotADirectoryError(f"submission_dir does not exist: {submission_dir}")

        spec = load_yaml(run_spec_path)
        validate_run_spec(spec)

        overwrite    = spec.get("overwrite", False)
        launcher_dir = resolve_launcher_dir(spec, work_dir_override=work_dir_override)

        prepare_launcher_dir(launcher_dir, overwrite)
        staged = stage_inputs(launcher_dir, spec, run_spec_path, submission_dir=submission_dir)
        cmd    = build_child_cmd(spec, staged, launcher_dir)

        (launcher_dir / "child_command.txt").write_text(" ".join(cmd) + "\n")
        returncode, timed_out = run_child(cmd, timeout=max_time)
        write_status(launcher_dir, returncode)
        output_dir = resolve_output_dir(
            spec, run_spec_path, launcher_dir,
            output_dir_override=output_dir_override,
            submission_dir=submission_dir,
        )

        work_dir = launcher_dir / spec.get("run_name") if spec.get("run_name") is not None else launcher_dir / "run"
        extract_outputs(work_dir, output_dir)

        if timed_out:
            print("Optimization timed out — partial results collected.")
        elif returncode != 0:
            print(f"Optimization child exited with code {returncode}")
        else:
            print("Optimization completed successfully.")

        return returncode

    except Exception as e:
        print(f"Launcher error: {e}", file=sys.stderr)

        try:
            write_status(launcher_dir, -1)
        except Exception:
            pass

        return 1

def cli() -> int:
    args = parse_args()

    run_spec = args.run_spec
    if not run_spec.is_absolute() and args.submission_dir is not None:
        run_spec = args.submission_dir / run_spec

    return main(
        run_spec,
        work_dir_override   = args.work_dir,
        output_dir_override = args.output_dir,
        submission_dir      = args.submission_dir,
        max_time            = args.max_time,
    )

if __name__ == "__main__":
    raise SystemExit(cli())