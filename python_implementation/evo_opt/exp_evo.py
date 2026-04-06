from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_spec", type=Path)
    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError("Run spec YAML must contain a top-level mapping/dictionary.")

    return data

def resolve_work_dir(spec: dict) -> Path:
    work_dir  = spec.get("work_dir")
    work_name = spec.get("work_name")

    if work_dir is not None:
        base = Path(work_dir).resolve()
    else:
        base = Path.cwd().resolve()

    if work_name is not None:
        if not isinstance(work_name, str):
            raise TypeError("work_name must be a string if provided.")
        work_name = work_name.strip()
        if not work_name:
            raise ValueError("work_name cannot be empty if provided.")
        return (base / work_name).resolve()

    return base

def prepare_work_dir(work_dir: Path, overwrite: bool) -> None:
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

def stage_inputs(work_dir: Path, spec: dict, run_spec_path: Path) -> dict:
    input_dir = work_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=False)

    required = {
        "expo": Path(spec["expo"]).resolve(),
        "template": Path(spec["template"]).resolve(),
        "run_script": Path(spec["run_script"]).resolve(),
        "extract_script": Path(spec["extract_script"]).resolve(),
        "config": Path(spec["config"]).resolve(),
    }

    staged = {}

    for key, src in required.items():
        if not src.exists():
            raise FileNotFoundError(f"Required input file not found: {src}")

        dst = input_dir / src.name
        shutil.copy2(src, dst)
        staged[key] = dst

    staged["run_spec"] = input_dir / run_spec_path.name
    shutil.copy2(run_spec_path.resolve(), staged["run_spec"])

    return staged

def resolve_output_dir(work_dir: Path, spec: dict) -> Path:
    output_name = spec.get("output_name", "output")
    if not isinstance(output_name, str):
        raise TypeError("output_name must be a string if provided.")

    output_name = output_name.strip() or "output"
    return (work_dir / output_name).resolve()

def run_child(spec: dict, staged: dict, work_dir: Path, executor_type: str):
    run_name = spec.get("run_name")

    cmd = [
        sys.executable,
        "-m",
        "evo_opt.optimize",
        str(staged["expo"]),
        str(staged["template"]),
        str(staged["run_script"]),
        str(staged["extract_script"]),
        str(work_dir),
        str(staged["config"]),
        str(executor_type),
    ]

    if run_name is not None:
        cmd.extend(["--run-name", run_name])

    return subprocess.run(cmd, check=False)

def copy_results_to_output(work_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in work_dir.iterdir():
        if item.name == output_dir.name:
            continue

        dst = output_dir / item.name

        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


def main(run_spec_path: Path) -> int:
    spec      = load_yaml(run_spec_path.resolve())
    overwrite = bool(spec.get("overwrite", False))
    work_dir  = resolve_work_dir(spec)
    prepare_work_dir(work_dir, overwrite)

    staged     = stage_inputs(work_dir, spec, run_spec_path)
    output_dir = resolve_output_dir(work_dir, spec)
    result     = run_child(spec, staged, work_dir)

    if result.returncode != 0:
        print(f"Optimization child exited with error code {result.returncode}")

    copy_results_to_output(work_dir, output_dir)

    return result.returncode


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args.run_spec))