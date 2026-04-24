from pathlib import Path
from .optimize import main

rc = main(
    exp_path=Path("testdata/expo.txt"),
    template_path=Path("testdata/template.txt"),
    run_script_path=Path("testdata/run.sh"),
    extract_script_path=Path("testdata/extract.sh"),
    work_dir_path=Path("test_runs"),
    config_path=Path("testdata/config.yaml"),
    executor_type="local",
    run_name="trial1",
)

print(rc)