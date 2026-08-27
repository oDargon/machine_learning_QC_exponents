import argparse
from pathlib import Path

from evo_opt.pipeline_core.gauntlet import Gauntlet_Config, run_gauntlet

_arg_parser = argparse.ArgumentParser(description="Gauntlet: cross-matrix of ground energies for every basis (.expo) x every input template")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_args = _arg_parser.parse_args()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

# the two source dirs (absolute paths, or relative to the submit dir). All matching
# files are staged into the work dir and run from there.
EXPO_DIR   = "expos"     # directory of .expo basis files  (matrix rows)
INPUT_DIR  = "inputs"    # directory of input templates    (matrix cols)
EXPO_GLOB  = "*.expo"    # which files in EXPO_DIR count as bases
INPUT_GLOB = "*.inp"     # which files in INPUT_DIR count as templates

RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

TOTAL_CORES = 6          # max MOLCAS jobs run concurrently (1 core per job assumed)

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

cfg = Gauntlet_Config(
    submit_dir     = _args.submit_dir,
    work_dir       = _args.work_dir,
    expo_dir       = EXPO_DIR,
    input_dir      = INPUT_DIR,
    run_script     = RUN_SCRIPT,
    extract_script = EXTRACT_SCRIPT,
    total_cores    = TOTAL_CORES,
    expo_glob      = EXPO_GLOB,
    input_glob     = INPUT_GLOB,
)
run_gauntlet(cfg)
