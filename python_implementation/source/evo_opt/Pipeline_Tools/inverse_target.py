import argparse
from pathlib import Path

from evo_opt.pipeline_core.inverse_target import Inverse_Target_Config, run_inverse_target

_arg_parser = argparse.ArgumentParser(description="Inverse target: per shell, how far another basis is from the sweep's CBS limit (and its equivalent N)")
_arg_parser.add_argument("--submit-dir", type=Path, default=Path.cwd())
_arg_parser.add_argument("--work-dir",   type=Path, required=True,
                         help="scratch dir for all job I/O — keep this OFF shared/home storage on HPC")
_arg_parser.add_argument("--results",    type=Path, required=True,
                         help="the sweep's cma_minima CSV")
_arg_parser.add_argument("--backdrop",   type=Path, required=True,
                         help="basis A: the sweep's input .expo, or results/<atom>_backdrop_contracted.expo")
_arg_parser.add_argument("--compare",    type=Path, nargs="+", required=True,
                         help="basis B .expo file(s) (abs, or relative to the submit dir)")
_args = _arg_parser.parse_args()

# ═══ USER CONFIGURATION ═══════════════════════════════════════════════════════

TEMPLATE_CONT  = "temp_cont.inp"
RUN_SCRIPT     = "run.sh"
EXTRACT_SCRIPT = "extract.sh"

# match the sweep. False: everything uncontracted. True: A's other shells keep the
# contraction stored in --backdrop (the sweep's <atom>_backdrop_contracted.expo).
USE_CONTRACTION = False

TOTAL_CORES    = 6       # max MOLCAS jobs run concurrently
CBS_MIN_POINTS = 3       # smallest tail window used when gauging the E_inf band

# ═══ END USER CONFIGURATION ═══════════════════════════════════════════════════

cfg = Inverse_Target_Config(
    submit_dir      = _args.submit_dir,
    work_dir        = _args.work_dir,
    results         = _args.results,
    backdrop        = _args.backdrop,
    compare         = _args.compare,
    template_cont   = TEMPLATE_CONT,
    run_script      = RUN_SCRIPT,
    extract_script  = EXTRACT_SCRIPT,
    use_contraction = USE_CONTRACTION,
    total_cores     = TOTAL_CORES,
    cbs_min_points  = CBS_MIN_POINTS,
)
run_inverse_target(cfg)
