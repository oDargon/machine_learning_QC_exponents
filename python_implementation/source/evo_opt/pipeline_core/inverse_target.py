import csv
import shutil
from pathlib import Path
from dataclasses import dataclass, field

from numpy import log

from ..exponent_handler import Exponent_Set
from ..objectives import Ground_Energy_Objective
from ..job_manager import Job_Manager_Config
from ..common import Executor_Type, L_LABELS, CHEMICAL_ACCURACY
from ..cbs_engine import _cbs_fit
from .target import read_results


@dataclass
class Inverse_Target_Config:
    submit_dir: Path
    work_dir:   Path
    results:    Path                                   # the sweep's cma_minima CSV
    backdrop:   Path                                   # basis A: the sweep's input .expo, or <atom>_backdrop_contracted.expo
    compare:    list = field(default_factory=list)     # .expo files of the B bases (abs, or relative to submit_dir)

    template_cont:  str = "temp_cont.inp"
    run_script:     str = "run.sh"
    extract_script: str = "extract.sh"

    # False: A and the swapped-in shell are all uncontracted.
    # True : A's other shells keep the contraction stored in the backdrop .expo (point
    #        backdrop at the sweep's <atom>_backdrop_contracted.expo); the swapped shell is uncontracted.
    use_contraction: bool = False

    total_cores:    int = 1        # max MOLCAS jobs run concurrently
    cbs_min_points: int = 3        # smallest tail window used when gauging the E_inf band


def equivalent_n(e, e_inf, A, b):
    """Invert the fitted curve E(N) = E_inf + A·exp(-b·sqrt(N)): the (continuous) N at
    which our optimized shell reaches energy e. Returns (N, note) or (None, reason)."""
    if not (A > 0.0 and b > 0.0):
        return None, "bad fit"
    gap = e - e_inf
    if gap <= 0.0:
        return None, "at/below E_inf"
    if gap >= A:
        return None, "above the whole curve"
    return float((log(A / gap) / b) ** 2), ""


def run_inverse_target(cfg: Inverse_Target_Config) -> Path:
    SUBMIT_DIR  = Path(cfg.submit_dir).resolve()
    WORK_DIR    = (Path(cfg.work_dir) / "inverse_target").resolve()
    RESULTS_CSV = Path(cfg.results).resolve()
    OUT_DIR     = RESULTS_CSV.parent

    if not cfg.compare:
        raise SystemExit("inverse_target: no comparison bases given")

    START_DIR  = WORK_DIR / "Start"
    STAGE_EXPO = START_DIR / "expos"
    STAGE_EXPO.mkdir(parents=True, exist_ok=True)

    def resolve(p) -> Path:
        p = Path(p)
        return p.resolve() if p.is_absolute() else (SUBMIT_DIR / p).resolve()

    def stage(src: Path, dst_dir: Path) -> Path:
        dst = dst_dir / src.name
        shutil.copy(src, dst)
        return dst

    template_cont = stage(SUBMIT_DIR / cfg.template_cont, START_DIR)
    run_scr       = stage(SUBMIT_DIR / cfg.run_script, START_DIR)
    extract_scr   = stage(SUBMIT_DIR / cfg.extract_script, START_DIR)

    base = Exponent_Set.from_file(stage(resolve(cfg.backdrop), STAGE_EXPO))
    if not cfg.use_contraction:
        base.uncontract_all()

    cmp_exps  = []
    cmp_names = []
    for i in range(len(cfg.compare)):
        src = resolve(cfg.compare[i])
        cmp_exps.append(Exponent_Set.from_file(stage(src, STAGE_EXPO)))
        cmp_names.append(src.stem)

    job_cfg = Job_Manager_Config(
        executor_type      = Executor_Type.LOCAL_BASH,
        execution_script   = run_scr,
        extraction_script  = extract_scr,
        overwrite_existing = True,
    )
    energy_objective = Ground_Energy_Objective(template_cont, job_cfg)

    lines = []
    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("=" * 72)
    out("Inverse CBS report: per-shell distance of other bases to the sweep's E_inf")
    out(f"source:   {RESULTS_CSV}")
    out(f"backdrop: {Path(cfg.backdrop).name}   contraction={'ON' if cfg.use_contraction else 'OFF'}   "
        f"primitives {base.lengths}")
    out(f"compare:  {', '.join(cmp_names)}")
    out("=" * 72)

    # ─── per-shell curve fits from the sweep ───────────────────────────────────

    shells, _ = read_results(RESULTS_CSV)
    fits      = {}   # shell -> dict
    for s in sorted(shells):
        pts = shells[s]["pts"]
        if shells[s]["failed"]:
            out(f"shell {s}: dropped failed sweep point(s) N={shells[s]['failed']} (1e6 sentinel)")
        if len(pts) < cfg.cbs_min_points:
            out(f"shell {s}: only {len(pts)} sweep point(s), need >= {cfg.cbs_min_points} — no reference")
            continue
        ns              = [pts[i][0] for i in range(len(pts))]
        es              = [pts[i][1] for i in range(len(pts))]
        e_inf, A, b, r2 = _cbs_fit(ns, es)
        e_infs          = [_cbs_fit(ns[k:], es[k:])[0] for k in range(0, len(ns) - cfg.cbs_min_points + 1)]
        fits[s] = {
            "l":       shells[s]["l"] or (L_LABELS[s] if s < len(L_LABELS) else str(s)),
            "ns":      ns,
            "e_inf":   e_inf,
            "A":       A,
            "b":       b,
            "r2":      r2,
            "band_lo": min(e_infs) - e_inf,
            "band_hi": max(e_infs) - e_inf,
            "best":    pts[-1],               # (N, E, a0, a1) at the largest sampled N
        }

    # ─── one batch: A with shell s taken from B, for every (B, s) ──────────────

    jobs    = []   # (basis_idx, shell)
    batch   = []
    names   = []
    skipped = {}   # basis_idx -> list of note strings
    for i in range(len(cmp_exps)):
        other      = cmp_exps[i]
        skipped[i] = []
        for s in sorted(fits):
            if s >= len(other.exponents) or other.lengths[s] == 0:
                skipped[i].append(f"shell {s} ({fits[s]['l']}): absent in {cmp_names[i]}")
                continue
            work = base.copy(no_energy=True)
            work.set_shell_exponents(s, other.exponents[s])
            jobs.append((i, s))
            batch.append(work)
            names.append(f"{cmp_names[i]}_s{s}")
        for s in range(len(other.exponents)):
            if s not in fits and other.lengths[s] > 0:
                skipped[i].append(f"shell {s}: {other.lengths[s]} primitive(s), no sweep curve — unreferenced")

    out(f"\nrunning {len(batch)} job(s), up to {cfg.total_cores} concurrent...")
    results = energy_objective.evaluate_batch(batch, work_dir=WORK_DIR / "jobs", threads=cfg.total_cores, names=names)

    swaps = {}   # (basis_idx, shell) -> E
    for k in range(len(jobs)):
        swaps[jobs[k]] = float(results[k].energy)

    # ─── report ────────────────────────────────────────────────────────────────

    rows = []
    for i in range(len(cmp_exps)):
        other = cmp_exps[i]
        out("")
        out("=" * 72)
        out(f"{cmp_names[i]}   primitives {other.lengths}")
        out("=" * 72)
        total = 0.0
        for s in sorted(fits):
            if (i, s) not in swaps:
                continue
            f    = fits[s]
            e    = swaps[(i, s)]
            n_b2 = other.lengths[s]
            out("")
            out(f"shell {s} ({f['l']}):  E_inf = {f['e_inf']:.8f} Eh   [band {f['band_lo'] * 1e6:+.1f}..{f['band_hi'] * 1e6:+.1f} uEh]"
                f"   b = {f['b']:.3f}   r2 = {f['r2']:.5f}   sweep N {f['ns'][0]}..{f['ns'][-1]}")
            if e >= 1e6:
                out(f"   {n_b2} primitives:  FAILED")
                rows.append([cmp_names[i], s, f["l"], n_b2, "FAILED", f"{f['e_inf']:.10f}", "", "", "", "", "", ""])
                continue

            delta      = e - f["e_inf"]
            n_best     = f["best"][0]
            vs_best    = e - f["best"][1]
            n_eq, note = equivalent_n(e, f["e_inf"], f["A"], f["b"])
            total     += delta

            out(f"   {n_b2} primitives:  E = {e:.10f} Eh")
            out(f"   distance to E_inf       : {delta * 1e6:+.1f} uEh   ({delta / CHEMICAL_ACCURACY:.3f} x chem. acc.)")
            out(f"   vs our best, N={n_best:<3d}      : {vs_best * 1e6:+.1f} uEh   (fit-free)")
            if n_eq is None:
                out(f"   equivalent N            : n/a ({note})")
                src = note
            else:
                if   n_eq < f["ns"][0]:  src = "extrapolated down"
                elif n_eq > f["ns"][-1]: src = "extrapolated up"
                else:                    src = "interpolated"
                out(f"   equivalent N            : {n_eq:.2f}   [{src}]   "
                    f"-> {n_eq / n_b2:.2f} of our functions per one of theirs")

            rows.append([cmp_names[i], s, f["l"], n_b2, f"{e:.10f}", f"{f['e_inf']:.10f}",
                         f"{f['band_lo']:+.3e}", f"{f['band_hi']:+.3e}", f"{delta:+.10f}", f"{vs_best:+.10f}",
                         f"{n_eq:.4f}" if n_eq is not None else "", src])

        for k in range(len(skipped[i])):
            out(f"   [skip] {skipped[i][k]}")
        out("")
        out(f"sum of per-shell distances (additive estimate): {total * 1e6:+.1f} uEh   "
            f"({total / CHEMICAL_ACCURACY:.3f} x chem. acc.)")

    csv_path = OUT_DIR / "inverse_target.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["basis", "shell", "l", "n_prims", "E_swap", "E_inf", "band_lo", "band_hi",
                    "delta_E_inf", "delta_best_sampled", "n_equiv", "n_equiv_source"])
        for k in range(len(rows)):
            w.writerow(rows[k])

    log_path = OUT_DIR / "inverse_target_report.log"
    log_path.write_text("\n".join(lines) + "\n")
    print(f"\nsaved {csv_path}")
    print(f"saved {log_path}")
    return csv_path
