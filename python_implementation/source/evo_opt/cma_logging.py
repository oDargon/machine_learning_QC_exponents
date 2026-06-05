from pathlib import Path
from .exponent_handler import Exponent_Set
from .opt_tools_new import exponent_primitive_difference_metrics
from numpy import exp, float64, array
import csv


def hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class FixedCountLogger:
    def __init__(
        self,
        work_dir: Path,
        n_shells: int,
        active_shell: int,
        start_exp: Exponent_Set,
        start_energy: float64,
        *,
        print_to_stdout: bool = False,
    ):
        self._n_shells     = n_shells
        self._active_shell = active_shell
        self._start_exp    = start_exp
        self._start_energy = start_energy
        self._print        = print_to_stdout

        self._log_f      = open(work_dir / "cma.log", "a")
        self._csv_f      = open(work_dir / "cma_trace.csv", "w", newline="")
        self._csv_writer = csv.writer(self._csv_f)

        self._write_header()

    def _write_header(self) -> None:
        header = [
            "generation",
            "fevals",
            "time_sec",
            "best_energy_gen",
            "best_energy_overall",
            "sigma",
            "total_x_change",
            "max_global_x_change",
        ]
        for l in range(self._n_shells):
            header.append(f"shell_{l}_rms_x")
            header.append(f"shell_{l}_max_x")
        for q in range(self._start_exp.lengths[self._active_shell]):
            header.append(f"ind_sigma_l{self._active_shell}_q{q}")
        header.append("max_pct_change_from_mean")
        header.append("avg_pct_change_from_mean")
        self._csv_writer.writerow(header)
        self._csv_f.flush()

    def log_generation(
        self,
        gen: int,
        fevals: int,
        elapsed: float,
        best_energy: float,
        best_energy_overall: float,
        es,
        best_exp_gen: Exponent_Set,
    ) -> None:
        delta_e = best_energy - self._start_energy
        line = (
            f"[Gen {gen:3d}] | "
            f"Fevals {fevals:6d} | "
            f"BestE {best_energy:14.8f} | "
            f"ΔE {delta_e: .8f} | "
            f"σ {es.sigma: .3e} | "
            f"T {hms(elapsed)}"
        )
        self._log_f.write(line + "\n")
        self._log_f.flush()
        if self._print:
            print(line)

        total_rms, per_shell_rms, max_global, per_shell_max = exponent_primitive_difference_metrics(self._start_exp, best_exp_gen)

        total_x_change      = float(exp(total_rms))
        max_global_x_change = float(exp(max_global))
        per_shell_rms_x     = exp(per_shell_rms)
        per_shell_max_x     = exp(per_shell_max)

        indiv_sigmas = es.sigma * (es.sm.C.diagonal() ** 0.5)
        mean_exp     = exp(es.mean)
        best_active  = array([float(v) for v in best_exp_gen.exponents[self._active_shell]], dtype=float64)
        pct_changes  = abs((best_active - mean_exp) / mean_exp) * 100.0

        row = [
            gen,
            fevals,
            float(elapsed),
            float(best_energy),
            float(best_energy_overall),
            float(es.sigma),
            total_x_change,
            max_global_x_change,
        ]
        for l in range(self._n_shells):
            row.append(float(per_shell_rms_x[l]))
            row.append(float(per_shell_max_x[l]))
        for s in indiv_sigmas:
            row.append(float(s))
        row.append(float(pct_changes.max()))
        row.append(float(pct_changes.mean()))

        self._csv_writer.writerow(row)
        self._csv_f.flush()

    def log_stop(self, stop_reason: str, gen: int, best_energy: float, es) -> None:
        line = f"[STOP] {stop_reason} | Gen {gen:3d} | BestE {best_energy:14.8f} | sigma {es.sigma:.3e}"
        self._log_f.write(line + "\n")
        self._log_f.flush()
        if self._print:
            print(line)

    def close(self) -> None:
        self._log_f.close()
        self._csv_f.close()
