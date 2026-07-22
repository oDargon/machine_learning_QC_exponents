from collections.abc import Callable
from numpy import array, float64, ndarray, ones, eye, zeros, diag, sqrt, maximum, abs as npabs, column_stack
from numpy.linalg import lstsq, solve, norm, eigvalsh, eigh


class Newton_6:
    def __init__(
        self,
        objective:         Callable[[ndarray], ndarray],
        start_point:       ndarray,
        trust_radius:      float = 0.3,
        trust_radius_max:  float = 2.0,
        trust_radius_min:  float = 1e-6,
        stencil_min:       float = 0.0,
        accept_threshold:  float = 0.1,
        pure_newton:       bool  = False,
    ) -> None:
        self.objective        = objective
        self.trust_radius     = trust_radius
        self.trust_radius_max = trust_radius_max
        self.trust_radius_min = trust_radius_min
        self.stencil_min      = stencil_min
        self.accept_threshold = accept_threshold
        self.pure_newton      = pure_newton
        self.eval_count       = 0
        self.reject_count     = 0
        self.refit_count      = 0

        # opening lump of 6 at the initial guess: fixes the center and its model
        self.center                 = array(start_point, dtype=float64)
        displacements, values       = self._lump(self.center)
        self.center_value           = float(values[0])
        self.gradient, self.hessian = self._fit(displacements, values)
        self.model_radius           = self._stencil()
        self.history                = [(self.center.copy(), self.center_value)]

        self._propose()

    # the stencil spans the trust radius but never shrinks below stencil_min, so a
    # collapsing trust region can't drive the finite-difference model into the
    # noise floor — the model stays fitted at a scale where signal beats jitter.
    def _stencil(self) -> float:
        return max(self.trust_radius, self.stencil_min)

    # the 5 stencil displacements around a point; paired with the center they form
    # the 6 points a 2D quadratic needs.
    def _offsets(self) -> ndarray:
        r = self._stencil()
        return array([[r, 0.0], [-r, 0.0], [0.0, r], [0.0, -r], [r, r]], dtype=float64)

    # one lump: evaluate {point, point+5 offsets} in a single batch of 6.
    def _lump(self, point: ndarray) -> tuple[ndarray, ndarray]:
        displacements    = array([[0.0, 0.0], *self._offsets()], dtype=float64)
        points           = point.reshape(1, -1) + displacements
        values           = self.objective(points)
        self.eval_count += len(points)
        return displacements, values

    def _fit(self, displacements: ndarray, values: ndarray) -> tuple[ndarray, ndarray]:
        dx         = displacements[:, 0]
        dy         = displacements[:, 1]
        design     = column_stack([ones(len(displacements)), dx, dy, dx * dx, dy * dy, dx * dy])
        coeffs, *_ = lstsq(design, values, rcond=None)

        gradient = array([coeffs[1], coeffs[2]])
        hessian  = array([[2.0 * coeffs[3], coeffs[5]],
                          [coeffs[5], 2.0 * coeffs[4]]])
        return gradient, hessian

    def _damp(self, hessian: ndarray) -> ndarray:
        min_eig = float(eigvalsh(hessian).min())
        if min_eig <= 1e-8:
            hessian = hessian + (abs(min_eig) + 1e-6) * eye(2)
        return hessian

    # solve the step from the current center's model; store where the next lump
    # will land. No evaluations happen here. In pure-Newton mode the full damped
    # step is taken uncapped; otherwise it is clipped to the trust radius.
    def _propose(self) -> None:
        newton_step = -solve(self._damp(self.hessian), self.gradient)
        if self.pure_newton:
            self.at_boundary   = False
            self.proposed_step = newton_step
        else:
            self.at_boundary   = norm(newton_step) > self.trust_radius
            self.proposed_step = newton_step * (self.trust_radius / norm(newton_step)) if self.at_boundary else newton_step
        self.next_point = self.center + self.proposed_step

    def step(self) -> dict:
        if self.pure_newton:
            return self._pure_step()

        step_taken = self.proposed_step
        step_norm  = float(norm(step_taken))

        displacements, values = self._lump(self.next_point)
        candidate_value       = float(values[0])

        predicted_drop = -(self.gradient @ step_taken + 0.5 * step_taken @ self.hessian @ step_taken)
        actual_drop    = self.center_value - candidate_value
        agreement      = actual_drop / predicted_drop if predicted_drop != 0.0 else 0.0

        accepted = agreement > self.accept_threshold and candidate_value < self.center_value
        if accepted:
            self.center                 = self.next_point
            self.center_value           = candidate_value
            self.gradient, self.hessian = self._fit(displacements, values)
            self.model_radius           = self._stencil()
            self.history.append((self.center.copy(), self.center_value))
            if agreement > 0.75 and self.at_boundary:
                self.trust_radius = min(self.trust_radius * 2.0, self.trust_radius_max)
        else:
            self.reject_count += 1
            self.trust_radius  = self.trust_radius * 0.5
            # if the stencil is now finer than the one the model was fit at, that
            # stale wider fit is unreliable for the smaller step — refit it locally.
            if self._stencil() < self.model_radius:
                self.refit_count           += 1
                displacements, values       = self._lump(self.center)
                self.center_value           = float(values[0])
                self.gradient, self.hessian = self._fit(displacements, values)
                self.model_radius           = self._stencil()

        self._propose()

        return {
            "accepted":      accepted,
            "agreement":     float(agreement),
            "trust_radius":  self.trust_radius,
            "gradient_norm": float(norm(self.gradient)),
            "step_norm":     step_norm,
            "value":         self.center_value,
            "point":         self.center.copy(),
        }

    # trust region off: take the full damped Newton step every time and always
    # move, refitting at each new point. Fast on a clean bowl, unguarded on a bad one.
    def _pure_step(self) -> dict:
        step_taken = self.proposed_step
        step_norm  = float(norm(step_taken))

        displacements, values = self._lump(self.next_point)
        new_value             = float(values[0])

        predicted_drop = -(self.gradient @ step_taken + 0.5 * step_taken @ self.hessian @ step_taken)
        actual_drop    = self.center_value - new_value
        agreement      = actual_drop / predicted_drop if predicted_drop != 0.0 else 0.0

        self.center                 = self.next_point
        self.center_value           = new_value
        self.gradient, self.hessian = self._fit(displacements, values)
        self.history.append((self.center.copy(), self.center_value))

        self._propose()

        return {
            "accepted":      True,
            "agreement":     float(agreement),
            "trust_radius":  self._stencil(),
            "gradient_norm": float(norm(self.gradient)),
            "step_norm":     step_norm,
            "value":         self.center_value,
            "point":         self.center.copy(),
        }

    def minimize(self, max_steps: int = 20, grad_tol: float = 1e-6, step_tol: float = 1e-8,
                 stall_tol: float = 0.0, stall_window: int = 5, verbose: bool = False) -> tuple[ndarray, float]:
        recent = [self.center_value]
        for step_idx in range(max_steps):
            info = self.step()
            recent.append(info["value"])
            if verbose:
                tag = "acc" if info["accepted"] else "rej"
                print(f"  step {step_idx:2d}  f={info['value']:.10f}  |g|={info['gradient_norm']:.3e}  "
                      f"|dx|={info['step_norm']:.3e}  trust={info['trust_radius']:.3e}  rho={info['agreement']:+.2f}  {tag}",
                      flush=True)
            if info["gradient_norm"] < grad_tol:
                break
            if self.trust_radius < self.trust_radius_min:
                break
            if info["accepted"] and info["step_norm"] < step_tol:
                break
            # energy-stagnation stop: quit once the flat valley floor stops paying out
            if stall_tol > 0.0 and len(recent) > stall_window and recent[-1 - stall_window] - recent[-1] < stall_tol:
                break
        return self.center, self.center_value


# Optimise in coordinates whitened by the Hessian, then convert back. One probe
# lump at x0 estimates H; y = M^-1 (x - x0) with M = H^-1/2 turns the coupled valley
# into a round bowl, so the isotropic batch-of-6 method sees no anisotropy. The soft
# (near-flat) eigenvalue is floored — harmless, since motion along it barely changes E.
def minimize_whitened(
    objective:    Callable[[ndarray], ndarray],
    start_point:  ndarray,
    *,
    probe_radius: float = 0.15,
    eig_floor:    float = 1e-3,
    max_steps:    int   = 20,
    grad_tol:     float = 1e-6,
    step_tol:     float = 1e-8,
    stall_tol:    float = 0.0,
    stall_window: int   = 5,
    verbose:      bool  = False,
    **newton_kwargs,
) -> tuple[ndarray, float, "Newton_6", ndarray]:
    x0    = array(start_point, dtype=float64)
    probe = Newton_6(objective, x0, trust_radius=probe_radius)

    w, V = eigh(probe.hessian)
    w    = maximum(npabs(w), eig_floor)
    M    = V @ diag(1.0 / sqrt(w)) @ V.T   # x = x0 + M y, whitens H to ~I in y

    def objective_y(Y: ndarray) -> ndarray:
        return objective(x0 + Y @ M)

    opt          = Newton_6(objective_y, zeros(2), **newton_kwargs)
    y_star, f    = opt.minimize(max_steps=max_steps, grad_tol=grad_tol, step_tol=step_tol,
                                stall_tol=stall_tol, stall_window=stall_window, verbose=verbose)
    x_star       = x0 + M @ y_star
    opt.eval_count += probe.eval_count     # count the probe lump
    return x_star, float(f), opt, M
