"""FP64 sparse, damped Newton reference solver and shared result contract."""
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.sparse.linalg import splu
from core.utils.pf_system import PFSystem, InvalidPFInput, equilibrate


def norm_inf(x):
    return float(np.max(np.abs(x), initial=0.))


@dataclass
class SolverConfig:
    tol: float = 1e-6
    max_outer: int = 30
    max_backtracks: int = 10
    armijo: float = 1e-4

    def __post_init__(self):
        if (not np.isfinite(self.tol) or self.tol <= 0 or self.max_outer < 0
                or self.max_backtracks < 0 or not 0 < self.armijo < 1):
            raise ValueError('Invalid Newton tolerance or iteration budget')


@dataclass
class SolverResult:
    converged: bool
    status: str
    x: np.ndarray | None
    output: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    residual_history: list = field(default_factory=list)


def solve(system, config=None, v0=None):
    """Solve one PFSystem or unnormalized PFDelta graph with fixed bus types."""
    config = config or SolverConfig()
    start = time.perf_counter()
    stats = dict(outer_iterations=0, jv=0, function_evaluations=0,
                 jacobian_builds=0, factorizations=0, backtrack_rejections=0)
    history = []
    try:
        system = system if isinstance(system, PFSystem) else PFSystem.from_data(system)
        x = system.initial_state(v0)
    except (InvalidPFInput, ValueError) as exc:
        stats['seconds'] = time.perf_counter()-start
        return SolverResult(False, 'invalid_input: '+str(exc), None, stats=stats)

    def residual(state):
        stats['function_evaluations'] += 1
        return system.residual(state)

    def finish(ok, status):
        stats['seconds'] = time.perf_counter()-start
        return SolverResult(ok, status, x.copy(), system.output(x), stats, history)

    f = residual(x)
    previous, status = None, 'iteration_budget'
    for iteration in range(config.max_outer+1):
        history.append(norm_inf(f))
        if not np.isfinite(f).all():
            status = 'nonfinite_residual'
            break
        if norm_inf(f) <= config.tol:
            return finish(True, 'converged')
        if iteration == config.max_outer:
            break
        stats['outer_iterations'] += 1
        j = system.jacobian(x)
        stats['jacobian_builds'] += 1
        eta = .3 if previous is None else float(np.clip(.9*(np.linalg.norm(f)/previous)**1.5, 1e-4, .5))
        a, dr, dc = equilibrate(j)
        try:
            stats['factorizations'] += 1
            d = dc*splu(a.tocsc()).solve(-dr*f)
            jd = j @ d
            stats['jv'] += 1
            if np.linalg.norm(jd+f) > eta*np.linalg.norm(f):
                status = 'direction_rejected'
                break
            # Count the independent descent check, as in the research solver.
            jd = j @ d
            stats['jv'] += 1
            slope = float(f @ jd)
            if not np.isfinite(d).all() or not np.isfinite(slope) or slope >= 0:
                status = 'non_descent_direction'
                break
            for step in range(config.max_backtracks+1):
                alpha = 2.**(-step)
                trial = x+alpha*d
                ft = residual(trial)
                if np.isfinite(ft).all() and .5*(ft@ft) <= .5*(f@f)+config.armijo*alpha*slope:
                    previous = np.linalg.norm(f)
                    x, f = trial, ft
                    break
                stats['backtrack_rejections'] += 1
            else:
                status = 'line_search_failed'
                break
        except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            status = 'linear_failure: '+str(exc)
            break
    return finish(False, status)
