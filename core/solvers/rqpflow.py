"""RQPFlow: Reusable Quadratic Power Flow (SciPy CPU, FP64).

The quadratic equations are exact for fixed bus types. Reference LU is immutable
across requests; difficult requests use matrix-free Newton/GMRES and local LU
refreshes. No test targets or previous request solutions enter the cache.

Optional v2 numerics (``RQPFlowConfig.v2()``): exact Sherman-Morrison-Woodbury
topology/bus-type updates of the reference LU, a Ruiz-scaled 2-norm merit,
Iwamoto optimal-multiplier steps and a stagnation status.
"""
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres, splu

from core.utils.pf_system import PFSystem, equilibrate
from core.solvers.newton import SolverConfig, SolverResult, norm_inf, solve


class QuadraticPF:
    """Unknowns [Re(V_free), Im(V_free)], equations [P_free,Q_PQ,|V_PV|²]."""

    def __init__(self, system):
        self.system = system
        self.free, self.pq = system.free, system.pq
        self.pv = np.flatnonzero(system.bus_type == 2)
        self.k = len(self.free)

    def pack(self, voltage):
        return np.r_[voltage.real[self.free], voltage.imag[self.free]]

    def voltage(self, x, direction=False):
        v = np.zeros(self.system.n, dtype=complex)
        if not direction:
            s = self.system.slack
            v[s] = self.system.known[s, 1]*np.exp(1j*self.system.known[s, 0])
        v[self.free] = x[:self.k]+1j*x[self.k:]
        return v

    def residual(self, x):
        v = self.voltage(x)
        s = v*(self.system.Y@v).conj()
        return np.r_[s.real[self.free]-self.system.p_spec[self.free],
                     s.imag[self.pq]-self.system.q_spec[self.pq],
                     abs(v[self.pv])**2-self.system.known[self.pv, 1]**2]

    def quadratic(self, d):
        v = self.voltage(d, direction=True)
        s = v*(self.system.Y@v).conj()
        return np.r_[s.real[self.free], s.imag[self.pq], abs(v[self.pv])**2]

    def linear_operator(self, x):
        v = self.voltage(x)
        current = self.system.Y@v
        def product(d):
            w = self.voltage(d, direction=True)
            ds = w*current.conj()+v*(self.system.Y@w).conj()
            return np.r_[ds.real[self.free], ds.imag[self.pq],
                         2*(v[self.pv].conj()*w[self.pv]).real]
        return LinearOperator((2*self.k, 2*self.k), matvec=product, dtype=float)

    def jacobian(self, x):
        return self.jacobian_at(self.voltage(x))

    def jacobian_at(self, v):
        a = sp.diags((self.system.Y@v).conj())
        b = sp.diags(v)@self.system.Y.conjugate()
        de, df = (a+b).tocsr(), (1j*(a-b)).tocsr()
        ve, vf = sp.diags(2*v.real).tocsr(), sp.diags(2*v.imag).tocsr()
        return sp.bmat([[de.real[self.free][:, self.free], df.real[self.free][:, self.free]],
                        [de.imag[self.pq][:, self.free], df.imag[self.pq][:, self.free]],
                        [ve[self.pv][:, self.free], vf[self.pv][:, self.free]]], format='csc')


def optimal_multiplier(f, jd, qd, scale, alpha_max):
    """Iwamoto step: minimize ½‖D(F+αJd+α²Q(d))‖² over (0, alpha_max].

    Exact because the rectangular equations are quadratic. Returns None when d
    is not a descent direction of the scaled merit.
    """
    fw, aw, cw = scale*f, scale*jd, scale*qd
    slope = float(fw@aw)
    if not np.isfinite(slope) or slope >= 0:
        return None
    roots = np.roots([2*cw@cw, 3*aw@cw, aw@aw+2*fw@cw, slope])
    alphas = [r.real for r in roots if abs(r.imag) <= 1e-10*max(1., abs(r)) and 0 < r.real <= alpha_max]
    return min(alphas+[alpha_max], key=lambda a: float(np.sum((fw+a*aw+a*a*cw)**2)))


class LowRankCorrection:
    """Apply (J + E_R D)^{-1} with the reference LU of J (Sherman-Morrison-Woodbury).

    ``rows`` are the changed equation rows R and ``delta_rows`` their increments D.
    The r×r capacitance matrix S = I + D J^{-1} E_R is factorized once per request.
    """

    def __init__(self, lu, delta_rows, rows):
        self.lu, self.rows = lu, np.asarray(rows)
        self.d = np.ascontiguousarray(delta_rows.toarray() if sp.issparse(delta_rows) else delta_rows)
        self.rank = len(self.rows)
        e = np.zeros((lu.shape[0], self.rank), order='F')
        e[self.rows, np.arange(self.rank)] = 1.
        z = lu.solve(e)
        self.s = lu_factor(np.eye(self.rank)+self.d@z, check_finite=False)
        pivots = np.abs(np.diag(self.s[0]))
        self.ok = bool(np.isfinite(z).all() and np.isfinite(pivots).all()
                       and pivots.min() > 1e-12*max(1., pivots.max()))
        # Z S^{-1}: one triangular solve pair plus two small products per application.
        self.m = np.ascontiguousarray(lu_solve(self.s, z.T, trans=1, check_finite=False).T) if self.ok else z
        self._w = None

    def solve(self, b, trans='N'):
        if trans == 'N':
            y = self.lu.solve(b)
            return y-self.m@(self.d@y)
        if self._w is None:
            w = self.lu.solve(np.asfortranarray(self.d.T), trans='T')
            self._w = np.ascontiguousarray(lu_solve(self.s, w.T, trans=0, check_finite=False).T)
        y = self.lu.solve(b, trans='T')
        return y-self._w@y[self.rows]


@dataclass
class TopologyOperator:
    """Request-order view of B^{-1}, B = J(V_ref; request topology and bus types).

    Without an exact low-rank correction this is the reference LU (a preconditioner
    only). The shared reference LU is never modified.
    """
    q: QuadraticPF
    lu: object
    rhs_order: np.ndarray
    row_scale: np.ndarray
    topology_changed: bool
    exact: bool
    rank: int
    correction: LowRankCorrection = None
    touched_buses: np.ndarray = None

    def solve(self, b):
        b = b[self.rhs_order]
        return self.correction.solve(b) if self.correction is not None else self.lu.solve(b)

    def solve_transpose(self, g):
        y = self.correction.solve(g, 'T') if self.correction is not None else self.lu.solve(g, trans='T')
        out = np.empty_like(y)
        out[self.rhs_order] = y
        return out


@dataclass(frozen=True)
class RQPFlowConfig:
    tol: float = 1e-8
    max_iterations: int = 40
    contraction_threshold: float = .5
    restart: int = 20
    krylov_cycles: int = 3
    max_backtracks: int = 12
    allow_topology_reuse: bool = False
    allow_local_refactorization: bool = True
    low_rank_topology: bool = False
    max_low_rank: int = 96
    merit: str = 'inf'  # inf or scaled_l2
    step_rule: str = 'backtrack'  # backtrack or optimal
    optimal_alpha_max: float = 2.
    krylov_on_reuse: bool = True
    use_krylov: bool = True
    stagnation_patience: int = 0
    stagnation_ratio: float = .9

    def __post_init__(self):
        if (not np.isfinite(self.tol) or self.tol <= 0 or self.max_iterations < 0
                or not 0 < self.contraction_threshold < 1 or self.restart < 1
                or self.krylov_cycles < 1 or self.max_backtracks < 0):
            raise ValueError('Invalid fixed-topology solver configuration')
        if (self.merit not in ('inf', 'scaled_l2') or self.step_rule not in ('backtrack', 'optimal')
                or self.max_low_rank < 0 or not self.optimal_alpha_max > 0
                or self.stagnation_patience < 0 or not 0 < self.stagnation_ratio < 1):
            raise ValueError('Invalid RQPFlow v2 configuration')

    @classmethod
    def v2(cls, **overrides):
        # Frozen on the case118 Task 1.3 feasible validation split in the source research repository.
        # step_rule='optimal' is kept as an option: slower on feasible validation requests.
        return cls(**{**dict(allow_topology_reuse=True, low_rank_topology=True, merit='scaled_l2',
                             step_rule='backtrack', krylov_on_reuse=False, contraction_threshold=.1,
                             stagnation_patience=3),
                      **overrides})


class RQPFlow:
    """Compile one observable reference PFSystem, then solve matching requests.

    Default mode requires identical bus ordering/types, branches and parameters.
    allow_topology_reuse permits graph, parameter, shunt and PQ/PV changes,
    with the same bus identities/order, slack indices and island partition.
    All prescribed operating values, including slack/PV voltages, may change.
    Compilation cost is exposed separately; request LU refreshes are not cached.
    Disable allow_local_refactorization to prohibit all online LU factorizations.
    """

    def __init__(self, reference: PFSystem, config=None):
        start = time.perf_counter()
        self.config = config or RQPFlowConfig()
        self.reference = PFSystem(reference.bus_type, reference.known, reference.shunt,
                                  reference.edge_index, reference.edge_attr)
        result = solve(self.reference, config=SolverConfig(tol=min(self.config.tol*.01, 1e-10), max_outer=60))
        if not result.converged:
            raise ValueError('Reference Newton solve failed: '+result.status)
        self.reference_voltage = self.reference.voltage(result.x)
        q = QuadraticPF(self.reference)
        self.lu, self.reference_jacobian, self.row_scale = None, None, np.ones(0)
        if q.k:
            j = q.jacobian(q.pack(self.reference_voltage))
            self.lu = splu(j)
            self.reference_jacobian = j.tocsr()
            self.row_scale = equilibrate(j)[1]
            self.jacobian_scale = max(1., float(abs(j).max()))
        self.compile_stats = dict(seconds=time.perf_counter()-start,
                                  reference_newton=result.stats,
                                  reference_lu_factorizations=int(q.k > 0),
                                  lu_nnz=self.lu.L.nnz+self.lu.U.nnz if self.lu else 0)

    def check_compatible(self, system, config=None):
        if (config or self.config).allow_topology_reuse:
            # PFSystem has no external bus IDs: callers must preserve bus identity
            # and ordering. Fixed slack indices preserve rectangular coordinates.
            if system.n != self.reference.n or not np.array_equal(system.slack, self.reference.slack):
                raise ValueError('Cross-topology cache mismatch: bus count or slack indices')
            if not np.array_equal(system.islands, self.reference.islands):
                raise ValueError('Cross-topology cache mismatch: electrical islands')
            return
        for name in ('bus_type', 'edge_index', 'edge_attr', 'shunt'):
            if not np.array_equal(getattr(system, name), getattr(self.reference, name)):
                raise ValueError('Fixed topology cache mismatch: '+name)

    def initial_voltage(self, system):
        """Reference voltage rotated into the request slack frame, PV magnitudes projected."""
        v = self.reference_voltage*np.exp(1j*(system.reference_angle-self.reference.reference_angle))
        pv = system.bus_type == 2
        v[pv] *= system.known[pv, 1]/abs(v[pv])
        return v

    def prepare(self, system, config=None):
        """Request-local operator; with low_rank_topology B is exact at V_ref."""
        cfg = config or self.config
        self.check_compatible(system, cfg)
        q = QuadraticPF(system)
        # Second equation blocks are grouped PQ then PV. Align them by bus ID
        # before applying the reference inverse when PQ/PV membership changes.
        reference_second = np.r_[self.reference.pq, np.flatnonzero(self.reference.bus_type == 2)]
        target_second = np.r_[q.pq, q.pv]
        second_position = np.empty(system.n, dtype=int)
        second_position[target_second] = np.arange(q.k)
        rhs_order = np.r_[np.arange(q.k), q.k + second_position[reference_second]]
        row_scale = np.empty(2*q.k)
        row_scale[rhs_order] = self.row_scale
        topology_changed = any(not np.array_equal(getattr(system, name), getattr(self.reference, name))
                               for name in ('bus_type', 'edge_index', 'edge_attr', 'shunt'))
        operator = TopologyOperator(q, self.lu, rhs_order, row_scale, topology_changed,
                                    exact=not topology_changed, rank=0)
        if topology_changed and cfg.low_rank_topology and q.k:
            rows, delta, buses = self._jacobian_row_changes(system, q, reference_second)
            operator.rank, operator.touched_buses = len(rows), buses
            if not len(rows):
                operator.exact = True
            elif len(rows) <= cfg.max_low_rank:
                correction = LowRankCorrection(self.lu, delta, rows)
                if correction.ok:
                    operator.correction, operator.exact = correction, True
        return operator

    def _jacobian_row_changes(self, system, q, reference_second):
        """Rows of J(V_ref; request) - J_ref in reference row order.

        Only buses whose admittance row or bus type changed can differ, so the
        request rows are assembled for those buses alone.
        """
        reference, v, k = self.reference, self.reference_voltage, q.k
        dy = (system.Y-reference.Y).tocsr()
        dy.data[abs(dy.data) <= 1e-14*self.jacobian_scale] = 0
        dy.eliminate_zeros()
        changed = np.union1d(np.flatnonzero(np.diff(dy.indptr)),
                             np.flatnonzero(system.bus_type != reference.bus_type))
        changed = changed[system.bus_type[changed] != 3]
        free_position = np.full(system.n, -1)
        free_position[q.free] = np.arange(k)
        second_position = np.empty(system.n, dtype=int)
        second_position[reference_second] = np.arange(k)
        y, jr = system.Y, self.reference_jacobian
        rows, values = [], []
        for bus in changed:
            cols = y.indices[y.indptr[bus]:y.indptr[bus+1]]
            vals = y.data[y.indptr[bus]:y.indptr[bus+1]]
            keep = free_position[cols] >= 0
            position = free_position[cols[keep]]
            b = v[bus]*vals[keep].conj()
            a = (vals@v[cols]).conj()
            i = free_position[bus]
            de, df = np.zeros(k, complex), np.zeros(k, complex)
            de[position] += b
            df[position] -= 1j*b
            de[i] += a
            df[i] += 1j*a
            own = [(i, np.r_[de.real, df.real])]
            if system.bus_type[bus] == 1:
                own.append((k+second_position[bus], np.r_[de.imag, df.imag]))
            else:
                row = np.zeros(2*k)
                row[i], row[k+i] = 2*v[bus].real, 2*v[bus].imag
                own.append((k+second_position[bus], row))
            for r, row in own:
                start, end = jr.indptr[r], jr.indptr[r+1]
                row[jr.indices[start:end]] -= jr.data[start:end]
                if np.abs(row).max() > 1e-12*self.jacobian_scale:
                    rows.append(r)
                    values.append(row)
        rows = np.asarray(rows, dtype=int)
        order = np.argsort(rows)
        buses = np.unique(np.where(rows < k, q.free[np.minimum(rows, k-1)], reference_second[np.maximum(rows-k, 0)]))
        return rows[order], (np.array(values)[order] if len(rows) else np.zeros((0, 2*k))), buses

    def solve(self, system: PFSystem, state_callback=None, v0=None, config=None, prepared=None):
        start = time.perf_counter()
        cfg = config or self.config
        request = prepared if prepared is not None else self.prepare(system, cfg)
        q = request.q
        if v0 is None:
            v = self.initial_voltage(system)
        else:
            v = np.array(v0, dtype=complex)
            pv = system.bus_type == 2
            v[pv] *= system.known[pv, 1]/np.maximum(abs(v[pv]), 1e-12)
        x = q.pack(v)
        lu = self.lu
        using_reference_lu = True
        w = request.row_scale
        stats = dict(iterations=0, factorizations=0, jacobian_builds=0,
                     linear_solves=0, jv=0, krylov_calls=0, krylov_failures=0,
                     function_evaluations=1, backtracks=0,
                     topology_changed=request.topology_changed,
                     bus_type_changes=int(np.count_nonzero(system.bus_type != self.reference.bus_type)),
                     low_rank=request.rank, low_rank_applied=request.correction is not None,
                     optimal_steps=0, non_descent=0, stagnation_refreshes=0,
                     min_alpha=None, initial_residual=None)
        f = q.residual(x)
        history = [norm_inf(f)]
        stats['initial_residual'] = history[0]
        ratio, status = (float('inf') if not request.exact else 0.), 'iteration_budget'
        def merit(g):
            return norm_inf(g) if cfg.merit == 'inf' else .5*float(np.dot(w*g, w*g))
        def acceptable(trial, trial_f, alpha):
            return (np.isfinite(trial_f).all() and np.all(abs(q.voltage(trial)) > 0)
                    and merit(trial_f) < (1-1e-4*min(alpha, 1.))*merit(f))
        def backtrack(d):
            alpha = 1.
            for _ in range(cfg.max_backtracks+1):
                trial = x+alpha*d
                trial_f = q.residual(trial)
                stats['function_evaluations'] += 1
                if acceptable(trial, trial_f, alpha):
                    return trial, trial_f, alpha
                alpha *= .5
                stats['backtracks'] += 1
            return None
        def optimal(d):
            stats['jv'] += 1
            alpha = optimal_multiplier(f, q.linear_operator(x)@d, q.quadratic(d), w, cfg.optimal_alpha_max)
            if alpha is None:
                stats['non_descent'] += 1
                return None
            trial = x+alpha*d
            trial_f = q.residual(trial)
            stats['function_evaluations'] += 1
            if acceptable(trial, trial_f, alpha):
                stats['optimal_steps'] += 1
                return trial, trial_f, alpha
            return None
        def step(d):
            candidate = optimal(d) if cfg.step_rule == 'optimal' else None
            return candidate if candidate is not None else backtrack(d)
        def inverse(b):
            stats['linear_solves'] += 1
            return request.solve(b) if using_reference_lu else lu.solve(b)
        def refactorize():
            nonlocal lu, using_reference_lu
            stats['jacobian_builds'] += 1
            stats['factorizations'] += 1
            lu = splu(q.jacobian(x))
            using_reference_lu = False
        slow = 0
        for iteration in range(cfg.max_iterations+1):
            polar = system.initial_state(q.voltage(x))
            if state_callback is not None:
                state_callback(system, polar.copy(), None)
            # Projection restores prescribed PV magnitudes exactly. Check the
            # delivered state as well as the rectangular equations.
            if norm_inf(f) <= cfg.tol and norm_inf(system.residual(polar)) <= cfg.tol:
                status = 'converged'
                break
            if iteration == cfg.max_iterations:
                break
            stats['iterations'] += 1
            d = inverse(-f)
            if cfg.use_krylov and ((cfg.allow_topology_reuse and using_reference_lu and cfg.krylov_on_reuse)
                                   or ratio > cfg.contraction_threshold):
                op = q.linear_operator(x)
                def product(z):
                    stats['jv'] += 1
                    return op@z
                operator = LinearOperator(op.shape, matvec=product, dtype=float)
                preconditioner = LinearOperator(op.shape, matvec=inverse, dtype=float)
                stats['krylov_calls'] += 1
                d, info = gmres(operator, -f, x0=d, M=preconditioner,
                                rtol=min(.1, max(1e-6, np.sqrt(norm_inf(f)))), atol=0.,
                                restart=cfg.restart, maxiter=cfg.krylov_cycles)
                if info != 0:
                    stats['krylov_failures'] += 1
                    d = inverse(-f)
            candidate = step(d)
            if candidate is None:
                if not cfg.allow_local_refactorization:
                    status = 'line_search_failed_no_refactorization'
                    break
                try:
                    refactorize()
                    candidate = step(inverse(-f))
                except RuntimeError:
                    status = 'singular_jacobian'
                    break
            if candidate is None:
                status = 'line_search_failed'
                break
            old_norm = norm_inf(f)
            x, f, alpha = candidate
            stats['min_alpha'] = alpha if stats['min_alpha'] is None else min(stats['min_alpha'], alpha)
            ratio = norm_inf(f)/old_norm
            history.append(norm_inf(f))
            if cfg.stagnation_patience and norm_inf(f) > cfg.tol:
                slow = slow+1 if ratio > cfg.stagnation_ratio else 0
                if slow >= cfg.stagnation_patience:
                    if using_reference_lu and cfg.allow_local_refactorization:
                        try:
                            refactorize()
                        except RuntimeError:
                            status = 'singular_jacobian'
                            break
                        stats['stagnation_refreshes'] += 1
                        slow = 0
                    else:
                        status = 'stagnated'
                        break
        polar = system.initial_state(q.voltage(x))
        output = system.output(polar)
        stats['seconds'] = time.perf_counter()-start
        return SolverResult(status == 'converged', status, polar, output, stats, history)
