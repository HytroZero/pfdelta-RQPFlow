import numpy as np
import pytest
from scipy.sparse.linalg import splu
from rqpflow_helpers import rich_system
from test_rqpflow import changed
from core.utils.pf_system import PFSystem
from core.solvers.newton import solve, SolverConfig
from core.solvers.rqpflow import QuadraticPF, RQPFlow, RQPFlowConfig, optimal_multiplier


def redundant(base):
    return PFSystem(base.bus_type, base.known, base.shunt,
                    np.column_stack([base.edge_index, base.edge_index[:, :1], base.edge_index[:, :1]]),
                    np.vstack([base.edge_attr, base.edge_attr[:1], base.edge_attr[:1]]))


def contingency(reference, outages, switch_types=False):
    types, known = reference.bus_type.copy(), reference.known.copy()
    if switch_types:
        nr = solve(reference, config=SolverConfig(tol=1e-12))
        pv, pq = np.flatnonzero(types == 2)[0], reference.pq[0]
        types[pv], known[pv] = 1, (-nr.output['p_net'][pv], -nr.output['q_net'][pv])
        types[pq], known[pq] = 2, (nr.output['p_net'][pq], nr.output['voltage'][pq])
    return PFSystem(types, known*[[1.03, 1.]], reference.shunt,
                    reference.edge_index[:, :len(reference.edge_attr)-outages],
                    reference.edge_attr[:len(reference.edge_attr)-outages])


@pytest.mark.parametrize('outages,switch', [(0, True), (1, False), (2, False), (2, True)])
def test_low_rank_update_matches_direct_factorization(outages, switch):
    reference = redundant(rich_system())
    model = RQPFlow(reference, RQPFlowConfig.v2())
    target = contingency(reference, outages, switch)
    request = model.prepare(target)
    assert request.exact and request.correction is not None and request.rank > 0
    direct = splu(QuadraticPF(target).jacobian_at(model.reference_voltage))
    b = np.random.default_rng(3).normal(size=2*request.q.k)
    np.testing.assert_allclose(request.solve(b), direct.solve(b), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(request.solve_transpose(b), direct.solve(b, trans='T'), rtol=1e-10, atol=1e-12)


def test_optimal_multiplier_matches_dense_line_search():
    q = QuadraticPF(changed(rich_system(), 1.2))
    x = q.pack(q.system.voltage(q.system.initial_state()))
    f = q.residual(x)
    d = -splu(q.jacobian(x)).solve(f)
    w = np.linspace(.5, 2., len(f))
    alpha = optimal_multiplier(f, q.jacobian(x)@d, q.quadratic(d), w, 2.)
    grid = np.linspace(1e-4, 2., 20001)
    merit = [np.sum((w*q.residual(x+a*d))**2) for a in grid]
    assert abs(alpha-grid[int(np.argmin(merit))]) < 2e-4
    assert optimal_multiplier(f, -(q.jacobian(x)@d), q.quadratic(d), w, 2.) is None


@pytest.mark.parametrize('outages', [1, 2])
def test_v2_strict_solves_contingencies_without_online_lu(outages, monkeypatch):
    reference = redundant(rich_system())
    model = RQPFlow(reference, RQPFlowConfig.v2(tol=1e-10, allow_local_refactorization=False))
    target = contingency(reference, outages, switch_types=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('strict mode must not factorize')
    monkeypatch.setattr('core.solvers.rqpflow.splu', forbidden)
    result = model.solve(target)
    assert result.converged, result.status
    assert result.stats['low_rank_applied'] and result.stats['factorizations'] == 0
    assert result.stats['krylov_calls'] == 0
    nr = solve(target, config=SolverConfig(tol=1e-12))
    np.testing.assert_allclose(result.output['branch_from'], nr.output['branch_from'], atol=2e-9)


def test_v2_reports_stagnation_on_overloaded_grid():
    reference = rich_system()
    model = RQPFlow(reference, RQPFlowConfig.v2(allow_local_refactorization=False))
    result = model.solve(changed(reference, 25.))
    assert not result.converged
    assert result.status in ('stagnated', 'line_search_failed_no_refactorization')


