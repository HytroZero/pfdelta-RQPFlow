import numpy as np
import pytest
from rqpflow_helpers import rich_system
from core.utils.pf_system import PFSystem
from core.solvers.newton import solve, SolverConfig
from core.solvers.rqpflow import QuadraticPF, RQPFlow, RQPFlowConfig
from core.utils.pf_metrics import official_pbl


def changed(s, scale=1.05):
    known = s.known.copy()
    known[s.pq] *= scale
    known[s.slack, 0] += .12
    known[s.bus_type != 1, 1] *= 1.01
    return PFSystem(s.bus_type, known, s.shunt, s.edge_index, s.edge_attr)


def test_exact_quadratic_expansion_and_analytic_jvp():
    q = QuadraticPF(rich_system())
    rng = np.random.default_rng(15)
    x = q.pack(q.system.voltage(q.system.initial_state()))+rng.normal(0, .05, 2*q.k)
    d = rng.normal(0, .03, 2*q.k)
    np.testing.assert_allclose(q.residual(x+d), q.residual(x)+q.jacobian(x)@d+q.quadratic(d), atol=2e-14)
    np.testing.assert_allclose(q.linear_operator(x)@d, q.jacobian(x)@d, atol=2e-14)
    np.testing.assert_allclose(q.linear_operator(x)@d, (q.residual(x+1e-5*d)-q.residual(x-1e-5*d))/2e-5, atol=2e-9)


def test_reuse_accuracy_and_independent_physics():
    reference = rich_system()
    model = RQPFlow(reference, RQPFlowConfig(tol=1e-10))
    for factor in (1., 1.1, .85):
        system = changed(reference, factor)
        result = model.solve(system)
        nr = solve(system, config=SolverConfig(tol=1e-12))
        assert result.converged and nr.converged
        for name in ('branch_from', 'branch_to'):
            np.testing.assert_allclose(result.output[name], nr.output[name], atol=2e-9)
        mean, maximum = official_pbl(system, result.output)
        assert maximum < 2e-10
        np.testing.assert_allclose([mean, maximum], [result.output['pbl_mean'], result.output['pbl_max']], atol=1e-12)
    a, b = model.solve(system), model.solve(system)
    np.testing.assert_array_equal(a.x, b.x)


def test_cache_invalidation_and_budgets():
    reference = rich_system()
    model = RQPFlow(reference, RQPFlowConfig(max_iterations=0))
    assert not model.solve(changed(reference)).converged
    other = changed(reference)
    other.edge_attr[0, 0] *= 1.01
    with pytest.raises(ValueError, match='cache mismatch'):
        model.solve(other)
    with pytest.raises(ValueError):
        RQPFlowConfig(tol=0)


def test_krylov_path():
    s = rich_system()
    model = RQPFlow(s, RQPFlowConfig(tol=1e-11, contraction_threshold=1e-8))
    result = model.solve(changed(s, 1.3))
    assert result.converged
    assert result.stats['krylov_calls'] > 0


def test_failed_cached_direction_refreshes_without_mutating_cache():
    s = rich_system()
    model = RQPFlow(s)
    class BadReference:
        def solve(self, b):
            return np.zeros_like(b)
    broken = BadReference()
    model.lu = broken
    result = model.solve(changed(s))
    assert result.converged
    assert result.stats['factorizations'] >= 1
    assert model.lu is broken


def test_slack_only_islands():
    s = PFSystem(np.array([3, 3]), np.array([[.1, 1.], [-.2, 1.05]]),
                 np.zeros((2, 2)), np.empty((2, 0), dtype=int), np.empty((0, 8)))
    result = RQPFlow(s).solve(s)
    assert result.converged
    assert result.stats['factorizations'] == 0


@pytest.mark.parametrize('outages', [1, 2])
@pytest.mark.parametrize('switch_types', [False, 'swap', 'pv_to_pq', 'pq_to_pv'])
def test_cross_topology_single_cache(outages, switch_types, monkeypatch):
    base = rich_system()
    # Add two redundant circuits, so either outage leaves the grid connected.
    reference = PFSystem(base.bus_type, base.known, base.shunt,
        np.column_stack([base.edge_index, base.edge_index[:, :1], base.edge_index[:, :1]]),
        np.vstack([base.edge_attr, base.edge_attr[:1], base.edge_attr[:1]]))
    model = RQPFlow(reference, RQPFlowConfig(allow_topology_reuse=True,
                    allow_local_refactorization=False, tol=1e-10))
    original_lu = model.lu
    types, known = base.bus_type.copy(), base.known.copy()
    if switch_types:
        nr = solve(base, config=SolverConfig(tol=1e-12))
        assert nr.converged
        pv, pq = np.flatnonzero(types == 2)[0], base.pq[0]
        if switch_types in ('swap', 'pv_to_pq'):
            types[pv] = 1
            known[pv] = (-nr.output['p_net'][pv], -nr.output['q_net'][pv])
        if switch_types in ('swap', 'pq_to_pv'):
            types[pq] = 2
            known[pq] = (nr.output['p_net'][pq], nr.output['voltage'][pq])
    target = PFSystem(types, known, base.shunt, reference.edge_index[:, :-outages],
                      reference.edge_attr[:-outages])
    nr = solve(target, config=SolverConfig(tol=1e-12))
    assert nr.converged
    def forbidden(*args, **kwargs):
        raise AssertionError('Strict single-cache mode must never factorize')
    monkeypatch.setattr('core.solvers.rqpflow.splu', forbidden)
    result = model.solve(target)
    assert result.converged, result.status
    assert result.stats['topology_changed']
    assert result.stats['factorizations'] == 0
    assert model.lu is original_lu
    assert official_pbl(target, result.output)[1] < 2e-10
    np.testing.assert_allclose(result.output['branch_from'], nr.output['branch_from'], atol=2e-9)
    np.testing.assert_array_equal(result.x, model.solve(target).x)


def test_cross_topology_rejects_changed_slack():
    reference = rich_system()
    types, known = reference.bus_type.copy(), reference.known.copy()
    slack, pv = reference.slack[0], np.flatnonzero(types == 2)[0]
    types[slack], types[pv] = 2, 3
    known[slack], known[pv] = [0., 1.], [0., 1.]
    target = PFSystem(types, known, reference.shunt, reference.edge_index, reference.edge_attr)
    model = RQPFlow(reference, RQPFlowConfig(allow_topology_reuse=True))
    with pytest.raises(ValueError, match='slack indices'):
        model.solve(target)


def test_strict_cache_failure_does_not_refactorize():
    reference = rich_system()
    model = RQPFlow(reference, RQPFlowConfig(allow_local_refactorization=False))
    class BadReference:
        def solve(self, b):
            return np.zeros_like(b)
    model.lu = BadReference()
    result = model.solve(changed(reference))
    assert not result.converged
    assert result.status == 'line_search_failed_no_refactorization'
    assert result.stats['factorizations'] == 0
