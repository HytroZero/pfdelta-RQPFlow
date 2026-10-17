import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from rqpflow_helpers import rich_system
from core.solvers.newton import solve, SolverConfig
from core.solvers.rqpflow import RQPFlow
from core.utils.pf_system import PFSystem, InvalidPFInput
from core.utils.pf_metrics import official_pbl
from scripts.rqpflow_data import selected_source, native_plan


def graph():
    system = rich_system()
    data = HeteroData()
    data['bus'].num_nodes = system.n
    for key, value in [('x', system.known), ('bus_type', system.bus_type), ('shunt', system.shunt)]:
        data['bus'][key] = torch.tensor(value)
    branch = data['bus', 'branch', 'bus']
    branch.edge_index = torch.tensor(system.edge_index)
    branch.edge_attr = torch.tensor(system.edge_attr)
    return data


def test_cache_uses_only_inputs_and_needs_no_storage_marker(tmp_path):
    data, corrupted = graph(), graph()
    for key in ('y', 'bus_voltages', 'bus_gen'):
        data['bus'][key] = torch.zeros(3, 2)
        corrupted['bus'][key] = torch.full((3, 2), float('nan'))
    for item, value in ((data, 0.), (corrupted, float('nan'))):
        item['bus', 'branch', 'bus'].edge_label = torch.full((2, 4), value)
    torch.save(InMemoryDataset.collate([data, corrupted]), tmp_path/'test.pt')
    systems = [system for _, system in selected_source({'format': 'pfdelta-cache', 'path': 'test.pt'}, tmp_path)]
    assert len(systems) == 2
    model = RQPFlow(systems[0])
    a, b = (model.solve(system) for system in systems)
    assert a.converged and b.converged
    np.testing.assert_array_equal(a.x, b.x)


def test_prescribed_values_and_independent_jacobian():
    system = rich_system()
    x = system.initial_state()+np.array([.04, -.03, .02])
    tx = torch.tensor(x, requires_grad=True)
    np.testing.assert_allclose(system.torch_residual(tx).detach(), system.residual(x), atol=1e-12)
    np.testing.assert_allclose(torch.autograd.functional.jacobian(system.torch_residual, tx),
                               system.jacobian(x).toarray(), atol=1e-12)
    result = RQPFlow(system).solve(system)
    assert result.converged
    for rows, key, expected in [(system.free, 'p_net', system.p_spec),
                                (system.pq, 'q_net', system.q_spec)]:
        np.testing.assert_array_equal(result.output[key][rows], expected[rows])
    prescribed = system.bus_type != 1
    np.testing.assert_array_equal(result.output['voltage'][prescribed], system.known[prescribed, 1])
    np.testing.assert_array_equal(result.output['angle'][system.slack], system.known[system.slack, 0])
    assert official_pbl(system, result.output)[1] < 2e-8


def test_islands_invalid_voltage_and_iteration_failure():
    data = graph()
    data['bus', 'branch', 'bus'].edge_status = torch.tensor([False, True])
    with pytest.raises(InvalidPFInput, match='slack bus'):
        PFSystem.from_data(data)
    system = rich_system()
    assert solve(system, v0=np.zeros(system.n)).status.startswith('invalid_input')
    result = solve(system, config=SolverConfig(max_outer=0))
    assert not result.converged and result.status == 'iteration_budget'
    types = system.bus_type.copy()
    types[0] = 3
    known = system.known.copy()
    known[0] = [0., 1.]
    with pytest.raises(ValueError, match='cache mismatch'):
        RQPFlow(system).solve(PFSystem(types, known, system.shunt, system.edge_index[:, 1:], system.edge_attr[1:]))


def test_native_case_validation(tmp_path):
    with pytest.raises(ValueError, match='unique PFDelta'):
        native_plan(tmp_path, '1.3', ['custom'])
    with pytest.raises(ValueError, match='only for Task 1.3'):
        native_plan(tmp_path, '1.2', ['case2000'])


def test_solution_cli_example_and_failure_exit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-m', 'scripts.solve_rqpflow',
               '--reference', 'examples/rqpflow/reference.json',
               '--input', 'examples/rqpflow/requests.jsonl', '--output', str(tmp_path/'solution.json')]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path/'solution.json').read_text())
    assert len(report['rows']) == 2 and all(row['converged'] for row in report['rows'])
    assert len(report['rows'][0]['output']['branch_to']['q']) == 2
    result = subprocess.run(command+['--max-iterations', '0'], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads((tmp_path/'solution.json').read_text())
    assert all(row['status'] == 'iteration_budget' for row in report['rows'])
