import json
import numpy as np
import pytest
from rqpflow_helpers import rich_system
from scripts.rqpflow_data import FIELDS, native_plan, selected_source
from scripts.eval_rqpflow import evaluate


def write_system(path, system):
    np.savez(path, **{key: getattr(system, key) for key in FIELDS})


def test_native_tasks_and_cross_grid_contract(tmp_path):
    for task in ('1.1', '1.2', '1.3', '2.1', '2.2', '2.3', '4.1', '4.2', '4.3'):
        plan = native_plan(tmp_path, task, ['case14'])
        assert len(plan['pools']) == 6
        ref = plan['references']['case14']['source']['path']
        assert ('near infeasible' in ref) == (task == '4.3')
        assert ref.endswith('train.pt')
    with pytest.raises(ValueError, match='explicit target-grid'):
        native_plan(tmp_path, '3.2', None)
    plan = native_plan(tmp_path, '3.2', ['case118'], references={'case118': 'ref.npz'})
    assert len(plan['pools']) == 6
    with pytest.raises(ValueError, match='cases must'):
        native_plan(tmp_path, '3.2', ['case14'])
    assert len(native_plan(tmp_path, '1.1', ['case14'], split='val')['pools']) == 1
    assert len(native_plan(tmp_path, '4.3', ['case14'], split='val')['pools']) == 3


def test_input_sources_and_indices(tmp_path):
    s = rich_system()
    write_system(tmp_path / 'sample.npz', s)
    records = [{key: getattr(s, key).tolist() for key in FIELDS}] * 3
    (tmp_path / 'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    loaded = list(selected_source('samples.jsonl', tmp_path, indices=[1, 2]))
    assert len(loaded) == 2 and loaded[0][0].endswith('#1')
    np.testing.assert_array_equal(loaded[0][1].known, s.known)
    with pytest.raises(IndexError):
        list(selected_source('sample.npz', tmp_path, indices=[2]))
    with pytest.raises(FileNotFoundError, match='Missing PFDelta'):
        list(selected_source({'format': 'pfdelta-cache', 'path': 'test.pt'}, tmp_path))
    # Reference selection must stop before decoding subsequent records.
    (tmp_path / 'short.jsonl').write_text(json.dumps(records[0]) + '\ninvalid json')
    assert len(list(selected_source('short.jsonl', tmp_path, indices=[0]))) == 1


def test_evaluation_excludes_reference_and_reports_incompatibility(tmp_path):
    s = rich_system()
    write_system(tmp_path / 'reference.npz', s)
    s.known[s.pq] *= 1.02
    write_system(tmp_path / 'target.npz', s)
    plan = dict(schema_version=1, references={'grid': {'source': 'reference.npz'}}, pools=[
        dict(name='target', reference='grid', source='target.npz')])
    evaluate(plan, tmp_path, tmp_path / 'report.json')
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['complete'] and len(report['rows']) == 4
    assert all(r['converged'] for r in report['rows'])
    assert report['summary']['target']['RQP-strict']['factorizations'] == 0
    assert max(r['pbl_max'] for r in report['rows']) < 1e-7
    plan['pools'][0]['source'] = 'reference.npz'
    with pytest.raises(ValueError, match='Empty evaluation pool'):
        evaluate(plan, tmp_path, tmp_path / 'empty.json')


def test_incompatible_reference_preserves_nr_result(tmp_path):
    s = rich_system()
    write_system(tmp_path / 'ref.npz', s)
    s.bus_type[s.slack] = 1
    s.bus_type[1] = 3
    s.known[1] = [0, 1]
    write_system(tmp_path / 'other.npz', s)
    plan = dict(schema_version=1, references={'grid': {'source': 'ref.npz'}}, pools=[
        dict(name='other', reference='grid', source='other.npz')])
    evaluate(plan, tmp_path, tmp_path / 'report.json')
    report = json.loads((tmp_path / 'report.json').read_text())
    assert sum(r['status'] == 'incompatible_reference' for r in report['rows']) == 3
    assert next(r for r in report['rows'] if r['method'] == 'NR')['seconds'] is not None
