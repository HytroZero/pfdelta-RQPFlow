"""Data sources and native split plans for the standalone RQPFlow evaluator."""
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from core.utils.pf_system import PFSystem

FIELDS = ('bus_type', 'known', 'shunt', 'edge_index', 'edge_attr')
TASKS = tuple(f'{group}.{variant}' for group in range(1, 5) for variant in range(1, 4))
CASES = ('case14', 'case30', 'case57', 'case118', 'case500', 'case2000')
REGIMES = {'feasible': 'feasible', 'c2i': 'near infeasible', 'approaching': 'approaching infeasible'}


def cache_source(root, task, case, split, regime, topology):
    directory = Path(root).resolve() / case / topology / 'processed' / f'task_{task}_{REGIMES[regime]}_'
    return {'format': 'pfdelta-cache', 'path': str(directory / f'{split}.pt')}


def native_plan(root, task, cases, split='test', topologies=('n', 'n-1', 'n-2'),
                regimes=('feasible', 'c2i'), references=None, train_case='case118'):
    task = str(task)
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    if split not in ('val', 'test'):
        raise ValueError('Native evaluation split must be val or test')
    if task.startswith('3.'):
        train = {'3.1': [train_case], '3.2': list(CASES[:3]), '3.3': list(CASES[3:])}[task]
        allowed = [c for c in CASES if c not in train] if split == 'test' else train
        cases = cases or [case for case in allowed if case != 'case2000']
        if any(c not in allowed for c in cases):
            raise ValueError(f'Task {task} {split} cases must be in {allowed}')
    else:
        cases = cases or ['case118']
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError(f'Select unique PFDelta cases from {CASES}')
    if 'case2000' in cases and task != '1.3':
        raise ValueError('Official case2000 evaluation is supported only for Task 1.3')
    refs, pools = {}, []
    for case in cases:
        if references and case in references:
            refs[case] = {'source': references[case], 'index': 0}
        elif task.startswith('3.') and split == 'test':
            raise ValueError(f'Task {task} needs an explicit target-grid reference for {case}; use --reference {case}=FILE')
        else:
            regime = 'c2i' if task == '4.3' else 'feasible'
            refs[case] = {'source': cache_source(root, task, case, 'train', regime, 'n'), 'index': 0}
        for regime in regimes:
            for topology in topologies:
                if split == 'val':
                    if regime == 'c2i' and not task.startswith('4.'):
                        continue
                    if regime == 'feasible' and task == '4.3':
                        continue
                    if task == '1.1' and topology != 'n':
                        continue
                    if task == '1.2' and topology == 'n-2':
                        continue
                pools.append(dict(name=f'{case}/{regime}/{topology}', reference=case,
                                  case=case, regime=regime, topology=topology, split=split,
                                  source=cache_source(root, task, case, split, regime, topology)))
    if not pools:
        raise ValueError('Selected task has no samples in these validation regimes/topologies')
    return dict(schema_version=1, task=task,
                protocol='PFDelta splits with explicit target-grid references' if references else 'PFDelta splits with one training reference per grid',
                references=refs, pools=pools)


def source_spec(value):
    if isinstance(value, str):
        return {'path': value, 'format': Path(value).suffix.lstrip('.')}
    return dict(value)


def load_source(value, base):
    """Yield stable source IDs and input-only PFSystems; no dataset processing."""
    spec = source_spec(value)
    path = (Path(base) / spec['path']).resolve()
    fmt = spec.get('format', path.suffix.lstrip('.'))
    if fmt == 'npz':
        with np.load(path, allow_pickle=False) as arrays:
            yield str(path), PFSystem(**{key: arrays[key] for key in FIELDS})
    elif fmt in ('json', 'jsonl'):
        with path.open(encoding='utf-8-sig') as stream:
            records = (json.loads(line) for line in stream if line.strip()) if fmt == 'jsonl' else iter([json.load(stream)])
            for i, record in enumerate(records):
                record = record.get('data', record)
                if 'bus_type' not in record:
                    raise ValueError('JSON sources require PFSystem input arrays; convert raw PFDelta samples with PFDeltaDataset')
                system = PFSystem(**{key: record[key] for key in FIELDS})
                yield f'{path}#{i}', system
    elif fmt in ('pt', 'pfdelta-cache'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing PFDelta input file: {path}; prepare caches with PFDeltaDataset')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(payload, tuple) and len(payload) == 2:
            dataset = InMemoryDataset()
            dataset.data, dataset.slices = payload
        else:
            dataset = payload if isinstance(payload, list) else [payload]
        for i in range(len(dataset)):
            data = dataset[i]
            system = data if isinstance(data, PFSystem) else PFSystem.from_data(data)
            yield f'{path}#{i}', system
    else:
        raise ValueError(f'Unsupported source format: {fmt}')


def selected_source(value, base, indices=None, limit=None, offset=0):
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError('offset must be nonnegative; limit must be positive')
    wanted = None if indices is None else set(indices)
    if wanted is not None and (len(wanted) != len(indices) or any(type(i) is not int or i < 0 for i in wanted)):
        raise ValueError('indices must be unique nonnegative integers')
    if wanted == set():
        return
    count, found = 0, set()
    for i, sample in enumerate(load_source(value, base)):
        if wanted is not None:
            if i not in wanted:
                continue
            found.add(i)
        if count >= offset:
            yield sample
            if limit is not None and count - offset + 1 >= limit:
                return
        count += 1
        if wanted is not None and found == wanted:
            return
    if wanted is not None and wanted != found:
        raise IndexError(f'Source indices not found: {sorted(wanted - found)}')
