"""General RQPFlow evaluation: native PFDelta splits or explicit data manifests.

Run from the repository root: python -m scripts.eval_rqpflow --help.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from core.solvers.rqpflow import RQPFlow, RQPFlowConfig
from core.solvers.newton import solve, SolverConfig, norm_inf
from core.utils.pf_metrics import official_pbl
from scripts.rqpflow_data import TASKS, native_plan, selected_source

METHODS = ('NR', 'NR-reference', 'RQP-strict', 'RQP-fallback')


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v.item() if isinstance(v, np.generic) else v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def code_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['group'], row['method'])].append(row)
    report = {}
    for (group, method), values in groups.items():
        measured = [r for r in values if r['seconds'] is not None]
        times = [r['seconds'] * 1000 for r in measured]
        pbl = [r['pbl_mean'] for r in measured]
        maxima = [r['pbl_max'] for r in measured]
        report.setdefault(group, {})[method] = dict(
            samples=len(values), converged=sum(r['converged'] for r in values),
            statuses=dict(Counter(r['status'] for r in values)),
            mean_ms=float(np.mean(times)) if times else None,
            median_ms=float(np.median(times)) if times else None,
            p95_ms=float(np.percentile(times, 95)) if times else None,
            pbl_mean=float(np.mean(pbl)) if pbl else None,
            pbl_sample_max_mean=float(np.mean(maxima)) if maxima else None,
            pbl_max=float(np.max(maxima)) if maxima else None,
            factorizations=sum(r.get('stats', {}).get('factorizations', 0) for r in values))
    return report


def evaluate(plan, base, output, methods=METHODS, config=None, samples=None, offset=0, checkpoint_every=100):
    if plan.get('schema_version') != 1 or not plan.get('pools'):
        raise ValueError('Manifest needs schema_version=1 and nonempty pools')
    config = config or RQPFlowConfig(allow_topology_reuse=True)
    if offset < 0 or checkpoint_every < 1 or (samples is not None and samples < 1):
        raise ValueError('offset must be nonnegative; samples and checkpoint-every must be positive')
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Select unique supported methods')
    names = [p['name'] for p in plan['pools']]
    if len(set(names)) != len(names):
        raise ValueError('Pool names must be unique')
    for pool in plan['pools']:
        if pool['reference'] not in plan['references']:
            raise ValueError(f'Unknown reference: {pool["reference"]}')
        if pool.get('limit') is not None and pool['limit'] < 1:
            raise ValueError('Pool limit must be positive')
    models, reference_ids, compilations = {}, set(), {}
    revision = code_revision()
    for key, entry in plan['references'].items():
        selected = list(selected_source(entry['source'], base, indices=[entry.get('index', 0)]))
        sample_id, system = selected[0]
        model = RQPFlow(system, config)
        models[key] = model
        reference_ids.add(sample_id)
        compilations[key] = dict(sample_id=sample_id, **model.compile_stats)
        for method in methods:
            if method.startswith('RQP'):
                model.config = replace(config, allow_local_refactorization=method != 'RQP-strict')
                model.solve(system)
            else:
                solve(system, config=SolverConfig(tol=config.tol, max_outer=config.max_iterations),
                      v0=model.reference_voltage if method == 'NR-reference' else None)
    rows, excluded, seen = [], 0, set()
    def save(complete):
        write_json(output, dict(schema_version=1, complete=complete, code_revision=revision, plan=plan,
            methods=list(methods), config=vars(config), selection=dict(samples_per_pool=samples, offset=offset),
            hardware=dict(platform=platform.platform(), processor=platform.processor()),
            runtime=dict(numpy=np.__version__, torch=torch.__version__, precision='float64', batch_size=1),
            timing='CPU FP64, one torch/BLAS thread, rotating method order; solve and recovery included; loading, PBL, warmup and compilation excluded',
            excluded_reference_samples=excluded, compile=compilations, summary=summarize(rows), rows=rows))
    for pool in plan['pools']:
        model = models[pool['reference']]
        count = 0
        limit = samples if samples is not None else pool.get('limit')
        for sample_id, system in selected_source(pool['source'], base, pool.get('indices'), offset=offset + pool.get('offset', 0)):
            if sample_id in reference_ids:
                excluded += 1
                continue
            identity = (pool['name'], sample_id)
            if identity in seen:
                raise ValueError(f'Duplicate evaluation sample: {identity}')
            seen.add(identity)
            results = {}
            rotation = len(seen) % len(methods)
            for method in methods[rotation:] + methods[:rotation]:
                row = dict(group=pool['name'], sample_id=sample_id, reference=pool['reference'], method=method,
                           **{k: pool.get(k) for k in ('case', 'regime', 'topology', 'split')})
                model.config = replace(config, allow_local_refactorization=method != 'RQP-strict')
                incompatible = None
                if method != 'NR':
                    try:
                        model.check_compatible(system)
                    except ValueError as error:
                        incompatible = str(error)
                if incompatible:
                    row.update(converged=False, status='incompatible_reference', error=incompatible,
                               seconds=None, pbl_mean=None, pbl_max=None, residual=None)
                else:
                    start = time.perf_counter()
                    result = model.solve(system) if method.startswith('RQP') else solve(
                        system, config=SolverConfig(tol=config.tol, max_outer=config.max_iterations),
                        v0=model.reference_voltage if method == 'NR-reference' else None)
                    elapsed = time.perf_counter() - start
                    mean, maximum = official_pbl(system, result.output)
                    row.update(converged=result.converged, status=result.status, seconds=elapsed,
                               pbl_mean=mean, pbl_max=maximum, residual=norm_inf(system.residual(result.x)),
                               stats=result.stats, residual_history=result.residual_history)
                    results[method] = result
                rows.append(row)
            nr = results.get('NR')
            for row in rows[-len(methods):]:
                result = results.get(row['method'])
                row['flow_error_vs_nr'] = max(norm_inf(result.output[k] - nr.output[k]) for k in ('branch_from', 'branch_to')) if nr is not None and result is not None and nr.converged and result.converged else None
            count += 1
            if len(seen) % checkpoint_every == 0:
                save(False)
                print(f'Evaluated {len(seen)} samples; current group: {pool["name"]}', flush=True)
            if limit is not None and count >= limit:
                break
        if not count:
            raise ValueError(f'Empty evaluation pool after reference exclusion: {pool["name"]}')
        save(False)
    save(True)
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', help='JSON plan; paths relative to its directory')
    parser.add_argument('--root', default='data')
    parser.add_argument('--task', choices=TASKS, default='1.3')
    parser.add_argument('--case', '--cases', dest='cases', nargs='+')
    parser.add_argument('--train-case', default='case118', help='Task 3.1 source grid')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--topologies', nargs='+', choices=('n', 'n-1', 'n-2'), default=['n', 'n-1', 'n-2'])
    parser.add_argument('--regimes', nargs='+', choices=('feasible', 'c2i'), default=['feasible', 'c2i'])
    parser.add_argument('--reference', action='append', default=[], metavar='CASE=FILE', help='Override native reference with NPZ, JSON, JSONL or PT inputs')
    parser.add_argument('--samples', type=int, help='Maximum samples per pool after reference exclusion; default all')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-iterations', type=int, default=40)
    parser.add_argument('--fixed-topology', action='store_true')
    parser.add_argument('--v2', action='store_true', help='Use optional low-rank/scaled-merit preset')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--write-manifest', help='Write resolved plan and exit without loading data')
    parser.add_argument('--output', default='runs/rqpflow/evaluation.json')
    args = parser.parse_args()
    if args.offset < 0 or args.checkpoint_every < 1 or (args.samples is not None and args.samples < 1):
        parser.error('offset must be nonnegative; samples and checkpoint-every must be positive')
    if len(set(args.methods)) != len(args.methods):
        parser.error('methods must be unique')
    try:
        if args.manifest:
            base = Path(args.manifest).resolve().parent
            plan = json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
        else:
            base = Path.cwd()
            references = {}
            for item in args.reference:
                case, path = item.split('=', 1)
                references[case] = str(Path(path).resolve())
            plan = native_plan(args.root, args.task, args.cases, args.split, args.topologies,
                               args.regimes, references, args.train_case)
        if args.write_manifest:
            # Preserve relative custom source paths when exporting elsewhere.
            for entry in list(plan['references'].values()) + plan['pools']:
                from scripts.rqpflow_data import source_spec
                source = source_spec(entry['source'])
                source['path'] = str((base / source['path']).resolve())
                entry['source'] = source
            write_json(args.write_manifest, plan)
            return
        config_factory = RQPFlowConfig.v2 if args.v2 else RQPFlowConfig
        config = config_factory(tol=args.tol, max_iterations=args.max_iterations,
                                allow_topology_reuse=not args.fixed_topology)
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            summary = evaluate(plan, base, args.output, tuple(args.methods), config,
                               args.samples, args.offset, args.checkpoint_every)
        print(json.dumps(summary, indent=2))
    except (ValueError, FileNotFoundError, IndexError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
