"""Solve input-only PFSystem files and export voltages, injections and branch flows."""
import argparse
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from core.solvers.rqpflow import RQPFlow, RQPFlowConfig
from core.utils.pf_metrics import official_pbl
from scripts.rqpflow_data import selected_source
from scripts.eval_rqpflow import write_json, code_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, help='NPZ, input JSON/JSONL or unnormalized PT')
    parser.add_argument('--reference-index', type=int, default=0)
    parser.add_argument('--input', required=True, help='Request source in the same formats')
    parser.add_argument('--output', default='runs/rqpflow/solutions.json')
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-iterations', type=int, default=40)
    parser.add_argument('--topology-reuse', action='store_true')
    parser.add_argument('--no-local-refactorization', action='store_true')
    parser.add_argument('--v2', action='store_true')
    args = parser.parse_args()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            reference_id, reference = next(selected_source(args.reference, Path.cwd(), indices=[args.reference_index]))
            factory = RQPFlowConfig.v2 if args.v2 else RQPFlowConfig
            config = factory(tol=args.tol, max_iterations=args.max_iterations,
                             allow_topology_reuse=args.topology_reuse,
                             allow_local_refactorization=not args.no_local_refactorization)
            model = RQPFlow(reference, config)
            rows = []
            for sample_id, system in selected_source(args.input, Path.cwd()):
                result = model.solve(system)
                mean, maximum = official_pbl(system, result.output)
                values = {key: result.output[key].tolist() for key in ('voltage', 'angle', 'p_net', 'q_net')}
                for key in ('branch_from', 'branch_to'):
                    values[key] = dict(p=result.output[key].real.tolist(), q=result.output[key].imag.tolist())
                rows.append(dict(sample_id=sample_id, converged=result.converged, status=result.status,
                                 pbl_mean=mean, pbl_max=maximum, output=values, stats=result.stats,
                                 residual_history=result.residual_history))
            if not rows:
                raise ValueError('No requests in input source')
            write_json(args.output, dict(schema_version=1, code_revision=code_revision(), reference=reference_id,
                                         config=vars(config), compile=model.compile_stats, rows=rows))
    except (ValueError, FileNotFoundError, IndexError, StopIteration) as error:
        parser.error(str(error))
    print(f'Solved {sum(row["converged"] for row in rows)}/{len(rows)} requests; output: {args.output}')
    return 0 if all(row['converged'] for row in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
