# RQPFlow: Reusable Quadratic Power Flow

RQPFlow is a training-free AC power-flow solver for repeated operating conditions. It solves a reference case once, caches a sparse Jacobian LU factorization, and reuses it as loads and generator setpoints change. The solver runs on CPU in double precision and returns bus voltages, net injections, and complex power flows at both ends of every branch.

[Method and experimental report](docs/rqpflow_report.md) (Chinese) · [Four-solver comparison](docs/four_method_compare.md) (Chinese) · [MIT License](LICENSE)

## Results at a glance

On the first 100 feasible N-topology validation cases from PFDelta Task 1.3 (IEEE 118-bus), RQPFlow reached a power-mismatch threshold of **1e-8 p.u. in 3.01 ms**, compared with **11.50 ms** for flat-start sparse Newton: **3.82x faster** by median time to the same error.

![IEEE 118-bus: power mismatch versus elapsed time and time to reach the same error](figures/fig_rqpflow_case118_error_time.png)

*From the [RQPFlow report](docs/rqpflow_report.md). Left: power mismatch at equal elapsed time. Right: time to reach each error threshold. Shading shows the interquartile range.*

| Method | Median time to 1e-8 p.u. | Converged solves |
| --- | ---: | ---: |
| **RQPFlow** | **3.01 ms** | **300/300** |
| Sparse Newton, flat start | 11.50 ms | 300/300 |
| Sparse Newton, reference start | 11.78 ms | 300/300 |

The reported protocol uses CPU FP64, batch size 1, warmup, rotating method order, and three timing repetitions per case. Repetitions are reduced to a median per case before taking the median across cases. Online timing excludes data loading and reference precomputation; precomputation took 14.95 ms. Error is the largest absolute active- or reactive-power mismatch component at the returned voltage state.

A separate single-pass evaluation of the same 100 cases recorded **17 online LU factorizations**, compared with **456** for flat-start Newton. Including six factorizations during reference construction, RQPFlow used **23 factorizations in total**, a **95.0% reduction**. That single-pass evaluation achieved a **3.49x total-time speedup including reference precomputation**. The report gives the measurement details and branch-flow accuracy checks.

## How it works

For a fixed network and bus-type assignment, rectangular voltage coordinates turn the AC power-flow equations into an exactly quadratic system:

$$
F(x+d;u) = F(x;u) + J(x;u)d + Q(d,d).
$$

Here, $x$ contains the real and imaginary voltages at non-slack buses, and $u$ contains the prescribed operating quantities. RQPFlow uses this structure to compute analytic Jacobian-vector products and reuse numerical work across requests.

1. **Build a reusable reference.** Solve a reference operating point with sparse Newton, form its rectangular-coordinate Jacobian, and cache its sparse LU factors. Reference construction uses prescribed inputs and requires no stored solution labels.
2. **Start each request from the reference.** Adjust the reference voltage to the requested slack frame and PV voltage magnitudes, then use the cached factors for inexpensive chord-iteration steps.
3. **Apply current-state corrections when needed.** If residual contraction is insufficient, use matrix-free Newton-GMRES with analytic Jacobian-vector products and the cached LU as a preconditioner.
4. **Refresh locally after a failed step.** A residual-decreasing line search checks each update. If a direction fails, factor the current Jacobian for that request. The shared reference cache stays unchanged.

Each request starts from the same reference cache, so solving a new case does not depend on the previous request's solution. Success requires both the internal rectangular residual and the power residual of the returned voltage state to meet the configured tolerance.

By default, reuse requires matching bus types, branch connectivity and ordering, branch parameters, and shunts. Loads, active-power setpoints, PV voltage magnitudes, and slack voltage settings may vary. Optional cross-topology reuse is available through `--topology-reuse`; `--v2` enables the low-rank/scaled-merit preset. These modes require consistent bus identity and ordering, slack indices, and electrical islands.

## Quick start

Run the following commands from the repository root. Create a dedicated environment and install the solver dependencies:

```sh
conda create -n pfdelta python=3.11
conda activate pfdelta
python -m pip install -r requirements-rqpflow.txt
```

Solve the included three-bus example, which requires no dataset download or training:

```sh
python -m scripts.solve_rqpflow --reference examples/rqpflow/reference.json --input examples/rqpflow/requests.jsonl --output runs/rqpflow/solutions.json
```

The output JSON contains a convergence flag and status for each request, voltage magnitudes and angles, net active/reactive injections, branch-end power flows, PBL Mean/Max, solver statistics, and residual histories. The command exits with code `2` if any request fails to converge.

The example files illustrate the input-only `PFSystem` format: `bus_type`, `known`, `shunt`, `edge_index`, and `edge_attr`. Electrical quantities use per-unit conventions, and angles in this format use radians. JSON/JSONL, NPZ, and unnormalized PT sources are supported. See the [input adapter](scripts/rqpflow_data.py) and [PFSystem contract](core/utils/pf_system.py) for field definitions and conversions.

## Evaluation

Run the bundled example through the evaluator to compare flat-start Newton, reference-start Newton, and RQPFlow with and without local refactorization:

```sh
python -m scripts.eval_rqpflow --manifest examples/rqpflow/manifest.json --output runs/rqpflow/example_evaluation.json
```

For PFDelta experiments, the evaluator accepts task, case, split, topology, and feasibility-regime selections. Prepare the corresponding processed caches with the [PFDelta loader](core/datasets/pfdelta_dataset.py) before running a dataset evaluation. Inspect the available options with:

```sh
python -m scripts.eval_rqpflow --help
```

Evaluation exports the code revision, reference-construction cost, convergence statuses, per-request timings, factorization counts, PBL Mean/Max, and branch-flow discrepancies against converged Newton solutions. Feasible and close-to-infeasible results are grouped separately.

## Dataset

This repository builds on **PFDelta**, a benchmark containing 859,800 solved AC power-flow instances across IEEE 14/30/57/118-bus and GOC 500/2000-bus systems. It covers load and generation variations, N/N-1/N-2 topologies, and close-to-infeasible operating conditions. Dataset loaders, generation code, and original learning baselines are retained here.

- [Dataset on Hugging Face](https://huggingface.co/datasets/pfdelta/pfdelta/tree/main)
- [PFDelta paper](https://arxiv.org/abs/2510.22048)

If you use the dataset or its original benchmark code, please cite:

```bibtex
@inproceedings{rivera2025pf,
  title={{PF}\ensuremath{\Delta}: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations},
  author={Ana K. Rivera and Anvita Bhagavathula and Alvaro Carbonero and Priya L. Donti},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2025},
  url={https://openreview.net/forum?id=Gi1HtsTAkv}
}
```

## License

Released under the [MIT License](LICENSE). The original PFDelta copyright notice is retained.
