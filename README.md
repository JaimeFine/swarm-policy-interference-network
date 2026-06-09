# SPIN

SPIN is a research repository for **decentralized swarm control via tensorized policy coordination**. The project explores how multi-agent coordination can be represented with compressed tensor-network structure so that swarm decision-making becomes more tractable on constrained compute budgets.

In practical terms, this repo appears to combine:
- a custom swarm simulation environment
- SPIN-specific tensor/policy logic
- baseline methods for comparison
- experiment scripts for scaling, robustness, and benchmarking
- result artifacts for analysis and plotting

## Repository Structure

```text
SPIN/
├── baselines/
├── results/
├── src/
├── main.py
├── noise_robustness_study.py
├── plot_noise_degradation.py
├── scaling_study.py
├── tensor_core_benchmark.py
└── visualizer.py
```

## Top-Level Overview

### `src/`
Core research implementation for the SPIN framework and the simulation stack.

Notable modules include:
- `spin_policy.py`: SPIN policy logic / coordination behavior
- `tensor.py`: tensor operations and compressed policy representation utilities
- `topology.py`: swarm or communication topology construction
- `network.py`: network-level abstractions used by the coordination model
- `simulation.py`: simulation loop / rollout logic
- `coverage.py`: coverage-style swarm behavior utilities
- `pettingzoo_env.py`: multi-agent environment wrapper or integration
- `quantum.py`: experimental or mathematically inspired coordination utilities

### `baselines/`
Reference implementations and comparison methods used to evaluate SPIN against simpler or established approaches.

This folder includes:
- `apf_velocity.py`: artificial potential field style baseline
- `cbba.py`: consensus-based bundle allocation style baseline
- `mappo.py`: MAPPO-related baseline logic
- `mappo_env.py`: environment adapter for the MAPPO baseline
- `common.py`: shared helpers across baseline experiments
- `run_baselines.py`: entry point for running baseline comparisons
- `figures/` and `results/`: baseline-specific outputs and visual summaries

### `results/`
Stored experiment summaries and tabular outputs, including:
- `trial_results.csv`
- `trial_summary.csv`

These files likely capture aggregated run metrics for later plotting or discussion in the paper/report.

## Main Research Scripts

### `main.py`
Primary entry point for running the main SPIN simulation or experiment flow.

### `scaling_study.py`
Study script focused on how the method behaves as swarm size, topology size, or coordination complexity increases.

### `noise_robustness_study.py`
Experiment script for evaluating robustness under noise, perturbation, or imperfect sensing/communication assumptions.

### `plot_noise_degradation.py`
Plotting utility for visualizing degradation trends from the noise robustness study.

### `tensor_core_benchmark.py`
Benchmark script for measuring tensor-related computational performance or cost.

### `visualizer.py`
Visualization tool for swarm trajectories, spatial coverage, or experiment playback.

## Notes

This README intentionally omits the following folders from the structural walkthrough, per current repo-documentation scope:
- `output/`
- `legacy/`
- `reference/`

## Research Focus

From the current layout, this repository is best understood as an **experimental research codebase** rather than a polished application package. Its main purpose is to:
- prototype the SPIN coordination framework
- run comparative swarm-control experiments
- analyze scaling and robustness behavior
- generate result tables and visualizations for research reporting
