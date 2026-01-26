# Experiment: Stationarity and White Noise on Directed Graphs

## Objectives

This experiment demonstrates the basic usage of the advection-diffusion operator for signal processing on directed graphs. The goals are:

1. Create a directed graph and set up the advection-diffusion operator
2. Generate a test signal on the graph
3. Apply spectral filtering using the advection-diffusion framework
4. Visualize and compare original vs filtered signals

## Setup

### Requirements

- FlowGSP package installed
- Python 3.8+

### Installation

```bash
# From the repository root
pip install -e .
```

## Running the Experiment

From the repository root:

```bash
# Using the CLI
python -m experiments.run --paper paper1_advection_diffusion --experiment exp1_advection_diffusion

# Or directly
python experiments/paper1_advection_diffusion/exp1_advection_diffusion/code/experiment.py
```

## Configuration

The experiment can be configured by modifying parameters in `code/experiment.py`:

- `n_nodes`: Number of nodes in the graph (default: 20)
- `edge_probability`: Probability of edge creation (default: 0.3)
- `noise_level`: Standard deviation of added noise (default: 0.5)
- `random_seed`: Random seed for reproducibility (default: 42)

## Expected Results

The experiment produces:
1. Signal statistics printed to console (MSE, improvement ratio)
2. Results saved to the `results/` directory as JSON
3. Results saved to the `results/` directory

## Directory Structure

```
exp1_advection_diffusion/
├── README.md          # This file
├── code/              # Experiment scripts
│   └── experiment.py  # Main experiment script
├── data/              # Input data (optional)
└── results/           # Output results
```
