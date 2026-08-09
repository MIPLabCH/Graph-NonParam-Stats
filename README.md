## Graph Non Parametric Testing Via Surrogates

This is the repository gathering the experiments for the paper ``Statistical Testing on Directed Graphs by Surrogate Data Generation" (10.1109/TSIPN.2026.3717613). All figures found can be re-generated from this repository.

### Installation

Run the command:
`pip install GyRAPH`

### Run Experiments
```
# List all available experiments
python -m experiments.run --list

# Run a specific experiment
python -m experiments.run --paper paper_surrogates_stats --experiment exp_non-linearity
```

In each experiment folders are located config files you can change to modify the input parameters to experiments.

#### Contents

- Figures from paper ``Statistical Testing on Directed Graphs by Surrogate Data Generation"

### Core Module

This repository uses the core module [GyRAPH](https://github.com/miki998/GyRAPH/) implementation powering the Statistical Testing on Directed Graphs by Surrogate Data Generation experiments and all reproducible figures from the paper.
