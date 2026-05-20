"""
Experiment: Stationarity and White Noise on Directed Graphs

This experiment investigates the stationarity properties of signals on directed graphs, specifically focusing on the behavior of white noise and its surrogates. We generate a directed graph based on a vortex flow pattern on an inverted parabola surface and analyze how white noise signals behave under the advection-diffusion operator defined on this graph. The experiment includes visualizations of the original signal and its surrogates to illustrate the effects of the graph structure on signal properties.
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import networkx as nx
import numpy as np


# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def run(
    save_results: bool = True, verbose: bool = True, recompute: bool = False
) -> dict:
    """
    Run the stationarity and white noise surrogates experiment.

    Parameters
    ----------
    save_results : bool
        Whether to save results to the results directory.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing experiment results.
    """
    from flowgsp.utils import (
        load_json,
        configure_experiment_logging,
        set_library_log_levels,
    )

    # Set up logging
    logger = configure_experiment_logging(
        experiment_name="stationarity",
        verbose=verbose,
        log_file=None
        if not save_results
        else os.path.join(RESULTS_DIR, "stationarity.log"),
        results_dir=RESULTS_DIR if save_results else None,
    )

    # Suppress noisy libraries
    if not verbose:
        set_library_log_levels("ERROR")

    logger.info("=" * 60)
    logger.info("Experiment: Stationarity and White Noise on Directed Graphs")
    logger.info("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config", "config.json"))

    logger.info(f"\nConfiguration: {config}")

    experiments = Experiments(
        config, verbose=verbose, logger=logger, recompute=recompute
    )
    # Plot 1: Plot Observed Signals and associated Surrogates
    fig = experiments.run_experiment1()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        from experiments.logging_utils import save_figures

        save_figures(
            figures=[fig],
            filenames=["wn-1d-plot_syn.png"],
            results_dir=RESULTS_DIR,
            logger=logger,
            dpi=300,
        )

        results_file = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {RESULTS_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 60)


class Experiments:
    def __init__(
        self, config: dict, verbose: bool = True, logger=None, recompute: bool = False
    ):
        self.config = config
        self.verbose = verbose
        self.logger = logger
        self.recompute = recompute
        self.path_to_resources = "./data/"

        from flowgsp.graphs import Graph
        from flowgsp.graphs.physical_graphs import (
            create_inverted_parabola_grid,
            create_mesh_graph,
            create_vortex_graph_surface,
        )
        from flowgsp.utils import (
            save,
            load,
        )

        if (
            os.path.exists(os.path.join(DATA_DIR, "parabola_graph.pkl"))
            and not self.recompute
        ):
            self.logger.info(f"Loading precomputed data from {DATA_DIR}...")
            # Load precomputed data if available
            self.graph_obj = load(os.path.join(DATA_DIR, "parabola_graph.pkl"))
            self.graph = self.graph_obj["graph"]
            self.surface_type = "inverted_parabola"
        else:
            positions_3d, positions_2d, X, Y, Z = create_inverted_parabola_grid(
                grid_size=config["grid_size"],
                parabola_scale=config["parabola_scale"],
                curve_scale=config["curve_scale"],
            )
            G, vortex_edges, cross_vectors = create_vortex_graph_surface(
                positions_3d,
                positions_2d,
                config["grid_size"],
                config["vortex_radius"],
                config["max_distance"],
            )
            self.surface_type = "inverted_parabola"

            G_mesh = create_mesh_graph(positions_3d, positions_2d, config["grid_size"])

            directed_array = nx.to_numpy_array(G)
            support_array = nx.to_numpy_array(G_mesh)

            # Combine mesh and vortex edges
            lbd = 0.5
            adj_matrix = lbd * directed_array + (1 - lbd) * support_array

            self.graph = Graph(adj_matrix=adj_matrix)
            self.graph.set_operator(name="advection_diffusion")
            self.graph_obj = {
                "graph": self.graph,
                "G": G,
                "adj_matrix": adj_matrix,
                "vortex_edges": vortex_edges,
                "cross_vectors": cross_vectors,
                "pos_3d": positions_3d,
                "pos_2d": positions_2d,
                "X": X,
                "Y": Y,
                "Z": Z,
                "N": len(positions_3d),
                "config": self.config,
            }
            save(os.path.join(DATA_DIR, "parabola_graph.pkl"), self.graph_obj)

    def run_experiment1(self):
        from flowgsp.utils.plot_mesh import plot_signal_on_regular_surface
        from flowgsp.surrogates import Surrogate

        surrogate = Surrogate(self.graph)

        graphsig = np.zeros((self.config["grid_size"], self.config["grid_size"]))
        center = self.config["grid_size"] // 2
        graphsig[center, center] = 1.0
        graphsig = graphsig.flatten()

        surrs = surrogate.directed_random_surrogate(
            graphsig, self.config["nb_repeat"], seed=99
        ).real
        sigs = [graphsig, surrs[0]]

        fig = plot_signal_on_regular_surface(
            X=self.graph_obj["X"],
            Y=self.graph_obj["Y"],
            Z=self.graph_obj["Z"],
            grid_size=int(np.sqrt(self.graph_obj["N"])),
            upsample=self.config["upsample_factor"],
            sigs=sigs,
            surface_type=self.surface_type,
            curve_scale=self.config["curve_scale"],
            parabola_scale=self.config["parabola_scale"],
            bump_scale=self.config["bump_scale"],
            labels=["Initial Signal", "Example Surrogate"],
            elev=30,
            azim=20,
            smooth_sigma=0.5,
        )

        return fig


if __name__ == "__main__":
    run()
