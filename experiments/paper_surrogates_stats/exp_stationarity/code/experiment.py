"""
Experiment: Stationarity and White Noise on Directed Graphs
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm

# Import FlowGSP components
from flowgsp.utils import load
from flowgsp.graphs import Graph
from flowgsp.graphs.basic_graphs import create_cycle_graph
from flowgsp.surrogates import Surrogate

# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def run(save_results: bool = True, verbose: bool = True) -> dict:
    """
    Run experiment.

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

    experiments = Experiments(config, verbose=verbose)
    # Plot 1: Create flower graph and plot
    fig1, fig2, fig3 = experiments.run_experiment1()

    # Plot 2: Plot flower graph with original signal
    fig4, fig5, fig6 = experiments.run_experiment2()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig1.savefig(
            os.path.join(RESULTS_DIR, "wn-1d-plot_syn.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(RESULTS_DIR, "wn-variance_graph_visualization_syn.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(RESULTS_DIR, "wn-covar_syn.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(RESULTS_DIR, "wn-1d-plot_usa.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(RESULTS_DIR, "wn-variance_graph_visualization_usa.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig6.savefig(
            os.path.join(RESULTS_DIR, "wn-covar_usas.png"),
            dpi=300,
            bbox_inches="tight",
        )

        results_file = os.path.join(RESULTS_DIR, "experiment_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {RESULTS_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 60)


class Experiments:
    def __init__(self, config: dict, verbose: bool = True, logger=None):
        self.config = config
        self.verbose = verbose
        self.logger = logger
        self.path_to_resources = "./data/"

    def run_experiment1(self):
        N = self.config["length"]
        G, pos = create_cycle_graph(N, graph_type=1)
        custom = nx.to_numpy_array(G)
        for k in range(1, N // 2 + 2):
            custom[k, 0] = 1.0
        custom_u = (custom.T + custom).astype(float)

        vis_custom = deepcopy(custom)

        vis_custom[N // 2 : N // 2 + 2, :] = 0

        vis_custom[0, 12] = 1
        vis_custom[1:2, :] = 0
        vis_custom[:, 1:2] = 0

        G = nx.from_numpy_array(vis_custom, create_using=nx.DiGraph)
        pos = (
            nx.get_node_attributes(G, "pos")
            if "pos" in G.nodes[0]
            else nx.spring_layout(G)
        )

        rotate = 0
        cpoint_x, cpoint_y = (
            np.exp(1j * (np.linspace(0, 2 * np.pi, N // 2) - np.pi / 2 - rotate)).real,
            np.exp(1j * (np.linspace(0, 2 * np.pi, N // 2) - np.pi / 2 - rotate)).imag,
        )
        cpoint_y = cpoint_y + 1
        for k in range(1, N // 2 + 1):
            pos[k][0] = cpoint_x[k - 1]
            pos[k][1] = cpoint_y[k - 1]

        cpoint_x, cpoint_y = (
            np.exp(1j * (np.linspace(0, 2 * np.pi, N // 2) + np.pi / 2 + rotate)).real,
            np.exp(1j * (np.linspace(0, 2 * np.pi, N // 2) + np.pi / 2 + rotate)).imag,
        )
        cpoint_y = cpoint_y - 1
        for k in range(N // 2 + 1, N):
            pos[k][0] = cpoint_x[k - 1 - N // 2]
            pos[k][1] = cpoint_y[k - 1 - N // 2]

        graph = Graph(adj_matrix=custom, pos=pos)
        graph.set_operator("adjacency")
        surrogate = Surrogate(graph)

        graph_u = Graph(adj_matrix=custom_u, pos=pos)
        graph_u.set_operator("adjacency")
        surrogate_u = Surrogate(graph_u)

        # Plot Mean and Std Dev
        distrib_z = surrogate.white_noise_generator(self.config["nb_repeat"])
        distrib_undz = surrogate_u.white_noise_generator(self.config["nb_repeat"])

        mean_distrib_z = np.mean(distrib_z, axis=0)
        std_distrib_z = np.std(distrib_z, axis=0)

        mean_distrib_undz = np.mean(distrib_undz, axis=0)
        std_distrib_undz = np.std(distrib_undz, axis=0)

        fig1, ax = plt.subplots(1, figsize=(4, 2))
        ax.plot(mean_distrib_z, label="Direct", color="magenta")
        ax.fill_between(
            range(len(mean_distrib_z)),
            mean_distrib_z - std_distrib_z,
            mean_distrib_z + std_distrib_z,
            alpha=0.3,
            color="magenta",
        )

        ax.plot(mean_distrib_undz, label="Undirect", color="k")
        ax.fill_between(
            range(len(mean_distrib_undz)),
            mean_distrib_undz - std_distrib_undz,
            mean_distrib_undz + std_distrib_undz,
            alpha=0.3,
            color="k",
        )

        # ax.set_xlabel('Node Index', fontname='Helvetica', fontsize=10)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            labelcolor="black",
            labelrotation=0,
            labeltop=False,
            labelright=False,
            labelbottom=True,
            labelleft=True,
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Helvetica")
        ax.legend(prop={"family": "Helvetica", "size": 10})

        if not self.verbose:
            plt.close()
        plt.show()

        # Plot Variance on Graph
        covariance_dir = surrogate.exact_covariance(np.eye(N)).real
        cmap = cm.get_cmap("plasma")
        fig2, ax = plt.subplots(1, figsize=(2, 3))
        graph.draw_signal(
            np.diag(covariance_dir),
            axes=ax,
            arrow_size=5,
            arrow_width=1,
            cmap=cmap,
            edgecolors="black",
            nodetype="color",
            node_size=100,
        )
        ax.set_axis_off()
        if not self.verbose:
            plt.close()
        plt.show()

        # Plot covariance matrix
        fig3, ax = plt.subplots(1, figsize=(4, 2))
        im = ax.imshow(covariance_dir, cmap="plasma")

        ax.plot(0.5 + np.arange(23), -0.5 + np.ones(23), linewidth=2, color="g")
        ax.plot(
            0.5 + np.arange(23),
            -0.5 + 23 * np.ones(23),
            linewidth=2,
            color="g",
            linestyle="solid",
        )
        ax.plot(
            -0.5 + np.ones(23),
            0.5 + np.arange(23),
            linewidth=2,
            color="g",
            linestyle="solid",
        )
        ax.plot(
            -0.5 + 23 * np.ones(23),
            0.5 + np.arange(23),
            linewidth=2,
            color="g",
            linestyle="solid",
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)  # Increase the tick size

        ticks = [-0.4, 0.0, 0.4, 0.8, 1.2]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t: .2f}" for t in ticks])
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("Helvetica")
            tick.set_fontsize(8)

        ax.set_xticks([])
        ax.set_yticks([])
        # Set font to Helvetica
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 10
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3

    def run_experiment2(self):
        G = nx.from_numpy_array(
            load(
                os.path.join(
                    self.path_to_resources, "usa_graph_data", "diag_usagraph.pkl"
                )
            ),
            create_using=nx.DiGraph(),
        )
        Gu = nx.from_numpy_array(
            load(
                os.path.join(
                    self.path_to_resources, "usa_graph_data", "diag_usagraph.pkl"
                )
            ),
            create_using=nx.Graph(),
        )
        pos = load(
            os.path.join(self.path_to_resources, "usa_graph_data", "state_coords.pkl")
        )

        graph = Graph(G=G, pos=pos)
        graph.set_operator("adjacency")
        surrogate = Surrogate(graph)

        graph_u = Graph(G=Gu, pos=pos)
        graph_u.set_operator("adjacency")
        surrogate_u = Surrogate(graph_u)
        N = graph.N
        # Flip the direction of all edges in the directed graph
        graph.G = graph.G.reverse(
            copy=True
        )  # Flip directions due to networkx drawing convention

        distrib_z = surrogate.white_noise_generator(self.config["nb_repeat"])
        distrib_undz = surrogate_u.white_noise_generator(self.config["nb_repeat"])

        mean_distrib_z = np.mean(distrib_z, axis=0)
        std_distrib_z = np.std(distrib_z, axis=0)

        mean_distrib_undz = np.mean(distrib_undz, axis=0)
        std_distrib_undz = np.std(distrib_undz, axis=0)

        fig1, ax = plt.subplots(1, figsize=(4, 2))
        ax.plot(mean_distrib_z, label="Direct", color="magenta")
        ax.fill_between(
            range(len(mean_distrib_z)),
            mean_distrib_z - std_distrib_z,
            mean_distrib_z + std_distrib_z,
            alpha=0.3,
            color="magenta",
        )

        ax.plot(mean_distrib_undz, label="Undirect", color="k")
        ax.fill_between(
            range(len(mean_distrib_undz)),
            mean_distrib_undz - std_distrib_undz,
            mean_distrib_undz + std_distrib_undz,
            alpha=0.3,
            color="k",
        )

        # ax.set_xlabel('Node Index', fontname='Helvetica', fontsize=10)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            labelcolor="black",
            labelrotation=0,
            labeltop=False,
            labelright=False,
            labelbottom=True,
            labelleft=True,
        )
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Helvetica")
        ax.legend(prop={"family": "Helvetica", "size": 10})

        if not self.verbose:
            plt.close()
        plt.show()

        # Plot Variance on Graph
        covariance_dir = surrogate.exact_covariance(np.eye(N)).real
        cmap = cm.get_cmap("plasma")
        fig2, ax = plt.subplots(1, figsize=(3, 2))
        graph.draw_signal(
            np.diag(covariance_dir),
            axes=ax,
            arrow_size=5,
            arrow_width=1,
            cmap=cmap,
            edgecolors="black",
            nodetype="color",
            node_size=100,
        )
        ax.set_axis_off()

        if not self.verbose:
            plt.close()
        plt.show()

        # Plot reordered covariance matrix
        # Reorder covariance matrix by latitude
        ordered_latitude = np.array(list(pos.values()))[:, 1].argsort()
        covariance_dir_reordered = covariance_dir[ordered_latitude, :][
            :, ordered_latitude
        ]
        fig3, ax = plt.subplots(1, figsize=(4, 2))
        # im = ax.imshow(covariance_dir)
        im = ax.imshow(covariance_dir_reordered, vmin=None, cmap="plasma")

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)  # Increase the tick size
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname("Helvetica")
            tick.set_fontsize(8)

        ticks = [-0.5, 0.5, 1.5, 2.5]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t: .2f}" for t in ticks])
        ax.set_xticks([])
        ax.set_yticks([])

        # Set font to Helvetica
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 8

        ax.axis("off")
        if not self.verbose:
            plt.close()
        plt.show()

        return fig1, fig2, fig3


if __name__ == "__main__":
    run()
