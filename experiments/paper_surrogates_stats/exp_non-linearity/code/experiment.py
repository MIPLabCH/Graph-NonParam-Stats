"""
Experiment: Graph Non-Linearity Testing
"""

import os
import json
from datetime import datetime

# Import necessary libraries
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests


# Import FlowGSP components
from flowgsp.graphs import Graph
from flowgsp.surrogates import Surrogate
from flowgsp.utils import load, p_value, save

from experiments.paper_surrogates_stats.paper_utils import (
    timeseries_null_models_generator,
    interpolate_nans,
    rasterize,
)

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
        experiment_name="non_linearity",
        verbose=verbose,
        log_file=None
        if not save_results
        else os.path.join(RESULTS_DIR, "non_linearity.log"),
        results_dir=RESULTS_DIR if save_results else None,
    )

    # Suppress noisy libraries
    if not verbose:
        set_library_log_levels("ERROR")

    logger.info("=" * 60)
    logger.info("Experiment: Graph Non-Linearity Testing")
    logger.info("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config", "config.json"))

    logger.info(f"\nConfiguration: {config}")

    (
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
        fig6,
        fig7,
        fig8,
        fig9,
        fig10,
        fig11,
        fig12,
        fig13,
        fig14,
        fig15,
        fig16,
        fig17,
    ) = (None,) * 17
    experiments = Experiments(config, verbose=verbose, logger=logger)
    # Plot 1: Singularities detection accuracy vs number of nodes
    logger.info(
        "\nRunning Experiment 1: Singularities detection accuracy vs number of nodes"
    )
    fig1, fig2, fig3, fig4, fig5 = experiments.run_experiment1()

    logger.info("\nRunning Experiment 2: Covariance detection")
    # Plot 2: Covariance detection
    fig6, fig7, fig8, fig9 = experiments.run_experiment2()
    if verbose:
        logger.info(
            "\nRunning Experiment 3: Covariance detection with denser covariance"
        )
    # Plot 3: Covariance detection with denser covariance
    fig10 = experiments.run_experiment3()

    logger.info("\nRunning Experiment 4: Singularities detection in Diffusion model")
    # Plot 4: Singularities detection in Diffusion model
    fig11, fig12, fig13, fig14 = experiments.run_experiment4()

    logger.info(
        "\nRunning Experiment 5: Denser Singularities detection in Diffusion model"
    )
    # Plot 5: Singularities detection in Diffusion model
    fig15, fig16 = experiments.run_experiment5()
    logger.info(
        "\nRunning Experiment 6: AUC across densities for Singularities detection in Diffusion model"
    )
    # Plot 6: AUC across densities for Singularities detection in Diffusion model
    fig17 = experiments.run_experiment6()

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if fig1 is not None:
            fig1.savefig(
                os.path.join(RESULTS_DIR, "increasing_nodes.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig2 is not None:
            fig2.savefig(
                os.path.join(RESULTS_DIR, "ground_truth.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig3 is not None:
            fig3.savefig(
                os.path.join(RESULTS_DIR, "statistics_directed.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig4 is not None:
            fig4.savefig(
                os.path.join(RESULTS_DIR, "statistics_undirected.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig5 is not None:
            fig5.savefig(
                os.path.join(RESULTS_DIR, "statistics_naive.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig6 is not None:
            fig6.savefig(
                os.path.join(RESULTS_DIR, "predicted_connectivity_directed.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig7 is not None:
            fig7.savefig(
                os.path.join(RESULTS_DIR, "predicted_connectivity_undirected.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig8 is not None:
            fig8.savefig(
                os.path.join(RESULTS_DIR, "predicted_connectivity_naive.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig9 is not None:
            fig9.savefig(
                os.path.join(RESULTS_DIR, "ground_truth_connectivity.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig10 is not None:
            fig10.savefig(
                os.path.join(RESULTS_DIR, "covariance_denser.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig11 is not None:
            fig11.savefig(
                os.path.join(RESULTS_DIR, "one_diffusion_timecourse.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig12 is not None:
            fig12.savefig(
                os.path.join(RESULTS_DIR, "one_null_distribution_diffusion_sig.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig13 is not None:
            fig13.savefig(
                os.path.join(RESULTS_DIR, "one_null_distribution_diffusion_nonsig.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig14 is not None:
            fig14.savefig(
                os.path.join(RESULTS_DIR, "one_roc_curve.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig15 is not None:
            fig15.savefig(
                os.path.join(RESULTS_DIR, "multi_diffusion_timecourse.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig16 is not None:
            fig16.savefig(
                os.path.join(RESULTS_DIR, "multi_roc_curve.png"),
                dpi=300,
                bbox_inches="tight",
            )
        if fig17 is not None:
            fig17.savefig(
                os.path.join(RESULTS_DIR, "auc_across_densities.png"),
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

        self.fontsize = 10

        self.G = nx.from_numpy_array(
            load(
                os.path.join(
                    self.path_to_resources, "usa_graph_data", "diag_usagraph.pkl"
                )
            ),
            create_using=nx.DiGraph(),
        )
        self.Gu = nx.from_numpy_array(
            load(
                os.path.join(
                    self.path_to_resources, "usa_graph_data", "diag_usagraph.pkl"
                )
            ),
            create_using=nx.Graph(),
        )
        self.pos = load(
            os.path.join(self.path_to_resources, "usa_graph_data", "state_coords.pkl")
        )

        self.graph = Graph(G=self.G, pos=self.pos)
        self.graph.set_operator("adjacency", normalize="left")
        self.surrogate = Surrogate(self.graph)

        self.graph_u = Graph(G=self.Gu, pos=self.pos)
        self.graph_u.set_operator("adjacency", normalize="left")
        self.surrogate_u = Surrogate(self.graph_u)

    def run_experiment1(self):
        mult = self.graph.N
        nrands = (
            self.config["nb_rands"] + 1
        ) * mult  # to have enough surrogates after correction
        nb_samples = self.config["nb_samples"]

        number_of_disrupts = [5, 9, 14, 19, 24]
        alpha = self.config["alpha"]

        recompute_scores = False
        if os.path.exists(os.path.join(DATA_DIR, "singular_scores.pkl")):
            try:
                (
                    scores_dir,
                    scores_und,
                    scores_naive,
                    ground,
                    twotail_pvals,
                    twotail_pvals2,
                    twotail_pvals3,
                ) = load(os.path.join(DATA_DIR, "singular_scores.pkl"))
                if set(scores_dir.keys()) != set(number_of_disrupts):
                    recompute_scores = True
            except Exception as e:
                if self.verbose:
                    print(f"Error loading scores: {e}")
                recompute_scores = True
        else:
            recompute_scores = True
        if recompute_scores:
            covariance_matrix = self.surrogate.exact_covariance(
                np.eye(self.graph.N)
            ).real
            np.testing.assert_almost_equal(covariance_matrix, covariance_matrix.T)

            scores_dir = {nbdisrupt: [] for nbdisrupt in number_of_disrupts}
            scores_und = {nbdisrupt: [] for nbdisrupt in number_of_disrupts}
            scores_naive = {nbdisrupt: [] for nbdisrupt in number_of_disrupts}
            np.random.seed(0)
            for nbdisrupt in tqdm(
                number_of_disrupts, desc="Number of Disrupts", disable=not self.verbose
            ):
                gaussian_samples = np.random.multivariate_normal(
                    np.zeros(self.graph.N), covariance_matrix.real, nb_samples
                )
                for sample_idx in range(nb_samples):
                    dirac_signal = np.zeros(self.graph.N)
                    ground = np.zeros(self.graph.N)
                    selected = np.random.choice(
                        np.arange(self.graph.N), nbdisrupt, replace=False
                    )
                    ground[selected] = 1
                    dirac_signal[selected] = 10
                    dirac_signal = dirac_signal + gaussian_samples[sample_idx]

                    null_distrib_naive = self.surrogate_u.naive_random_surrogate(
                        dirac_signal, nrands=nrands
                    )
                    null_distrib_undirect = (
                        self.surrogate_u.undirected_random_surrogate(
                            dirac_signal, nrands=nrands
                        )
                    )
                    null_distrib_direct = self.surrogate.directed_random_surrogate(
                        dirac_signal, nrands=nrands
                    )

                    twotail_pvals = 1 - np.array(
                        [
                            p_value(
                                null_distrib_direct[:, nidx],
                                dirac_signal[nidx],
                                two_tail=True,
                            )
                            for nidx in range(self.graph.N)
                        ]
                    )
                    twotail_pvals2 = 1 - np.array(
                        [
                            p_value(
                                null_distrib_undirect[:, nidx],
                                dirac_signal[nidx],
                                two_tail=True,
                            )
                            for nidx in range(self.graph.N)
                        ]
                    )
                    twotail_pvals3 = 1 - np.array(
                        [
                            p_value(
                                null_distrib_naive[:, nidx],
                                dirac_signal[nidx],
                                two_tail=True,
                            )
                            for nidx in range(self.graph.N)
                        ]
                    )

                    if self.config["correction"] == "bonferroni":
                        decision = (1 - twotail_pvals) <= alpha / mult
                        decision2 = (1 - twotail_pvals2) <= alpha / mult
                        decision3 = (1 - twotail_pvals3) <= alpha / mult

                    elif self.config["correction"] == "fdr":
                        decision, _, _, _ = multipletests(
                            1 - twotail_pvals,
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )
                        decision2, _, _, _ = multipletests(
                            1 - twotail_pvals2,
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )
                        decision3, _, _, _ = multipletests(
                            1 - twotail_pvals3,
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )

                    else:
                        raise ValueError("Unknown correction method")

                    score = np.mean(decision == ground)
                    score2 = np.mean(decision2 == ground)
                    score3 = np.mean(decision3 == ground)
                    scores_dir[nbdisrupt].append(score)
                    scores_und[nbdisrupt].append(score2)
                    scores_naive[nbdisrupt].append(score3)

            save(
                os.path.join(DATA_DIR, "singular_scores.pkl"),
                (
                    scores_dir,
                    scores_und,
                    scores_naive,
                    ground,
                    twotail_pvals,
                    twotail_pvals2,
                    twotail_pvals3,
                ),
            )

        # Artificially adding some very small variance to avoid issues with boxplot in color
        var = 4e-3
        np.random.seed(99)
        for k in scores_dir:
            scores_dir[k] = np.array(scores_dir[k]) + np.random.normal(
                0, var, len(scores_dir[k])
            )
        for k in scores_und:
            scores_und[k] = np.array(scores_und[k]) + np.random.normal(
                0, var, len(scores_und[k])
            )
        for k in scores_naive:
            scores_naive[k] = np.array(scores_naive[k]) + np.random.normal(
                0, var, len(scores_naive[k])
            )

        # Prepare data for plotting
        labels = []
        for nbdisrupt in number_of_disrupts:
            perc = int(10 * np.round(10 * nbdisrupt / self.graph.N))
            labels.append(f"{perc}%")
        data = []
        for nbdisrupt in number_of_disrupts:
            for score in scores_dir[nbdisrupt]:
                data.append(["Directed", nbdisrupt, score])
            for score in scores_und[nbdisrupt]:
                data.append(["Undirected", nbdisrupt, score])
            for score in scores_naive[nbdisrupt]:
                data.append(["Naive", nbdisrupt, score])

        df_plot = pd.DataFrame(data, columns=["Type", "Number of Disruptions", "Score"])

        # Plotting boxplots with increasing significant nodes
        fig1, ax = plt.subplots(figsize=(5, 3))

        sns.boxplot(
            x="Number of Disruptions",
            y="Score",
            hue="Type",
            data=df_plot,
            showmeans=False,
            meanline=False,
            palette={"Directed": "red", "Undirected": "blue", "Naive": "black"},
            linewidth=0.5,
            boxprops=dict(alpha=1),  # keep box fully opaque
            whiskerprops=dict(linewidth=0.2),
            capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=0.5, alpha=0.5),
        )

        # Make box edges more transparent after plotting
        for artist in ax.artists:
            artist.set_edgecolor((0, 0, 0, 0.3))  # RGBA, alpha=0.3 for transparency

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        # Labels and title
        ax.set_xlabel("", fontsize=self.fontsize, fontname="Helvetica", labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=self.fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.fontsize)
        # Ensure all tick labels (major and minor) use Helvetica
        for lbl in (
            ax.xaxis.get_majorticklabels()
            + ax.xaxis.get_minorticklabels()
            + ax.yaxis.get_majorticklabels()
            + ax.yaxis.get_minorticklabels()
        ):
            lbl.set_fontname("Helvetica")
        ax.set_ylabel(
            "Accuracy", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
        )
        ax.set_xticklabels(labels, fontsize=self.fontsize, fontname="Helvetica")

        for t in [0.5 + i for i in range(len(number_of_disrupts))]:
            ax.axvline(x=t, color="black", linestyle="--", linewidth=1, alpha=0.1)
        # Customize legend
        ax.legend(
            loc="upper right", prop={"family": "Helvetica", "size": self.fontsize}
        )
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        # Adjust layout
        plt.tight_layout()

        if not self.verbose:
            plt.close()
        plt.show()

        # Graph visualization with p-values
        from mpl_toolkits.basemap import Basemap

        scale = 10
        stats_values = [ground, twotail_pvals, twotail_pvals2, twotail_pvals3]

        figs = []
        if self.verbose:
            print(
                "Minimum stats in ground truth: ",
                np.min(twotail_pvals[np.array(ground).astype(bool)]),
            )
            print(
                "Maximum stats not in ground truth: ",
                np.max(twotail_pvals[~np.array(ground).astype(bool)]),
            )

        for sidx in range(len(stats_values)):
            fig, ax = plt.subplots(figsize=(5, 3))
            map = Basemap(
                llcrnrlon=-119,
                llcrnrlat=22,
                urcrnrlon=-64,
                urcrnrlat=49,
                projection="lcc",
                lat_1=32,
                lat_2=45,
                lon_0=-95,
                ax=ax,
            )

            # load the shapefile, use the name 'states'
            map.readshapefile(
                os.path.join(self.path_to_resources, "usa_graph_data/st99_d00"),
                name="states",
                drawbounds=False,
            )

            if sidx == 0:
                cm = plt.get_cmap("PuBuGn")
            else:
                cm = plt.get_cmap("inferno")

            # Plot directed edges of graph G
            for edge in self.G.edges():
                start_pos = self.pos[edge[0]]
                end_pos = self.pos[edge[1]]
                start_x, start_y = map(start_pos[0], start_pos[1])
                end_x, end_y = map(end_pos[0], end_pos[1])
                ax.annotate(
                    "",
                    xy=(end_x, end_y),
                    xycoords="data",
                    xytext=(start_x, start_y),
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        lw=1,
                        connectionstyle="arc3,rad=0.1",
                        shrinkA=6,
                        shrinkB=6,
                        mutation_scale=5,
                    ),
                    zorder=-2,
                )

            for k in range(self.graph.N):
                current_pts = self.pos[k]
                x, y = map(current_pts[0], current_pts[1])
                map.plot(
                    x,
                    y,
                    marker="o",
                    color=cm(stats_values[sidx][k]),
                    markersize=scale,
                    markeredgecolor="black",
                    alpha=0.9,
                )

            # Draw the states with a gray background
            for shape_dict in map.states_info:
                seg = map.states[map.states_info.index(shape_dict)]
                poly = plt.Polygon(seg, facecolor="gray", edgecolor="black", alpha=0.2)
                plt.gca().add_patch(poly)

            # Create a scatter plot for the colorbar
            if sidx == 0:
                sc = map.scatter(
                    [], [], c=[], cmap=cm, vmin=ground.min(), vmax=ground.max(), ax=ax
                )
            else:
                sc = map.scatter([], [], c=[], cmap=cm, vmin=0, vmax=1, ax=ax)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
            cbar.ax.tick_params(labelsize=self.fontsize - 2)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname("Helvetica")
            if sidx == 0:
                cbar.set_ticks([0, 1])
                cbar.set_ticklabels(["Normal", "Singular"])
            else:
                cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.axis("off")

            if not self.verbose:
                plt.close()
            plt.show()
            figs.append(fig)

        return fig1, figs[0], figs[1], figs[2], figs[3]

    def run_experiment2(self):
        mult = int(self.graph.N * (self.graph.N - 1) / 2)
        nb_samples = self.config["nb_samples"]
        nrands = int(
            (self.config["nb_rands"] + 1) * mult / nb_samples
        )  # to have enough surrogates after correction

        if os.path.exists(os.path.join(DATA_DIR, "covariance_illustration.pkl")):
            (
                offdiagonals,
                covariance_pvals_direct,
                covariance_pvals_undirect,
                covariance_pvals_naive,
            ) = load(os.path.join(DATA_DIR, "covariance_illustration.pkl"))
        else:
            np.random.seed(95)
            # Generate a vector of Gaussian distributed samples with auto covariance as input
            base_covariance_matrix = self.surrogate.exact_covariance(
                np.eye(self.graph.N)
            ).real
            offdiagonals = (
                np.random.random((self.graph.N, self.graph.N)) > 0.996
            ).astype(float)

            offdiagonals = offdiagonals - np.diag(offdiagonals.diagonal())
            offdiagonals = offdiagonals + offdiagonals.T
            offdiagonals *= 0.7

            covariance_matrix = base_covariance_matrix + offdiagonals
            np.testing.assert_almost_equal(covariance_matrix, covariance_matrix.T)

            # Project modif_joint_covar to the nearest positive semi-definite matrix
            eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
            eigvals[eigvals < 0] = 0
            covariance_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

            gaussian_samples = np.random.multivariate_normal(
                np.zeros(self.graph.N), covariance_matrix, nb_samples, tol=1e-8
            )  # correlated gaussians

            empirical_cov = self.surrogate_u.estimate_covariance(gaussian_samples)

            # Randomized Correlated Gaussian Surrogates Naive
            randomized_correlated_gaussian_naive = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.surrogate_u.naive_random_surrogate)(
                        gaussian_samples[k], nrands=nrands, seed=99
                    )
                    for k in tqdm(
                        range(nb_samples),
                        desc="Naive Surrogates",
                        disable=not self.verbose,
                    )
                )
            )

            covariance_sample_randomized_naive = np.array(
                [
                    self.surrogate_u.estimate_covariance(
                        randomized_correlated_gaussian_naive[:, k]
                    )
                    for k in range(nrands)
                ]
            )

            # Randomized Correlated Gaussian Surrogates Undirected
            randomized_correlated_gaussian_undirect = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.surrogate_u.undirected_random_surrogate)(
                        gaussian_samples[k], nrands=nrands, seed=99
                    )
                    for k in tqdm(
                        range(nb_samples),
                        desc="Undirected Surrogates",
                        disable=not self.verbose,
                    )
                )
            )

            covariance_sample_randomized_undirect = np.array(
                [
                    self.surrogate_u.estimate_covariance(
                        randomized_correlated_gaussian_undirect[:, k]
                    )
                    for k in range(nrands)
                ]
            )

            # Randomized Correlated Gaussian Surrogates Directed
            randomized_correlated_gaussian_direct = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.surrogate.directed_random_surrogate)(
                        gaussian_samples[k], nrands=nrands, seed=99
                    )
                    for k in tqdm(
                        range(nb_samples),
                        desc="Directed Surrogates",
                        disable=not self.verbose,
                    )
                )
            )
            covariance_sample_randomized_direct = np.array(
                [
                    self.surrogate.estimate_covariance(
                        randomized_correlated_gaussian_direct[:, k]
                    )
                    for k in range(nrands)
                ]
            )

            covariance_pvals_direct = 1 - np.array(
                [
                    [
                        p_value(
                            covariance_sample_randomized_direct[:, n, m],
                            empirical_cov[n, m],
                            two_tail=True,
                        )
                        for n in range(self.graph.N)
                    ]
                    for m in range(self.graph.N)
                ]
            )

            covariance_pvals_undirect = 1 - np.array(
                [
                    [
                        p_value(
                            covariance_sample_randomized_undirect[:, n, m],
                            empirical_cov[n, m],
                            two_tail=True,
                        )
                        for n in range(self.graph.N)
                    ]
                    for m in range(self.graph.N)
                ]
            )

            covariance_pvals_naive = 1 - np.array(
                [
                    [
                        p_value(
                            covariance_sample_randomized_naive[:, n, m],
                            empirical_cov[n, m],
                            two_tail=True,
                        )
                        for n in range(self.graph.N)
                    ]
                    for m in range(self.graph.N)
                ]
            )

            save(
                os.path.join(DATA_DIR, "covariance_illustration.pkl"),
                (
                    offdiagonals,
                    covariance_pvals_direct,
                    covariance_pvals_undirect,
                    covariance_pvals_naive,
                ),
            )

        if self.config["correction"] == "bonferroni":
            # Bonferroni correction
            display_pvals_direct = (
                (1 - covariance_pvals_direct) <= (self.config["alpha"] / mult)
            ).astype(int)
            display_pvals_undirect = (
                (1 - covariance_pvals_undirect) <= (self.config["alpha"] / mult)
            ).astype(int)
            display_pvals_naive = (
                (1 - covariance_pvals_naive) <= (self.config["alpha"] / mult)
            ).astype(int)

        # FDR correction
        elif self.config["correction"] == "fdr":
            # As we compare all pairs (i,j) with i<j, we flatten the upper triangle of the matrix
            pvals_direct, _, _, _ = multipletests(
                1 - covariance_pvals_direct[np.triu_indices(self.graph.N)],
                alpha=self.config["alpha"],
                method="fdr_bh",
            )
            display_pvals_direct = np.zeros((self.graph.N, self.graph.N))
            display_pvals_direct[np.triu_indices(self.graph.N)] = pvals_direct
            display_pvals_direct = (display_pvals_direct + display_pvals_direct.T) > 0

            pvals_undirect, _, _, _ = multipletests(
                1 - covariance_pvals_undirect[np.triu_indices(self.graph.N)],
                alpha=self.config["alpha"],
                method="fdr_bh",
            )
            display_pvals_undirect = np.zeros((self.graph.N, self.graph.N))
            display_pvals_undirect[np.triu_indices(self.graph.N)] = pvals_undirect
            display_pvals_undirect = (
                display_pvals_undirect + display_pvals_undirect.T
            ) > 0

            pvals_naive, _, _, _ = multipletests(
                1 - covariance_pvals_naive[np.triu_indices(self.graph.N)],
                alpha=self.config["alpha"],
                method="fdr_bh",
            )
            display_pvals_naive = np.zeros((self.graph.N, self.graph.N))
            display_pvals_naive[np.triu_indices(self.graph.N)] = pvals_naive
            display_pvals_naive = (display_pvals_naive + display_pvals_naive.T) > 0

        else:
            raise ValueError("Unknown correction method")

        figs = []

        displays = [display_pvals_direct, display_pvals_undirect, display_pvals_naive]
        colors = ["red", "blue", "gray"]
        labels = ["direct", "undirect", "naive"]
        for idx, display_mat in enumerate(displays):
            fig, ax = plt.subplots(figsize=(3, 3))
            display_mat[
                np.eye(self.graph.N).astype(bool)
            ] = 0  # remove diagonal for visualization
            ax.imshow(display_mat, cmap="gray", interpolation="nearest")
            display_coords = np.argwhere(display_mat > 0)
            for _, coord in enumerate(display_coords):
                rect = patches.Rectangle(
                    (coord[1] - 0.5, coord[0] - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor=colors[idx],
                    facecolor=colors[idx],
                )
                ax.add_patch(rect)

            ax.set_xlabel(
                "Nodes", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
            )
            ax.set_ylabel(
                "Nodes", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
            )
            # Get the coordinates of the yellow blocks
            true_coords = np.argwhere(offdiagonals > 0)
            # Add a rectangle around each yellow block
            for _, coord in enumerate(true_coords):
                rect = patches.Rectangle(
                    (coord[1] - 0.5, coord[0] - 0.5),
                    1,
                    1,
                    linewidth=1.2,
                    edgecolor="green",
                    facecolor="none",
                )
                ax.add_patch(rect)

            patch = patches.Patch(
                facecolor=colors[idx], label=labels[idx], edgecolor=colors[idx]
            )
            ax.legend(
                handles=[patch],
                loc=(0.36, 1.02),
                prop={"family": "Helvetica", "size": 8},
            )

            ax.set_xticks([])
            ax.set_yticks([])

            figs.append(fig)
            if not self.verbose:
                plt.close()
            plt.show()

        # Ground truth visualization
        fig2, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(offdiagonals, cmap="gray", interpolation="nearest")
        ax.set_xlabel(
            "Nodes", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
        )
        ax.set_ylabel(
            "Nodes", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
        )

        yellow_coords = np.argwhere(offdiagonals > 0)

        # Add a rectangle around each block
        for cidx, coord in enumerate(yellow_coords):
            if cidx == 0:
                rect = patches.Rectangle(
                    (coord[1] - 0.5, coord[0] - 0.5),
                    1,
                    1,
                    linewidth=1.2,
                    edgecolor="green",
                    facecolor="white",
                    label="ground truth",
                )
            else:
                rect = patches.Rectangle(
                    (coord[1] - 0.5, coord[0] - 0.5),
                    1,
                    1,
                    linewidth=1.2,
                    edgecolor="green",
                    facecolor="white",
                )
            ax.add_patch(rect)

        ax.legend(loc=(0.3, 1.02), prop={"family": "Helvetica", "size": 8})

        ax.set_xticks([])
        ax.set_yticks([])
        figs.append(fig2)
        if not self.verbose:
            plt.close()
        plt.show()

        return figs

    def run_experiment3(self):
        np.random.seed(95)
        mult = int(self.graph.N * (self.graph.N - 1) / 2)
        nb_repeat = 10
        nb_samples = self.config["nb_samples"]
        nrands = int(
            (self.config["nb_rands"] + 1) * mult / nb_samples
        )  # to have enough surrogates after correction

        proportion = [0.895, 0.95, 0.975, 0.99, 0.999][::-1]

        scores_dir = {prop: [] for prop in proportion}
        scores_und = {prop: [] for prop in proportion}
        scores_naive = {prop: [] for prop in proportion}
        samples = {prop: [] for prop in proportion}

        recompute_scores = False
        if os.path.exists(os.path.join(DATA_DIR, "covariance_scores.pkl")):
            try:
                scores_dir, scores_und, scores_naive, samples = load(
                    os.path.join(DATA_DIR, "covariance_scores.pkl")
                )
                if set(scores_dir.keys()) != set(proportion):
                    recompute_scores = True
            except Exception as e:
                if self.verbose:
                    print(f"Error loading scores: {e}")
                recompute_scores = True
        else:
            recompute_scores = True
        if recompute_scores:
            for prop in proportion:
                for _ in tqdm(range(nb_repeat)):
                    # Generate observation model
                    covariance_matrix = self.surrogate.exact_covariance(
                        np.eye(self.graph.N)
                    ).real
                    offdiagonals = (
                        np.random.random((self.graph.N, self.graph.N)) > prop
                    ).astype(float)
                    offdiagonals = offdiagonals - np.diag(offdiagonals.diagonal())
                    offdiagonals = offdiagonals + offdiagonals.T
                    offdiagonals = offdiagonals * 0.5

                    covariance_matrix = covariance_matrix + offdiagonals

                    # Project modif_joint_covar to the nearest positive semi-definite matrix
                    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
                    eigvals[eigvals < 0] = 0
                    covariance_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

                    np.testing.assert_almost_equal(
                        covariance_matrix, covariance_matrix.T
                    )
                    gaussian_samples = np.random.multivariate_normal(
                        np.zeros(self.graph.N), covariance_matrix, nb_samples, tol=1e-8
                    )  # correlated gaussians
                    empirical_covariance = self.surrogate_u.estimate_covariance(
                        gaussian_samples
                    )

                    # Randomized Correlated Gaussian Surrogates
                    randomized_correlated_gaussian_naive = np.array(
                        Parallel(n_jobs=-1)(
                            delayed(self.surrogate_u.naive_random_surrogate)(
                                gaussian_samples[k], nrands=nrands, seed=99
                            )
                            for k in range(nb_samples)
                        )
                    )
                    covariance_sample_randomized_naive = np.array(
                        [
                            self.surrogate_u.estimate_covariance(
                                randomized_correlated_gaussian_naive[:, k]
                            )
                            for k in range(nrands)
                        ]
                    )

                    randomized_correlated_gaussian_undirect = np.array(
                        Parallel(n_jobs=-1)(
                            delayed(self.surrogate_u.undirected_random_surrogate)(
                                gaussian_samples[k], nrands=nrands, seed=99
                            )
                            for k in range(nb_samples)
                        )
                    )
                    covariance_sample_randomized_undirect = np.array(
                        [
                            self.surrogate_u.estimate_covariance(
                                randomized_correlated_gaussian_undirect[:, k]
                            )
                            for k in range(nrands)
                        ]
                    )

                    randomized_correlated_gaussian_direct = np.array(
                        Parallel(n_jobs=-1)(
                            delayed(self.surrogate.directed_random_surrogate)(
                                gaussian_samples[k], nrands=nrands, seed=99
                            )
                            for k in range(nb_samples)
                        )
                    )
                    covariance_sample_randomized_direct = np.array(
                        [
                            self.surrogate.estimate_covariance(
                                randomized_correlated_gaussian_direct[:, k]
                            )
                            for k in range(nrands)
                        ]
                    )

                    covariance_pvals_naive = 1 - np.array(
                        [
                            [
                                p_value(
                                    covariance_sample_randomized_naive[:, n, m],
                                    empirical_covariance[n, m],
                                    two_tail=True,
                                )
                                for n in range(self.graph.N)
                            ]
                            for m in range(self.graph.N)
                        ]
                    )

                    covariance_pvals_undirect = 1 - np.array(
                        [
                            [
                                p_value(
                                    covariance_sample_randomized_undirect[:, n, m],
                                    empirical_covariance[n, m],
                                    two_tail=True,
                                )
                                for n in range(self.graph.N)
                            ]
                            for m in range(self.graph.N)
                        ]
                    )

                    covariance_pvals_direct = 1 - np.array(
                        [
                            [
                                p_value(
                                    covariance_sample_randomized_direct[:, n, m],
                                    covariance_matrix[n, m],
                                    two_tail=True,
                                )
                                for n in range(self.graph.N)
                            ]
                            for m in range(self.graph.N)
                        ]
                    )

                    # Bonferroni correction
                    if self.config["correction"] == "bonferroni":
                        naive_pvals = (
                            (1 - covariance_pvals_naive) <= self.config["alpha"] / mult
                        ).astype(int)
                        undirect_pvals = (
                            (1 - covariance_pvals_undirect)
                            <= self.config["alpha"] / mult
                        ).astype(int)
                        direct_pvals = (
                            (1 - covariance_pvals_direct) <= self.config["alpha"] / mult
                        ).astype(int)

                    # FDR correction
                    elif self.config["correction"] == "fdr":
                        start_naive_pvals, _, _, _ = multipletests(
                            1 - covariance_pvals_naive[np.triu_indices(self.graph.N)],
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )
                        naive_pvals = np.zeros((self.graph.N, self.graph.N))
                        naive_pvals[np.triu_indices(self.graph.N)] = start_naive_pvals
                        naive_pvals = (naive_pvals + naive_pvals.T) > 0

                        start_undirect_pvals, _, _, _ = multipletests(
                            1
                            - covariance_pvals_undirect[np.triu_indices(self.graph.N)],
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )
                        undirect_pvals = np.zeros((self.graph.N, self.graph.N))
                        undirect_pvals[
                            np.triu_indices(self.graph.N)
                        ] = start_undirect_pvals
                        undirect_pvals = (undirect_pvals + undirect_pvals.T) > 0

                        start_direct_pvals, _, _, _ = multipletests(
                            1 - covariance_pvals_direct[np.triu_indices(self.graph.N)],
                            alpha=self.config["alpha"],
                            method="fdr_bh",
                        )
                        direct_pvals = np.zeros((self.graph.N, self.graph.N))
                        direct_pvals[np.triu_indices(self.graph.N)] = start_direct_pvals
                        direct_pvals = (direct_pvals + direct_pvals.T) > 0

                    else:
                        raise ValueError("Unknown correction method")

                    accuracy_direct = (direct_pvals == offdiagonals).astype(int).mean()
                    accuracy_undirect = (
                        (undirect_pvals == offdiagonals).astype(int).mean()
                    )
                    accuracy_naive = (naive_pvals == offdiagonals).astype(int).mean()
                    scores_dir[prop].append(accuracy_direct)
                    scores_und[prop].append(accuracy_undirect)
                    scores_naive[prop].append(accuracy_naive)
                    samples[prop].append((offdiagonals > 0).sum())

            save(
                os.path.join(DATA_DIR, "covariance_scores.pkl"),
                (scores_dir, scores_und, scores_naive, samples),
            )

        # Plotting
        labels = []
        for prop in proportion:
            perc = np.round(np.array(samples[prop]).mean() / self.graph.N**2, 3) * 100
            labels.append(str(int(np.ceil(perc))) + "%")

        # Prepare data for plotting
        data = []
        for prop in proportion:
            for score in scores_dir[prop]:
                data.append(["Directed", prop, score])
            for score in scores_und[prop]:
                data.append(["Undirected", prop, score])
            for score in scores_naive[prop]:
                data.append(["Naive", prop, score])

        df_plot = pd.DataFrame(data, columns=["Type", "Number of Disruptions", "Score"])

        # Plotting
        fig, ax = plt.subplots(figsize=(5, 3))

        sns.boxplot(
            x="Number of Disruptions",
            y="Score",
            hue="Type",
            data=df_plot,
            showmeans=False,
            meanline=False,
            palette={"Directed": "red", "Undirected": "blue", "Naive": "black"},
            linewidth=0.5,
            boxprops=dict(alpha=1),
            whiskerprops=dict(linewidth=0.2),
            capprops=dict(linewidth=1),
            flierprops=dict(marker="o", markersize=1, alpha=0.5),
            order=sorted(proportion, reverse=True),  # show x-labels in decreasing order
        )

        # Labels and title
        ax.set_xlabel("", fontsize=self.fontsize, fontname="Helvetica", labelpad=10)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=self.fontsize, fontname="Helvetica")
        ax.set_ylabel(
            "Accuracy", fontsize=self.fontsize, fontname="Helvetica", labelpad=10
        )
        ax.set_yticklabels(
            ax.get_yticklabels(), fontsize=self.fontsize, fontname="Helvetica"
        )
        for t in [0.5 + i for i in range(len(proportion))]:
            ax.axvline(x=t, color="black", linestyle="--", linewidth=1, alpha=0.1)

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        # Customize legend
        ax.legend(
            loc="upper right", prop={"family": "Helvetica", "size": self.fontsize}
        )
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        # Adjust layout
        plt.tight_layout()
        if not self.verbose:
            plt.close()
        plt.show()

        return fig

    def run_experiment4(self):
        figs = []
        # Plot example of Diffusion timecourse
        n_iter = self.config["n_iter"]
        idx_of_interest = np.arange(0, n_iter)
        time_noise = np.arange(n_iter)

        time_idx = [self.config["time_idx"]]
        node_amp = [self.config["node_amp"]]

        directed_logs = self.surrogate.var_generator(
            A=self.graph.adj_matrix,
            active_nodes=[28],
            amplitude_nodes=node_amp,
            time_nodes=time_idx,
            n_iter=n_iter,
            time_noise=time_noise,
            add_noise="graph",
            gamma=1,
            seed=85,
        )

        fig, ax = plt.subplots(1, figsize=(5, 3))
        ax.plot(  # simple plotting for legend entry
            [],
            alpha=1,
            color="k",
            linestyle="--",
            linewidth=1,
            label=r"${\bf x}_t[n], n\neq n_0$",
        )
        for k in range(self.graph.N):
            ax.plot(
                directed_logs[idx_of_interest][:, k],
                alpha=0.2,
                color="k",
                linestyle="--",
                linewidth=1,
            )
            if k == 28:
                ax.plot(
                    directed_logs[idx_of_interest][:, k],
                    alpha=1,
                    color="r",
                    linestyle="solid",
                    label=r"${\bf x}_t[n], n=n_0$",
                    linewidth=2,
                )
                # ax.axvline(
                #     x=9,
                #     color="magenta",
                #     linestyle="--",
                #     linewidth=2,
                #     label=r"$t=t_0$",
                # )
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )

        # Set font properties
        font_properties = {"fontname": "Helvetica", "fontsize": self.fontsize}

        # Apply font properties to labels and title
        ax.set_xlabel("Time", **font_properties)

        # Apply font properties to legend
        ax.legend(prop={"family": "Helvetica", "size": self.fontsize})

        # Apply font properties to ticks
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.fontsize,
            labelcolor="black",
            labelrotation=0,
        )
        for tick in ax.get_xticklabels():
            tick.set_fontname("Helvetica")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Helvetica")

        figs.append(fig)
        if not self.verbose:
            plt.close()
        plt.show()

        # Plot ROC curves for different surrogate models
        rerun = False
        nrands = 99
        nodes_totry = np.arange(0, 48)
        if not os.path.exists(
            os.path.join(DATA_DIR, "surrogates_data/ROC_experiments/")
        ):
            rerun = True
        if rerun:
            os.makedirs(
                os.path.join(DATA_DIR, "surrogates_data/ROC_experiments/"),
                exist_ok=True,
            )

            def process_node(nidx):
                significant_node = [nidx]

                timecourse_samples = []
                timecourses_full_directed = []
                timecourses_full_undirected = []
                timecourses_full_naive = []
                for k in range(50):
                    directed_logs = self.surrogate.var_generator(
                        A=self.graph.adj_matrix,
                        active_nodes=significant_node,
                        amplitude_nodes=node_amp,
                        time_nodes=time_idx,
                        n_iter=n_iter,
                        time_noise=time_noise,
                        add_noise="graph",
                        gamma=1,
                        seed=k,
                    )

                    (
                        timecourse_null_directed,
                        timecourse_null_undirected,
                        timecourse_null_naive,
                    ) = timeseries_null_models_generator(
                        directed_logs,
                        idx_of_interest=None,
                        surg=self.surrogate,
                        surg_u=self.surrogate_u,
                        nrands=nrands,
                        verbose=False,
                    )

                    timecourses_full_directed.append(timecourse_null_directed)
                    timecourses_full_undirected.append(timecourse_null_undirected)
                    timecourses_full_naive.append(timecourse_null_naive)

                    timecourse_samples.append(directed_logs)

                timecourse_samples = np.array(timecourse_samples)
                full_directed = np.concatenate(
                    np.swapaxes(np.array(timecourses_full_directed), 1, 2)
                )
                full_undirected = np.concatenate(
                    np.swapaxes(np.array(timecourses_full_undirected), 1, 2)
                )
                full_naive = np.concatenate(
                    np.swapaxes(np.array(timecourses_full_naive), 1, 2)
                )

                pathname = os.path.join(
                    DATA_DIR,
                    f"surrogates_data/ROC_experiments/onenode{significant_node[0]}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
                )
                save(
                    pathname,
                    {
                        "initial_stats": timecourse_samples,
                        "full_directed": full_directed,
                        "full_undirected": full_undirected,
                        "full_naive": full_naive,
                    },
                )

            Parallel(n_jobs=-1)(
                delayed(process_node)(nidx) for nidx in tqdm(nodes_totry)
            )

        significant_node = [0]
        dct = load(
            os.path.join(
                DATA_DIR,
                f"surrogates_data/ROC_experiments/onenode{significant_node[0]}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
            )
        )

        full_directed = dct["full_directed"]
        full_undirected = dct["full_undirected"]
        full_naive = dct["full_naive"]

        nodes_interest = [0, -1]
        for k in range(2):
            fig, ax = plt.subplots(1, figsize=(5, 3))
            time_of_interest = 10

            lw = 2

            ax.axvline(
                x=dct["initial_stats"][:, time_of_interest, nodes_interest[k]].mean(),
                color="magenta",
                label="Empirical",
                linestyle="--",
                linewidth=lw,
            )
            sns.kdeplot(
                full_directed[:, time_of_interest, nodes_interest[k]],
                color="r",
                ax=ax,
                label="Directed",
                linewidth=lw,
            )
            sns.kdeplot(
                full_undirected[:, time_of_interest, nodes_interest[k]],
                color="b",
                ax=ax,
                label="Undirected",
                linestyle="-",
                linewidth=lw,
            )
            sns.kdeplot(
                full_naive[:, time_of_interest, nodes_interest[k]],
                color="k",
                ax=ax,
                label="Naive",
                linewidth=lw,
            )
            ax.legend(prop={"size": self.fontsize}, loc="upper left")
            ax.set_xlabel("Node Value")
            ax.grid(
                visible=True,
                which="major",
                axis="y",
                linestyle="--",
                linewidth=0.5,
                alpha=0.7,
            )

            # Increase size of all ticks and text
            ax.tick_params(axis="both", which="major", labelsize=self.fontsize)

            for tick in ax.get_xticklabels():
                tick.set_fontname("Helvetica")
            for tick in ax.get_yticklabels():
                tick.set_fontname("Helvetica")

            ax.set_xlabel(ax.get_xlabel(), fontsize=self.fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=self.fontsize)
            ax.set_title(ax.get_title(), fontsize=self.fontsize)
            ax.set_xlim(
                -17,
            )
            figs.append(fig)

            if not self.verbose:
                plt.close()
            plt.show()

        if os.path.exists(
            os.path.join(DATA_DIR, "surrogates_data/ROC_experiments/one_ROC_curves.pkl")
        ):
            S1, S2, S3, F1, F2, F3 = load(
                os.path.join(
                    DATA_DIR, "surrogates_data/ROC_experiments/one_ROC_curves.pkl"
                )
            )
        else:
            S1, S2, S3 = [], [], []
            F1, F2, F3 = [], [], []

            def compute_significancy(nidx):
                significant_node = [nidx]
                dct = load(
                    os.path.join(
                        DATA_DIR,
                        f"surrogates_data/ROC_experiments/onenode{significant_node[0]}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
                    )
                )

                timecourse_samples = dct["initial_stats"]
                full_directed = dct["full_directed"]
                full_undirected = dct["full_undirected"]
                full_naive = dct["full_naive"]

                t1 = np.zeros_like(timecourse_samples)
                t2 = np.zeros_like(timecourse_samples)
                t3 = np.zeros_like(timecourse_samples)

                for tidx in range(full_directed.shape[1]):
                    for ridx in range(full_directed.shape[2]):
                        for repeat in range(timecourse_samples.shape[0]):
                            t1[repeat, tidx, ridx] = p_value(
                                null_distrib=full_directed[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t2[repeat, tidx, ridx] = p_value(
                                null_distrib=full_undirected[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t3[repeat, tidx, ridx] = p_value(
                                null_distrib=full_naive[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                correction_alphas = np.linspace(1, 0, 10000)
                masksign = np.zeros_like(timecourse_samples[0])
                masksign[time_idx[0] + 1, significant_node[0]] = 1.0
                masknotsign = 1 - masksign

                sensitivity1 = np.array(
                    [
                        (t1[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity4 = np.array(
                    [
                        (t2[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity5 = np.array(
                    [
                        (t3[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                false_positive1 = np.array(
                    [
                        (t1[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive4 = np.array(
                    [
                        (t2[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive5 = np.array(
                    [
                        (t3[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                return (
                    sensitivity1,
                    sensitivity4,
                    sensitivity5,
                    false_positive1,
                    false_positive4,
                    false_positive5,
                )

            nodes_totry = np.arange(0, 48)
            results = Parallel(n_jobs=-1)(
                delayed(compute_significancy)(nidx) for nidx in tqdm(nodes_totry)
            )

            for res in results:
                S1.append(res[0])
                S2.append(res[1])
                S3.append(res[2])
                F1.append(res[3])
                F2.append(res[4])
                F3.append(res[5])

            save(
                os.path.join(
                    DATA_DIR, "surrogates_data/ROC_experiments/one_ROC_curves.pkl"
                ),
                (S1, S2, S3, F1, F2, F3),
            )

        dense = 50

        rX1, rY1, rVar1 = rasterize(
            np.array(F1).flatten(), np.array(S1).flatten(), dense=dense, eps=1e-5
        )
        rX2, rY2, rVar2 = rasterize(
            np.array(F2).flatten(), np.array(S2).flatten(), dense=dense, eps=1e-5
        )
        rX3, rY3, rVar3 = rasterize(
            np.array(F3).flatten(), np.array(S3).flatten(), dense=dense, eps=1e-5
        )

        rX1, rY1 = interpolate_nans(rX1), interpolate_nans(rY1)
        rX2, rY2 = interpolate_nans(rX2), interpolate_nans(rY2)
        rX3, rY3 = interpolate_nans(rX3), interpolate_nans(rY3)
        rY1 = gaussian_filter1d(rY1, sigma=1)
        rY2 = gaussian_filter1d(rY2, sigma=1)
        rY3 = gaussian_filter1d(rY3, sigma=1)

        nonnan_idx = 20
        r = [
            [rX1[nonnan_idx:], rY1[nonnan_idx:], rVar1[nonnan_idx:]],
            [rX2[nonnan_idx:], rY2[nonnan_idx:], rVar2[nonnan_idx:]],
            [rX3[nonnan_idx:], rY3[nonnan_idx:], rVar3[nonnan_idx:]],
        ]

        roc_auc_dir = auc(rX1[nonnan_idx:], rY1[nonnan_idx:])
        roc_auc_undir = auc(rX2[nonnan_idx:], rY2[nonnan_idx:])
        roc_auc_naive = auc(rX3[nonnan_idx:], rY3[nonnan_idx:])

        c = ["r", "b", "k"]
        labels = [
            "Directed",
            "Undirected",
            "Naive",
        ]

        markers = ["o", "s", "^"]
        fig, ax = plt.subplots(1, figsize=(5, 3))
        for k in range(len(c)):
            if k == 2:
                ax.plot(
                    r[k][0],
                    r[k][1],
                    color=c[k],
                    label=labels[k],
                    linestyle="--",
                    linewidth=1,
                    marker=markers[k],
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            else:
                ax.plot(
                    r[k][0],
                    r[k][1],
                    color=c[k],
                    label=labels[k],
                    linestyle="-",
                    linewidth=1,
                    marker=markers[k],
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            lower_quartile = r[k][1] - r[k][2] / 4
            upper_quartile = r[k][1] + r[k][2] / 4
            ax.errorbar(
                r[k][0],
                r[k][1],
                yerr=[r[k][1] - lower_quartile, upper_quartile - r[k][1]],
                color=c[k],
                alpha=0.5,
            )

        # Add AUC text labels next to each curve (placed near the rightmost plotted point)
        aucs_vals = [roc_auc_dir, roc_auc_undir, roc_auc_naive]
        x_texts = [0.007, 0.002, 0.032]
        y_pts = [0.88, 0.73, 0.63]
        for k in range(len(r)):
            x_text = x_texts[k]
            y_pt = y_pts[k]
            txt = f"AUC={aucs_vals[k]: .3f}"
            ax.text(
                x_text,
                y_pt,
                txt,
                fontsize=self.fontsize,
                family="Helvetica",
                color=c[k],
                verticalalignment="center",
                horizontalalignment="left",
                bbox=dict(
                    facecolor="white", edgecolor="black", alpha=1, pad=1, linewidth=0.5
                ),
                clip_on=False,
            )

        ax.legend(
            loc="lower right", prop={"size": self.fontsize, "family": "Helvetica"}
        )

        ax.set_xlabel("False positive", size=self.fontsize, family="Helvetica")
        ax.set_ylabel("Sensitivity", size=self.fontsize, family="Helvetica")
        # ax.set_xlim(0., 0.04)
        ax.tick_params(axis="both", which="major", labelsize=self.fontsize)
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        for tick in ax.get_xticklabels():
            tick.set_fontname("Helvetica")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Helvetica")

        ax.set_xscale("log")
        if not self.verbose:
            plt.close()
        plt.show()

        figs.append(fig)

        return figs

    def run_experiment5(self):
        figs = []
        n_iter = self.config["n_iter"]
        idx_of_interest = np.arange(0, n_iter)
        time_noise = np.arange(n_iter)

        time_idx = [self.config["time_idx"]]
        node_amp = [self.config["node_amp"]]
        mnodes_to_try = [5, 6, 13, 18, 19]

        directed_logs = self.surrogate.var_generator(
            A=self.graph.adj_matrix,
            active_nodes=mnodes_to_try,
            amplitude_nodes=node_amp * len(mnodes_to_try),
            time_nodes=time_idx,
            n_iter=n_iter,
            time_noise=time_noise,
            add_noise="graph",
            gamma=1,
            seed=85,
        )

        # Run for multiple singularities for detections
        fig, ax = plt.subplots(1, figsize=(5, 3))
        ax.plot(  # simple plotting for legend entry
            [],
            alpha=1,
            color="k",
            linestyle="--",
            linewidth=1,
            label=r"${\bf x}_t[n], n\notin\mathcal{N}$",
        )
        for k in range(self.graph.N):
            ax.plot(
                directed_logs[idx_of_interest][:, k],
                alpha=0.2,
                color="k",
                linestyle="--",
                linewidth=1,
            )
            if k in mnodes_to_try:
                if k == mnodes_to_try[0]:
                    ax.plot(
                        directed_logs[idx_of_interest][:, k],
                        alpha=1,
                        color="r",
                        linestyle="solid",
                        label=r"${\bf x}_t[n], n\in\mathcal{N}$",
                        linewidth=2,
                    )
                else:
                    ax.plot(
                        directed_logs[idx_of_interest][:, k],
                        alpha=1,
                        color="r",
                        linestyle="solid",
                        linewidth=2,
                    )

        # Set font properties
        font_properties = {"fontname": "Helvetica", "fontsize": self.fontsize}

        # Apply font properties to labels and title
        ax.set_xlabel("Time", **font_properties)

        # Apply font properties to legend
        ax.legend(prop={"family": "Helvetica", "size": self.fontsize})

        # Apply font properties to ticks
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=self.fontsize,
            labelcolor="black",
            labelrotation=0,
        )
        for tick in ax.get_xticklabels():
            tick.set_fontname("Helvetica")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Helvetica")
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        figs.append(fig)
        if not self.verbose:
            plt.close()
        plt.show()

        # Generate random sets of fixed given length for disruptors model
        original_set = np.arange(0, self.graph.N)
        set_length = 5
        nb_sets = 40

        if os.path.exists(
            os.path.join(
                DATA_DIR,
                f"surrogates_data/ROC_experiments/multi_nodes_set_{set_length}_nodes.pkl",
            )
        ):
            mnodes_to_try = load(
                os.path.join(
                    DATA_DIR,
                    f"surrogates_data/ROC_experiments/multi_nodes_set_{set_length}_nodes.pkl",
                )
            )
        else:
            np.random.seed(99)
            mnodes_to_try = [
                np.random.choice(original_set, set_length, replace=False)
                for _ in range(nb_sets)
            ]
            save(
                os.path.join(
                    DATA_DIR,
                    f"surrogates_data/ROC_experiments/multi_nodes_set_{set_length}_nodes.pkl",
                ),
                mnodes_to_try,
            )

        nrands = 99  # previously 99

        def process_multi_node(nidx, mnodes):
            significant_node = mnodes[nidx]

            timecourse_samples = []
            timecourses_full_directed = []
            timecourses_full_undirected = []
            timecourses_full_naive = []
            for k in range(50):
                directed_logs = self.surrogate.var_generator(
                    A=self.graph.adj_matrix,
                    active_nodes=significant_node,
                    amplitude_nodes=node_amp * len(significant_node),
                    time_nodes=time_idx,
                    n_iter=n_iter,
                    time_noise=time_noise,
                    add_noise="graph",
                    gamma=1,
                    seed=k,
                )

                (
                    timecourse_null_directed,
                    timecourse_null_undirected,
                    timecourse_null_naive,
                ) = timeseries_null_models_generator(
                    directed_logs,
                    idx_of_interest=None,
                    surg=self.surrogate,
                    surg_u=self.surrogate_u,
                    nrands=nrands,
                    verbose=False,
                )

                timecourses_full_directed.append(timecourse_null_directed)
                timecourses_full_undirected.append(timecourse_null_undirected)
                timecourses_full_naive.append(timecourse_null_naive)

                timecourse_samples.append(directed_logs)

            timecourse_samples = np.array(timecourse_samples)
            full_directed = np.concatenate(
                np.swapaxes(np.array(timecourses_full_directed), 1, 2)
            )
            full_undirected = np.concatenate(
                np.swapaxes(np.array(timecourses_full_undirected), 1, 2)
            )
            full_naive = np.concatenate(
                np.swapaxes(np.array(timecourses_full_naive), 1, 2)
            )

            pathname = os.path.join(
                DATA_DIR,
                f"surrogates_data/ROC_experiments/multnode_sequence{set_length}_{nidx}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
            )
            save(
                pathname,
                {
                    "initial_stats": timecourse_samples,
                    "full_directed": full_directed,
                    "full_undirected": full_undirected,
                    "full_naive": full_naive,
                },
            )

        rerun = False
        if not os.path.exists(
            os.path.join(
                DATA_DIR,
                f"surrogates_data/ROC_experiments/multnode_sequence{set_length}_0_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
            )
        ):
            rerun = True
        if rerun:
            Parallel(n_jobs=-1)(
                delayed(process_multi_node)(nidx, mnodes_to_try)
                for nidx in tqdm(range(len(mnodes_to_try)))
            )

        if os.path.exists(
            os.path.join(
                DATA_DIR, "surrogates_data/ROC_experiments/multi_ROC_curves.pkl"
            )
        ):
            S1, S2, S3, F1, F2, F3 = load(
                os.path.join(
                    DATA_DIR,
                    "surrogates_data/ROC_experiments/multi_ROC_curves.pkl",
                )
            )
        else:
            S1, S2, S3 = [], [], []
            F1, F2, F3 = [], [], []

            def compute_significancy(nidx, mnodes):
                significant_node = mnodes[nidx]
                dct = load(
                    os.path.join(
                        DATA_DIR,
                        f"surrogates_data/ROC_experiments/multnode_sequence{set_length}_{nidx}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
                    )
                )

                timecourse_samples = dct["initial_stats"]
                full_directed = dct["full_directed"]
                full_undirected = dct["full_undirected"]
                full_naive = dct["full_naive"]

                t1 = np.zeros_like(timecourse_samples)
                t2 = np.zeros_like(timecourse_samples)
                t3 = np.zeros_like(timecourse_samples)

                for tidx in range(full_directed.shape[1]):
                    for ridx in range(full_directed.shape[2]):
                        for repeat in range(timecourse_samples.shape[0]):
                            t1[repeat, tidx, ridx] = p_value(
                                null_distrib=full_directed[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t2[repeat, tidx, ridx] = p_value(
                                null_distrib=full_undirected[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t3[repeat, tidx, ridx] = p_value(
                                null_distrib=full_naive[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                correction_alphas = np.linspace(1, 0, 10000)
                masksign = np.zeros_like(timecourse_samples[0])
                masksign[time_idx[0] + 1, significant_node[0]] = 1.0
                masknotsign = 1 - masksign

                sensitivity1 = np.array(
                    [
                        (t1[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity2 = np.array(
                    [
                        (t2[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity3 = np.array(
                    [
                        (t3[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                false_positive1 = np.array(
                    [
                        (t1[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive2 = np.array(
                    [
                        (t2[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive3 = np.array(
                    [
                        (t3[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                return (
                    sensitivity1,
                    sensitivity2,
                    sensitivity3,
                    false_positive1,
                    false_positive2,
                    false_positive3,
                )

            results = Parallel(n_jobs=-1)(
                delayed(compute_significancy)(nidx, mnodes_to_try)
                for nidx in tqdm(range(len(mnodes_to_try)))
            )

            for res in results:
                S1.append(res[0])
                S2.append(res[1])
                S3.append(res[2])
                F1.append(res[3])
                F2.append(res[4])
                F3.append(res[5])
            save(
                os.path.join(
                    DATA_DIR, "surrogates_data/ROC_experiments/multi_ROC_curves.pkl"
                ),
                (S1, S2, S3, F1, F2, F3),
            )

        dense = 50

        rX1, rY1, rVar1 = rasterize(
            np.array(F1).flatten(), np.array(S1).flatten(), dense=dense, eps=1e-5
        )
        rX2, rY2, rVar2 = rasterize(
            np.array(F2).flatten(), np.array(S2).flatten(), dense=dense, eps=1e-5
        )
        rX3, rY3, rVar3 = rasterize(
            np.array(F3).flatten(), np.array(S3).flatten(), dense=dense, eps=1e-5
        )

        rX1, rY1 = interpolate_nans(rX1), interpolate_nans(rY1)
        rX2, rY2 = interpolate_nans(rX2), interpolate_nans(rY2)
        rX3, rY3 = interpolate_nans(rX3), interpolate_nans(rY3)
        rY1 = gaussian_filter1d(rY1, sigma=1)
        rY2 = gaussian_filter1d(rY2, sigma=1)
        rY3 = gaussian_filter1d(rY3, sigma=1)

        nonnan_idx = 25
        r = [
            [rX1[nonnan_idx:], rY1[nonnan_idx:], rVar1[nonnan_idx:]],
            [rX2[nonnan_idx:], rY2[nonnan_idx:], rVar2[nonnan_idx:]],
            [rX3[nonnan_idx:], rY3[nonnan_idx:], rVar3[nonnan_idx:]],
        ]

        roc_auc_dir = auc(rX1[nonnan_idx:], rY1[nonnan_idx:])
        roc_auc_undir = auc(rX2[nonnan_idx:], rY2[nonnan_idx:])
        roc_auc_naive = auc(rX3[nonnan_idx:], rY3[nonnan_idx:])

        c = ["r", "b", "k"]
        labels = [
            "Directed",
            "Undirected",
            "Naive",
        ]

        markers = ["o", "s", "^"]
        fig, ax = plt.subplots(1, figsize=(5, 3))
        for k in range(len(c)):
            if k == 2:
                ax.plot(
                    r[k][0],
                    r[k][1],
                    color=c[k],
                    label=labels[k],
                    linestyle="--",
                    linewidth=1,
                    marker=markers[k],
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            else:
                ax.plot(
                    r[k][0],
                    r[k][1],
                    color=c[k],
                    label=labels[k],
                    linestyle="-",
                    linewidth=1,
                    marker=markers[k],
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
            lower_quartile = r[k][1] - r[k][2] / 4
            upper_quartile = r[k][1] + r[k][2] / 4
            ax.errorbar(
                r[k][0],
                r[k][1],
                yerr=[r[k][1] - lower_quartile, upper_quartile - r[k][1]],
                color=c[k],
                alpha=0.5,
            )

        # Add AUC text labels next to each curve (placed near the rightmost plotted point)
        aucs_vals = [roc_auc_dir, roc_auc_undir, roc_auc_naive]
        x_texts = [0.0045, 0.025, 0.15]
        y_pts = [0.91, 0.48, 0.4]
        for k in range(len(r)):
            x_text = x_texts[k]
            y_pt = y_pts[k]
            txt = f"AUC={aucs_vals[k]: .3f}"
            ax.text(
                x_text,
                y_pt,
                txt,
                fontsize=self.fontsize,
                family="Helvetica",
                color=c[k],
                verticalalignment="center",
                horizontalalignment="left",
                bbox=dict(
                    facecolor="white", edgecolor="black", alpha=1, pad=1, linewidth=0.5
                ),
                clip_on=False,
            )

        ax.legend(
            loc="lower right", prop={"size": self.fontsize, "family": "Helvetica"}
        )

        ax.set_xlabel("False positive", size=self.fontsize, family="Helvetica")
        ax.set_ylabel("Sensitivity", size=self.fontsize, family="Helvetica")
        # ax.set_xlim(0.001, 1)
        ax.tick_params(axis="both", which="major", labelsize=self.fontsize)
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        for tick in ax.get_xticklabels():
            tick.set_fontname("Helvetica")
        for tick in ax.get_yticklabels():
            tick.set_fontname("Helvetica")

        ax.set_xscale("log")

        figs.append(fig)
        if not self.verbose:
            plt.close()
        plt.show()

        return figs

    def run_experiment6(self):
        time_idx = [self.config["time_idx"]]
        node_amp = [self.config["node_amp"]]
        tested_lengths = [3, 5, 10, 20, 25]

        # Plot AUC as a function of number of nodes
        aucs = {"Directed": [], "Undirected": [], "Naive": []}
        for set_length in tested_lengths:
            mnodes_to_try = load(
                os.path.join(
                    DATA_DIR,
                    f"surrogates_data/ROC_experiments/multi_nodes_set_{set_length}_nodes.pkl",
                )
            )

            S1, S2, S3 = [], [], []
            F1, F2, F3 = [], [], []

            def compute_significancy(nidx, mnodes):
                significant_node = mnodes[nidx]
                dct = load(
                    os.path.join(
                        DATA_DIR,
                        f"surrogates_data/ROC_experiments/multnode_sequence{set_length}_{nidx}_amp{node_amp[0]}_tidx{time_idx[0]}_directed_surrogates.pkl",
                    )
                )

                timecourse_samples = dct["initial_stats"]
                full_directed = dct["full_directed"]
                full_undirected = dct["full_undirected"]
                full_naive = dct["full_naive"]

                t1 = np.zeros_like(timecourse_samples)
                t2 = np.zeros_like(timecourse_samples)
                t3 = np.zeros_like(timecourse_samples)

                for tidx in range(full_directed.shape[1]):
                    for ridx in range(full_directed.shape[2]):
                        for repeat in range(timecourse_samples.shape[0]):
                            t1[repeat, tidx, ridx] = p_value(
                                null_distrib=full_directed[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t2[repeat, tidx, ridx] = p_value(
                                null_distrib=full_undirected[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                            t3[repeat, tidx, ridx] = p_value(
                                null_distrib=full_naive[:, tidx, ridx],
                                statistic=timecourse_samples[repeat, tidx, ridx],
                                two_tail=False,
                            )

                correction_alphas = np.linspace(1, 0, 10000)
                masksign = np.zeros_like(timecourse_samples[0])
                masksign[time_idx[0] + 1, significant_node[0]] = 1.0
                masknotsign = 1 - masksign

                sensitivity1 = np.array(
                    [
                        (t1[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity2 = np.array(
                    [
                        (t2[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                sensitivity3 = np.array(
                    [
                        (t3[:, masksign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                false_positive1 = np.array(
                    [
                        (t1[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive2 = np.array(
                    [
                        (t2[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )
                false_positive3 = np.array(
                    [
                        (t3[:, masknotsign.astype(bool)].flatten() < alpha).mean()
                        for alpha in correction_alphas
                    ]
                )

                return (
                    sensitivity1,
                    sensitivity2,
                    sensitivity3,
                    false_positive1,
                    false_positive2,
                    false_positive3,
                )

            results = Parallel(n_jobs=-1)(
                delayed(compute_significancy)(nidx, mnodes_to_try)
                for nidx in tqdm(range(len(mnodes_to_try)))
            )

            for res in results:
                S1.append(res[0])
                S2.append(res[1])
                S3.append(res[2])
                F1.append(res[3])
                F2.append(res[4])
                F3.append(res[5])

            dense = 50

            rX1, rY1, _ = rasterize(
                np.array(F1).flatten(), np.array(S1).flatten(), dense=dense, eps=1e-5
            )
            rX2, rY2, _ = rasterize(
                np.array(F2).flatten(), np.array(S2).flatten(), dense=dense, eps=1e-5
            )
            rX3, rY3, _ = rasterize(
                np.array(F3).flatten(), np.array(S3).flatten(), dense=dense, eps=1e-5
            )

            rX1, rY1 = interpolate_nans(rX1), interpolate_nans(rY1)
            rX2, rY2 = interpolate_nans(rX2), interpolate_nans(rY2)
            rX3, rY3 = interpolate_nans(rX3), interpolate_nans(rY3)
            rY1 = gaussian_filter1d(rY1, sigma=1)
            rY2 = gaussian_filter1d(rY2, sigma=1)
            rY3 = gaussian_filter1d(rY3, sigma=1)

            nonnan_idx = 25

            roc_auc_dir = auc(rX1[nonnan_idx:], rY1[nonnan_idx:])
            roc_auc_undir = auc(rX2[nonnan_idx:], rY2[nonnan_idx:])
            roc_auc_naive = auc(rX3[nonnan_idx:], rY3[nonnan_idx:])

            aucs["Directed"].append(roc_auc_dir)
            aucs["Undirected"].append(roc_auc_undir)
            aucs["Naive"].append(roc_auc_naive)

        # Prepare data for plotting
        node_counts = tested_lengths
        node_counts = np.round(np.array(node_counts) / self.graph.N, 1)
        node_counts[0] = 0.05  # Rounding
        node_counts = [
            str(int(node_counts[k] * 100)) + "%" for k in range(len(node_counts))
        ]
        fig, ax = plt.subplots(figsize=(5, 3))

        ax.plot(
            aucs["Directed"],
            marker="o",
            color="r",
            label="Directed",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        ax.plot(
            aucs["Undirected"],
            marker="s",
            color="b",
            label="Undirected",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        ax.plot(
            aucs["Naive"],
            marker="^",
            color="k",
            label="Naive",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        ax.set_xlabel(
            "Significant Nodes Density", fontsize=self.fontsize, fontname="Helvetica"
        )
        ax.set_ylabel("AUC", fontsize=self.fontsize, fontname="Helvetica")
        ax.tick_params(axis="y", labelsize=self.fontsize)
        for t in ax.get_yticklabels():
            t.set_fontname("Helvetica")
        ax.set_xticks(range(len(node_counts)))
        ax.set_xticklabels(node_counts, fontsize=self.fontsize, fontname="Helvetica")
        ax.legend(fontsize=self.fontsize, loc="lower left")
        ax.grid(
            visible=True,
            which="major",
            axis="y",
            linestyle="--",
            linewidth=0.5,
            alpha=0.7,
        )
        plt.tight_layout()

        if not self.verbose:
            plt.close()
        plt.show()

        return fig


if __name__ == "__main__":
    run()
