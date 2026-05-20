"""
Experiment: Graph Temperature Data Analysis

This experiment analyzes temperature data from various stations in Brittany, France, using graph signal processing techniques. The main steps include:
1. Graph Creation: Constructing a graph based on the geographical locations of the temperature stations and their pairwise distances.
2. Irregular Days Detection: Using surrogate data to identify days with significant deviations in temperature patterns.
3. Statistics Over Time of the Year: Analyzing how the temperature signal's properties evolve throughout the year, particularly for coastal vs. inland stations.
"""

import os
import json
from datetime import datetime, timedelta

# Import necessary libraries
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.patches as patches
from statsmodels.stats.multitest import multipletests

# Import FlowGSP components
from flowgsp.graphs import Graph
from flowgsp.surrogates import Surrogate
from flowgsp.utils import load, p_value, save
from flowgsp.operators import destroy_jordan_blocks, destroy_zero_eigenvals

from experiments.paper_surrogates_stats.paper_utils import extract_values


# Get the directory of this script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")


def run(
    save_results: bool = True, verbose: bool = True, recompute: bool = False
) -> dict:
    """
    Run the temperature data analysis experiment.

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
        experiment_name="temperature_data",
        verbose=verbose,
        log_file=None
        if not save_results
        else os.path.join(RESULTS_DIR, "temperature_data.log"),
        results_dir=RESULTS_DIR if save_results else None,
    )

    # Suppress noisy libraries
    if not verbose:
        set_library_log_levels("ERROR")

    logger.info("=" * 60)
    logger.info("Experiment: Graph Temperature Data Analysis")
    logger.info("=" * 60)

    # Configuration
    config = load_json(os.path.join(EXPERIMENT_DIR, "config", "config.json"))

    logger.info(f"\nConfiguration: {config}")

    experiments = Experiments(
        config, verbose=verbose, logger=logger, recompute=recompute
    )
    # Plot 1: Graph Creation
    logger.info("\nRunning Graph Creation...")
    fig1 = experiments.run_experiment1()

    # Plot 2: Irregular Days detection
    logger.info("\nRunning Irregular Days Detection...")
    (
        (fig2, fig3),
        demeaned_sig,
        null_distrib_directed,
        null_distrib_undirected,
    ) = experiments.run_experiment2()

    # Plot 3: Statistics over time of the year
    logger.info("\nRunning Statistics Over Time of the Year...")
    fig4, fig5 = experiments.run_experiment3(
        demeaned_sig, null_distrib_directed, null_distrib_undirected
    )

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    if save_results:
        from experiments.logging_utils import save_figures

        save_figures(
            figures=[fig1, fig2, fig3, fig4, fig5],
            filenames=[
                "signal_covariances.png",
                "irregular_days_detection_directed.png",
                "irregular_days_detection_undirected.png",
                "timecourse_statistics_coastal.png",
                "timecourse_statistics_inland.png",
            ],
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
        self.path_to_resources = "./data/temperature_bretagne_graph_data/"
        self.fontsize = 10
        self._create_graph()

    def _create_graph(self):
        if (
            os.path.exists(
                os.path.join(self.path_to_resources, "temperature_graph_all.pkl")
            )
            and not self.recompute
        ):
            self.logger.info("Loading precomputed temperature graph data...")
            # data = load(
            #     os.path.join(self.path_to_resources, "temperature_graph_all.pkl")
            # )
            data = load(os.path.join(self.path_to_resources, "figure_data/all.pkl"))
            self.graph_sig = data["graph_sig"]
            self.graph_adj = data["graph_adj"]
            self.uA = data["uA"]
            self.graph_nodes_id = data["nodes_id"]
            self.stations = data["stations"]
            self.stations_pos = data["stations_pos"]
            self.pos_dir = data["pos_dir"]
            self.pos = data["pos"]

            self.graph = Graph(adj_matrix=self.graph_adj, pos=self.stations_pos)
            self.graph.set_operator("adjacency")
            self.surrogate = Surrogate(self.graph)

            self.graph_u = Graph(adj_matrix=self.uA, pos=self.stations_pos)
            self.graph_u.set_operator("adjacency")
            self.surrogate_u = Surrogate(self.graph_u)

            fig, ax = plt.subplots(figsize=(4, 3))
            pos = {
                station: (self.stations_pos[idx][1], self.stations_pos[idx][0])
                for idx, station in enumerate(self.stations)
            }
            G_plot = nx.from_numpy_array(self.graph_adj)
            mapping = {
                i: self.graph_nodes_id[i] for i in range(len(self.graph_nodes_id))
            }
            G_plot = nx.relabel_nodes(G_plot, mapping)
            nx.draw(
                G_plot,
                pos,
                with_labels=True,
                node_size=50,
                font_size=3,
                ax=ax,
                edgecolors="black",
            )
            ax.set_title("Temperature Stations Graph")
            if not self.verbose:
                plt.close()
            plt.show()
            return fig
        else:
            self.logger.info(
                "Precomputed data not found. Processing raw temperature data..."
            )
            import netCDF4

            # Extracting temperature data from netCDF files
            dfs = []
            # Loop through all files in the directory
            for filename in os.listdir(
                os.path.join(self.path_to_resources, "ipsl-temperature/2024")
            ):
                if filename.endswith(".nc"):
                    # Read the netCDF file
                    file_path = os.path.join(
                        self.path_to_resources, "ipsl-temperature/2024", filename
                    )
                    file2read = netCDF4.Dataset(file_path, "r", format="NETCDF4")

                    # Extract variables
                    time = file2read.variables["time"][:]
                    temperature = file2read.variables["ta"][:]
                    latitude = file2read.variables["lat"][:]
                    longitude = file2read.variables["lon"][:]
                    altitude = file2read.variables["alt"][:]

                    # Convert time to datetime format
                    time_units = file2read.variables["time"].units
                    time_calendar = file2read.variables["time"].calendar
                    dates = netCDF4.num2date(
                        time, units=time_units, calendar=time_calendar
                    )

                    # Create a DataFrame
                    df_temp = pd.DataFrame(
                        {
                            "Date": dates,
                            "Temperature": temperature,
                            "Latitude": latitude,
                            "Longitude": longitude,
                            "Altitude": altitude,
                            "Station Name": filename.split("_")[
                                1
                            ],  # Assuming station name is part of the filename
                        }
                    )

                    # Append the dataframe to the list
                    dfs.append(df_temp)

            # Concatenate all dataframes into a single dataframe
            df_temperature_all = pd.concat(dfs, ignore_index=True)
            working_stations = df_temperature_all.dropna()["Station Name"].unique()
            df_temperature_all = df_temperature_all[
                df_temperature_all["Station Name"].isin(working_stations)
            ]

            # Convert cftime objects to datetime objects
            df_temperature_all["Date"] = df_temperature_all["Date"].apply(
                lambda x: x if isinstance(x, pd.Timestamp) else pd.Timestamp(str(x))
            )
            df_temperature_all.index = list(df_temperature_all["Date"])

            df_temperature_all.rename(columns={"Date": "time"}, inplace=True)

            # Create Graph from station locations
            from sklearn.neighbors import NearestNeighbors

            stations = df_temperature_all["Station Name"].unique()
            stations_pos = np.array(
                [
                    np.array(
                        df_temperature_all[
                            df_temperature_all["Station Name"] == station
                        ][["Latitude", "Longitude", "Altitude"]].iloc[0]
                    )
                    for station in stations
                ]
            )

            # Create a weighted graph
            G_weighted = nx.Graph()
            G_directed = nx.DiGraph()

            # Add nodes with positions
            for idx, station in enumerate(stations):
                G_weighted.add_node(
                    station, pos=(stations_pos[idx][1], stations_pos[idx][0])
                )
                G_directed.add_node(
                    station, pos=(stations_pos[idx][1], stations_pos[idx][0])
                )

            # Add edges with weights based on exponential decayed distance
            lbd = 1.2
            scaling = 1
            dist_thresh = 0.5
            # Fit NearestNeighbors model
            nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
                stations_pos[:, :2]
            )
            distances, indices = nbrs.kneighbors(stations_pos[:, :2])
            max_neighbors = 5
            for i in range(len(stations)):
                indirect_count = 0
                direct_count = 0
                for j in range(
                    1, 20
                ):  # Skip the first neighbor since it's the node itself
                    lat1, _ = stations_pos[i][:2]
                    lat2, _ = stations_pos[indices[i][j]][:2]
                    distance = distances[i][j]
                    weight = scaling * np.exp(-lbd * distance)
                    if weight >= dist_thresh:
                        if indirect_count < max_neighbors:
                            G_weighted.add_edge(
                                stations[i], stations[indices[i][j]], weight=weight
                            )
                            indirect_count += 1
                        if (lat1 > lat2) and (direct_count < max_neighbors):
                            G_directed.add_edge(
                                stations[i], stations[indices[i][j]], weight=weight
                            )
                            direct_count += 1

            pos = nx.get_node_attributes(G_weighted, "pos")
            pos_dir = nx.get_node_attributes(G_directed, "pos")

            np.random.seed(98)
            graph_nodes_id = list(np.array(list(G_directed.nodes))[2:])

            g = G_directed.subgraph(graph_nodes_id)

            good_adj = []
            good_graph = []
            lbd = 0.5

            prev_A = (nx.adjacency_matrix(g).todense()).astype(float)
            iter_A = destroy_jordan_blocks(prev_A, prefer_nodes=[])
            iter_A = destroy_zero_eigenvals(iter_A, eps=1e-3)
            add_edges = iter_A - prev_A
            iter_A = prev_A + add_edges * lbd
            U = np.linalg.eig(iter_A)[1]
            self.logger.info(f"condition number {np.linalg.cond(U)}")
            if np.linalg.cond(U) < 1e2:
                self.logger.info("Good graph found - Parameters")
                self.logger.info(f"number of edges added: {np.abs(add_edges).sum()}")
                self.logger.info(
                    f"Number of nodes in connected_subgraph: {len(g.nodes)}"
                )
                self.logger.info(
                    f"Number of edges in connected_subgraph: {len(g.edges)}"
                )
                self.logger.info("")
                good_adj.append((prev_A, iter_A))
                good_graph.append(g)

            start_timestamp = "2024-01-01 00:00:00"
            end_timestamp = "2024-12-31 00:00:00"

            # Drop the days with missing data
            missing_days = [
                pd.Timestamp("2024-12-09 00:00:00"),
                pd.Timestamp("2024-12-10 00:00:00"),
                pd.Timestamp("2024-12-13 00:00:00"),
                pd.Timestamp("2024-12-14 00:00:00"),
            ]
            for minday in missing_days:
                if minday in df_temperature_all.index:
                    df_temperature_all = df_temperature_all.drop(minday)

            # Initialize an empty list to store signals for each day
            all_signals = []

            date_range = pd.date_range(
                start=start_timestamp, end=end_timestamp, freq="D"
            )[:-1]
            filtered_date_range = date_range[date_range.isin(df_temperature_all.index)]

            def compute_signals(day_start_timestamp, day_end_timestamp):
                return np.array(
                    [
                        np.nanmean(
                            np.array(
                                extract_values(
                                    df_temperature_all,
                                    "Temperature",
                                    gidx,
                                    "Station Name",
                                    day_start_timestamp,
                                    day_end_timestamp,
                                )
                            )
                        )
                        for gidx in graph_nodes_id
                    ]
                )

            all_signals = np.array(
                Parallel(n_jobs=-1)(
                    delayed(compute_signals)(day_start_timestamp, day_end_timestamp)
                    for day_start_timestamp, day_end_timestamp in [
                        (
                            current_date.strftime("%Y-%m-%d %H:%M:%S"),
                            (current_date + timedelta(days=1)).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        )
                        for current_date in filtered_date_range
                        if ((current_date + timedelta(days=1)) in filtered_date_range)
                        and (current_date in filtered_date_range)
                    ]
                )
            )

            all_signals = pd.DataFrame(all_signals).interpolate(axis=1).values

            graph_sig = all_signals - np.mean(all_signals, axis=0)

            graph_adj = good_adj[0][1]
            uA = ((graph_adj.T + graph_adj) / 2).astype(float)

            tosave = {
                "graph_sig": graph_sig,
                "graph_adj": graph_adj,
                "uA": uA,
                "nodes_id": graph_nodes_id,
                "stations": stations,
                "pos": pos,
                "stations_pos": stations_pos,
                "pos_dir": pos_dir,
            }
            save(
                os.path.join(self.path_to_resources, "temperature_graph_all.pkl"),
                tosave,
            )

        fig, ax = plt.subplots(figsize=(4, 3))
        pos = {
            station: (stations_pos[idx][1], stations_pos[idx][0])
            for idx, station in enumerate(stations)
        }
        G_plot = nx.from_numpy_array(graph_adj)
        mapping = {i: graph_nodes_id[i] for i in range(len(graph_nodes_id))}
        G_plot = nx.relabel_nodes(G_plot, mapping)
        nx.draw(
            G_plot,
            pos,
            with_labels=True,
            node_size=50,
            font_size=3,
            ax=ax,
            edgecolors="black",
        )
        ax.set_title("Temperature Stations Graph")
        if not self.verbose:
            plt.close()
        plt.show()

        data = load(os.path.join(self.path_to_resources, "temperature_graph_all.pkl"))
        self.graph_sig = data["graph_sig"]
        graph_adj = data["graph_adj"]
        uA = data["uA"]
        self.graph_nodes_id = data["nodes_id"]
        self.stations = data["stations"]
        self.stations_pos = data["stations_pos"]
        self.pos_dir = data["pos_dir"]
        self.pos = data["pos"]

        self.graph = Graph(adj_matrix=graph_adj, pos=self.stations_pos)
        self.graph.set_operator("adjacency")
        self.surrogate = Surrogate(self.graph)

        self.graph_u = Graph(adj_matrix=uA, pos=self.stations_pos)
        self.graph_u.set_operator("adjacency")
        self.surrogate_u = Surrogate(self.graph_u)

        return fig

    def run_experiment1(self):
        from matplotlib.colors import LogNorm

        demeaned_sig = np.array(
            [
                self.graph_sig[sidx] - self.graph_sig[sidx].mean()
                for sidx in range(self.graph_sig.shape[0])
            ]
        )

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        directed_autocorr = np.abs(
            self.surrogate.estimate_psd(
                self.surrogate.estimate_covariance(demeaned_sig)
            )
        )
        undirected_autocorr = np.abs(
            self.surrogate_u.estimate_psd(
                self.surrogate_u.estimate_covariance(demeaned_sig)
            )
        )

        # Ensure positive values for LogNorm
        directed_autocorr[directed_autocorr == 0] = np.min(
            directed_autocorr[directed_autocorr > 0]
        )
        undirected_autocorr[undirected_autocorr == 0] = np.min(
            undirected_autocorr[undirected_autocorr > 0]
        )

        ax[0].imshow(directed_autocorr, norm=LogNorm())
        ax[1].imshow(undirected_autocorr, norm=LogNorm())

        ax[0].set_title("Directed", fontsize=self.fontsize, fontname="Helvetica")
        ax[1].set_title("Undirected", fontsize=self.fontsize, fontname="Helvetica")
        for axis in ax:
            axis.tick_params(axis="both", which="major", labelsize=self.fontsize)
            axis.tick_params(axis="both", which="minor", labelsize=self.fontsize)

        cax = fig.add_axes([0.92, 0.20, 0.02, 0.59])
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=LogNorm(
                vmin=min(directed_autocorr.min(), undirected_autocorr.min()),
                vmax=max(directed_autocorr.max(), undirected_autocorr.max()),
            ),
        )
        sm.set_array([])
        fig.colorbar(sm, cax=cax)

        cax.tick_params(labelsize=self.fontsize)
        for label in cax.get_yticklabels():
            label.set_fontsize(self.fontsize)
            label.set_fontname("Helvetica")

        cax.yaxis.set_label_position("right")
        if not self.verbose:
            plt.close()
        plt.show()

        self.logger.info("Stationary levels (Directed, Undirected):")
        self.logger.info(
            (
                self.surrogate.stationary_level(demeaned_sig),
                self.surrogate_u.stationary_level(demeaned_sig),
            )
        )

        return fig

    def run_experiment2(self):
        # Surrogate Generation
        nrands = self.config["nb_rands"]

        demeaned_sig = np.array(
            [
                self.graph_sig[sidx] - self.graph_sig[sidx].mean()
                for sidx in range(self.graph_sig.shape[0])
            ]
        )

        if (
            os.path.exists(
                os.path.join(
                    DATA_DIR, f"null_distributions_tempdata_{nrands}_rands.pkl"
                )
            )
            and not self.recompute
        ):
            null_distrib_directed, null_distrib_undirected = load(
                os.path.join(
                    DATA_DIR, f"null_distributions_tempdata_{nrands}_rands.pkl"
                )
            )
        else:
            null_distrib_directed = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.surrogate.directed_random_surrogate)(
                        demeaned_sig[sidx], nrands=nrands
                    )
                    for sidx in tqdm(
                        range(self.graph_sig.shape[0]),
                        desc="Generating Directed Surrogates",
                        disable=not self.verbose,
                    )
                )
            )
            null_distrib_undirected = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self.surrogate_u.undirected_random_surrogate)(
                        demeaned_sig[sidx], nrands=nrands
                    )
                    for sidx in tqdm(
                        range(self.graph_sig.shape[0]),
                        desc="Generating Undirected Surrogates",
                        disable=not self.verbose,
                    )
                )
            )
            save(
                os.path.join(
                    DATA_DIR, f"null_distributions_tempdata_{nrands}_rands.pkl"
                ),
                (null_distrib_directed, null_distrib_undirected),
            )

        pvalues_undirect = np.array(
            [
                [
                    p_value(
                        null_distrib=null_distrib_undirected[t, :, n],
                        statistic=demeaned_sig[t, n],
                        two_tail=False,
                    )
                    for t in range(self.graph_sig.shape[0])
                ]
                for n in range(self.graph_sig.shape[1])
            ]
        )
        pvalues_direct = np.array(
            [
                [
                    p_value(
                        null_distrib=null_distrib_directed[t, :, n],
                        statistic=demeaned_sig[t, n],
                        two_tail=False,
                    )
                    for t in range(self.graph_sig.shape[0])
                ]
                for n in range(self.graph_sig.shape[1])
            ]
        )

        # P-value significants
        multi_compare = (
            self.graph_sig.shape[0] * self.graph_sig.shape[1]
        )  # Number of tests
        cmap = plt.get_cmap("inferno")

        pvalues = [pvalues_direct, pvalues_undirect]
        poses = [self.pos_dir, self.pos_dir]
        figs = []
        norm_max = 0
        for k in range(2):
            pos = {
                i: poses[k][self.graph_nodes_id[i]]
                for i in range(len(self.graph_nodes_id))
            }
            pos_inv = {v: k for k, v in pos.items()}
            if self.config["correction"] == "bonferroni":
                decision = (
                    (pvalues[k] < (self.config["alpha"] / multi_compare))
                    .astype(float)
                    .sum(axis=1)
                )

            elif self.config["correction"] == "fdr":
                decision = (
                    multipletests(
                        pvalues[k].flatten(),
                        alpha=self.config["alpha"],
                        method="fdr_bh",
                    )[0]
                    .reshape(pvalues[k].shape)
                    .sum(axis=1)
                )
            else:
                raise ValueError("Invalid correction method specified in config.")

            # node_colors = decision / self.graph_sig.shape[0]
            node_colors = (
                decision / 150
            )  # so that the color matches the colobar up to 100 significant days, and saturates after that
            print(node_colors)
            norm_max = min(max(norm_max, node_colors.max()), 1)

            node_remax = node_colors / (norm_max + 0.1)

            # Plot all polygons from the GeoJSON file
            fig, ax = plt.subplots(figsize=(5, 3))

            map_background = load(
                os.path.join(self.path_to_resources, "map_background.pkl")
            )
            map_background.plot(ax=ax, color="gray", edgecolor="black", alpha=0.2)

            # Plot the graph nodes
            for idx, _ in enumerate(self.stations):
                x, y = self.stations_pos[idx][1], self.stations_pos[idx][0]
                if not (x, y) in pos_inv:
                    continue
                get_idx = pos_inv[(x, y)]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    color=cmap(node_remax[get_idx]),
                    markersize=8,
                    markeredgecolor="black",
                    alpha=0.8,
                )

            # Plot the edges
            for edge in self.graph.G.edges():
                start_pos = np.array(pos[edge[0]])[::-1]
                end_pos = np.array(pos[edge[1]])[::-1]
                direction = end_pos[:2] - start_pos[:2]
                arrow_start_pos = (
                    start_pos[:2] + direction / np.linalg.norm(direction) * 0.08
                )
                arrow_end_pos = (
                    end_pos[:2] - direction / np.linalg.norm(direction) * 0.08
                )
                if k == 0:
                    ax.add_patch(
                        patches.FancyArrowPatch(
                            (arrow_start_pos[1], arrow_start_pos[0]),
                            (arrow_end_pos[1], arrow_end_pos[0]),
                            arrowstyle="-|>",
                            mutation_scale=6,
                            color="black",
                            alpha=0.6,
                            lw=0.8,
                            connectionstyle="arc3,rad=.1",
                        )
                    )
                else:
                    # Draw an undirected (no arrowheads) curved edge
                    ax.add_patch(
                        patches.FancyArrowPatch(
                            (arrow_start_pos[1], arrow_start_pos[0]),
                            (arrow_end_pos[1], arrow_end_pos[0]),
                            arrowstyle="-",
                            mutation_scale=6,
                            color="black",
                            alpha=0.6,
                            lw=0.8,
                            connectionstyle="arc3,rad=.1",
                        )
                    )

            ax.set_axis_off()
            # Create a scatter plot for the colorbar
            sc = ax.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=1)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.8, location="left", pad=0.01)
            cbar.ax.tick_params(labelsize=self.fontsize)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname("Helvetica")

            ticklabels = [0, 30, 60, 90, 120, 150]
            cbar.set_ticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]))
            # cbar.set_ticklabels(
            #     (
            #         np.array([0, 0.195, 0.392, 0.62, 0.775, 1])
            #         * (pvalues[k] < alpha).astype(float).sum(axis=1).max()
            #     ).astype(int)
            # )
            cbar.set_ticklabels(ticklabels)
            cbar.set_label(
                "Number of significant days",
                fontsize=self.fontsize,
                fontname="Helvetica",
            )

            if not self.verbose:
                plt.close()
            plt.show()

            figs.append(fig)

        return figs, demeaned_sig, null_distrib_directed, null_distrib_undirected

    def run_experiment3(
        self, demeaned_sig, null_distrib_directed, null_distrib_undirected
    ):
        figs = []
        scale = 2

        node2plots = [12, 1]
        colors = ["r", "b"]
        for nidx, node2plot in enumerate(node2plots):
            self.logger.info(f"Station {self.graph_nodes_id[node2plot]}")

            fig, ax = plt.subplots(1, figsize=(12, 2.3))

            sample = -demeaned_sig[:, node2plot]
            # Calculate mean and standard deviation for directed and undirected null distributions
            mean_directed = -np.mean(null_distrib_directed[:, :, node2plot], axis=1)
            std_directed = scale * np.std(
                null_distrib_directed[:, :, node2plot], axis=1
            )
            mean_undirected = -np.mean(
                [np.real(x) for x in null_distrib_undirected[:, :, node2plot]], axis=1
            )
            std_undirected = scale * np.std(
                [np.real(x) for x in null_distrib_undirected[:, :, node2plot]], axis=1
            )

            # Plotting Main
            alpha = 0.3
            if nidx == 0:
                ax.plot(
                    sample,
                    label="Empirical: Coastal station",
                    color="darkorange",
                    linewidth=1.5,
                    alpha=0.9,
                    linestyle="--",
                )
            else:
                ax.plot(
                    sample,
                    label="Empirical: Inland station",
                    color="green",
                    linewidth=1.5,
                    alpha=0.9,
                    linestyle="--",
                )
            poly = ax.fill_between(
                range(len(mean_directed)),
                mean_directed - std_directed,
                mean_directed + std_directed,
                color=colors[0],
                label="Directed",
            )

            facecolor = poly.get_facecolor()
            edgecolor = poly.get_edgecolor()
            poly.set_facecolor(
                (facecolor[0][0], facecolor[0][1], facecolor[0][2], alpha)
            )  # RGBA with alpha=0.3
            poly.set_edgecolor(
                (edgecolor[0][0], edgecolor[0][1], edgecolor[0][2], 1)
            )  # RGBA with alpha=1.0

            poly2 = ax.fill_between(
                range(len(mean_undirected)),
                mean_undirected - std_undirected,
                mean_undirected + std_undirected,
                color=colors[1],
                label="Undirected",
            )

            facecolor = poly2.get_facecolor()
            edgecolor = poly2.get_edgecolor()
            poly2.set_facecolor(
                (facecolor[0][0], facecolor[0][1], facecolor[0][2], alpha)
            )  # RGBA with alpha=0.3
            poly2.set_edgecolor(
                (edgecolor[0][0], edgecolor[0][1], edgecolor[0][2], 1)
            )  # RGBA with alpha=1.0

            # Set xticks to be dates starting from January 1st
            ax.set_ylabel(
                "Temperature (SI)", fontsize=self.fontsize - 3, fontname="Helvetica"
            )
            date_range = pd.date_range(
                start="2023-01-01", periods=len(sample), freq="D"
            )
            # Group the dates by month and get the number of days for each month
            days_in_months = (
                date_range.to_series().groupby(date_range.month).size().tolist()
            )
            days_xticks = np.cumsum([0] + days_in_months[:-1])
            ax.set_xticks(days_xticks)  # Set xticks every 30 days
            ax.set_xticklabels(
                date_range.strftime("%b %d")[np.array(days_xticks)],
                rotation=45,
                ha="right",
                fontsize=self.fontsize - 3,
            )
            ax.set_xlim(-5, len(sample) + 5)
            ax.tick_params(axis="y", labelsize=self.fontsize - 3)
            ax.tick_params(axis="x", labelsize=self.fontsize - 3)
            ax.legend(ncol=3, fontsize=self.fontsize - 3, loc=(0.1, 0.05))
            ax.grid(visible=True, which="both", axis="x", linestyle="--", alpha=0.5)

            # Ensure current axis texts use Helvetica
            ax.set_title(ax.get_title(), fontname="Helvetica")
            ax.xaxis.label.set_fontname("Helvetica")
            ax.yaxis.label.set_fontname("Helvetica")
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_fontname("Helvetica")

            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontname("Helvetica")
                if leg.get_title() is not None:
                    leg.get_title().set_fontname("Helvetica")

            # Also enforce for all axes/texts in the figure (if present)
            for a in fig.get_axes():
                a.title.set_fontname("Helvetica")
                a.xaxis.label.set_fontname("Helvetica")
                a.yaxis.label.set_fontname("Helvetica")
                for lbl in a.get_xticklabels() + a.get_yticklabels():
                    lbl.set_fontname("Helvetica")
                lg = a.get_legend()
                if lg:
                    for t in lg.get_texts():
                        t.set_fontname("Helvetica")
                    if lg.get_title():
                        lg.get_title().set_fontname("Helvetica")

            figs.append(fig)
            if not self.verbose:
                plt.close()
            plt.show()

        return figs


if __name__ == "__main__":
    run()
