"""
Utilities for the asymmetry surrogate experiments.

This module provides utility functions and classes for conducting experiments
related to asymmetry surrogates in the context of graph signal processing.
"""

import numpy as np
from scipy.stats import binned_statistic
from flowgsp.surrogates import Surrogate
from typing import Optional
from tqdm import tqdm
import pandas as pd


def timeseries_null_models_generator(
    timeseries_data: np.ndarray,
    idx_of_interest: Optional[np.ndarray],
    surg: Surrogate,
    surg_u: Surrogate,
    nrands: int = 99,
    verbose: bool = True,
):
    """
    Generate null models for a timeseries data.

    Parameters:
    ----------
        timeseries_data (np.ndarray): The input timeseries data.
        use_GD (bool, optional): Whether to use gradient descent optimization. Defaults to True.
        idx_of_interest (np.ndarray, optional): The indices of interest in the timeseries data. Defaults to None, which means all indices.
        nrands (int, optional): The number of random surrogate samples to generate. Defaults to 99.
        cutoff (float, optional): The maximum allowed loss for the gradient descent optimization. Defaults to 1e-4.

    Returns:
    --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The null distributions for directed, undirected, and naive surrogate models.
    """

    timecourse_null_directed = []
    timecourse_null_undirected = []
    timecourse_null_naive = []
    if idx_of_interest is None:
        idx_of_interest = np.arange(timeseries_data.shape[0])
    for tidx in tqdm(idx_of_interest, disable=not verbose):
        null_distrib_naive = surg.naive_random_surrogate(
            timeseries_data[tidx], nrands=nrands
        )
        null_distrib_undirected = surg_u.undirected_random_surrogate(
            timeseries_data[tidx], nrands=nrands
        )
        null_distrib_directed = surg.directed_random_surrogate(
            timeseries_data[tidx], nrands=nrands
        )

        timecourse_null_naive.append(null_distrib_naive)
        timecourse_null_undirected.append(null_distrib_undirected)
        timecourse_null_directed.append(null_distrib_directed)

    timecourse_null_directed = np.array(timecourse_null_directed)
    timecourse_null_undirected = np.array(timecourse_null_undirected)
    timecourse_null_naive = np.array(timecourse_null_naive)

    return timecourse_null_directed, timecourse_null_undirected, timecourse_null_naive


def rasterize(fp, tp, dense=100, eps=1e-5):
    """
    Rasterize the data into bins.
    """
    # Bin edges
    bin_edges = np.logspace(np.log10(fp.min() + eps), np.log10(fp.max()), dense + 1)

    # Compute mean TP in each FP bin
    rY, _, _ = binned_statistic(fp, tp, statistic="mean", bins=bin_edges)
    rVar, _, _ = binned_statistic(fp, tp, statistic="std", bins=bin_edges)
    rX = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers

    return rX, rY, rVar


def interpolate_nans(arr):
    arr = np.array(arr, dtype=float)
    nans = np.isnan(arr)
    not_nans = ~nans
    indices = np.arange(len(arr))
    arr[nans] = np.interp(indices[nans], indices[not_nans], arr[not_nans])
    return arr


def extract_values(
    df, feature_of_interest, id_label, column_id, start_timestamp, end_timestamp
):
    """
    Extracts values for a given column ID and start and end timestamps.

    Parameters:
    df (pd.DataFrame): The DataFrame to extract values from.
    column_id (int): The column ID to extract values for.
    start_timestamp (str or pd.Timestamp): The start timestamp.
    end_timestamp (str or pd.Timestamp): The end timestamp.

    Returns:
    pd.Series: A series containing the extracted values.
    """
    # Ensure the timestamps are in datetime format
    start_timestamp = pd.to_datetime(start_timestamp)
    end_timestamp = pd.to_datetime(end_timestamp)

    # Extract the values
    df_of_interest = df[df[column_id] == id_label]

    extracted_values = df_of_interest.loc[start_timestamp:end_timestamp][
        feature_of_interest
    ]

    return extracted_values
