import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from typing import List


# Function to calculate mean squared error (MSE) between two time series
def error_function(
        time_series1: np.ndarray, time_series2: np.ndarray) -> float:
    """
    Calculates the mean squared error (MSE) between two time series.

    Args:
    - time_series1 (np.ndarray): First time series data.
    - time_series2 (np.ndarray): Second time series data.

    Returns:
    - float: Mean squared error between the time series.
    """
    if len(time_series1) != len(time_series2):
        raise ValueError("Lengths of time series should be the same")

    mse = np.mean((time_series1 - time_series2) ** 2)
    return mse


# Function to calculate Pearson correlation-based error between two time series
def correlation_error(
        time_series1: np.ndarray, time_series2: np.ndarray) -> float:
    """
    Calculates the error based on Pearson correlation
    coefficient between two time series.

    Args:
    - time_series1 (np.ndarray): First time series data.
    - time_series2 (np.ndarray): Second time series data.

    Returns:
    - float: Error based on Pearson correlation
    coefficient between the time series.
    """
    if len(time_series1) != len(time_series2):
        raise ValueError("Lengths of time series should be the same")

    correlation = np.corrcoef(time_series1, time_series2)[0, 1]
    error = 1 - correlation
    return error


# Function to calculate Dynamic Time Warping (DTW) distance
# between two time series
def dtw_error(
        time_series1: List[float], time_series2: List[float]) -> float:
    """
    Calculates the Dynamic Time Warping (DTW) distance between two time series.

    Args:
    - time_series1 (List[float]): First time series data.
    - time_series2 (List[float]): Second time series data.

    Returns:
    - float: Dynamic Time Warping (DTW) distance between the time series.
    """
    distance, _ = fastdtw(time_series1, time_series2, dist=euclidean)
    return distance


def mlh():
    pass
