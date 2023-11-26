import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from fastdtw import fastdtw
from scipy.fft import fft
import pywt
from typing import Any, List

# Type definitions
TimeSeries = List[float]
CovarianceMatrix = Any


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
def dtw_error_euclidean(
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


# Function to extract wavelet analysis features from time series
def wavelet_analysis_features(time_series: TimeSeries) -> Any:
    """
    Apply wavelet transformation to the time series data.

    Args:
    - time_series (TimeSeries): Time series data.

    Returns:
    - Any: Coefficients of wavelet transformation.
    """
    # Apply wavelet transformation (you can choose the desired wavelet)
    # Example: 'db4' wavelet and level of decomposition 3
    coeffs = pywt.wavedec(time_series, 'db4', level=3)
    return coeffs


def mahalanobis_distance(
        point1: Any,
        point2: Any,
        cov_matrix: CovarianceMatrix) -> float:
    """
    Calculate Mahalanobis distance between two points.

    Args:
    - point1 (Any): First point.
    - point2 (Any): Second point.
    - cov_matrix (CovarianceMatrix): Covariance matrix.

    Returns:
    - float: Mahalanobis distance between the points.
    """
    return mahalanobis(
        point1, point2, cov_matrix)


def dtw_with_mahalanobis(
        time_series1: TimeSeries,
        time_series2: TimeSeries,
        cov_matrix: CovarianceMatrix) -> float:
    """
    Calculate DTW distance using Mahalanobis distance as the metric.

    Args:
    - time_series1 (TimeSeries): First time series data.
    - time_series2 (TimeSeries): Second time series data.
    - cov_matrix (CovarianceMatrix): Covariance matrix for Mahalanobis dist.

    Returns:
    - float: DTW distance using Mahalanobis distance as the metric.
    """
    def distance_function(x: Any, y: Any) -> float:
        return mahalanobis_distance(x, y, cov_matrix)

    distance, _ = fastdtw(time_series1, time_series2, dist=distance_function)
    return distance


def fourier_features(time_series: TimeSeries) -> Any:
    """
    Compute Fourier transformation (using FFT) of the time series data.

    Args:
    - time_series (TimeSeries): Time series data.

    Returns:
    - Any: Result of the Fourier transformation.
    """
    return fft(time_series)


def calculate_covariance_matrix(time_series1: np.ndarray, time_series2: np.ndarray) -> np.ndarray:
    """
    Computes the covariance matrix for two time series.

    Args:
    time_series1 (np.ndarray): First time series.
    time_series2 (np.ndarray): Second time series.

    Returns:
    np.ndarray: Covariance matrix of the two time series.
    """
    # Combine two time series into a single matrix
    stacked_series = np.vstack((time_series1, time_series2))

    # Calculate the covariance matrix
    covariance_matrix = np.cov(stacked_series)

    return covariance_matrix
