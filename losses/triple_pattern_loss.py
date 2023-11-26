from typing import Any, Dict, List
import numpy as np
from base_functions import wavelet_analysis_features, fourier_features
from base_functions import dtw_with_mahalanobis


# Type definitions
TimeSeries = List[float]
CovarianceMatrix = Any


# Custom loss function with annotations
def triple_similarity_loss(
        original_time_series: TimeSeries,
        reconstructed_time_series: TimeSeries,
        weights: Dict[str, float], cov_matrix: CovarianceMatrix) -> float:
    """
    Calculate a custom loss function based on different features of time series.

    Args:
    - original_time_series (TimeSeries): Original time series data.

    - reconstructed_time_series (TimeSeries): Reconstructed time series data.

    - weights (Dict[str, float]): Dictionary of weights for different error components.

    - cov_matrix (CovarianceMatrix): Covariance matrix for Mahalanobis distance.

    Returns:
    - float: Total error calculated using specified weights and error components.
    """

    # Getting features for original and reconstructed time series
    original_wavelet = wavelet_analysis_features(original_time_series)
    reconstructed_wavelet = wavelet_analysis_features(
        reconstructed_time_series)

    original_fourier = fourier_features(original_time_series)
    reconstructed_fourier = fourier_features(reconstructed_time_series)

    # Calculate DTW distance with Mahalanobis distance
    dtw_mahalanobis_distance = dtw_with_mahalanobis(original_time_series,
                                                    reconstructed_time_series,
                                                    cov_matrix)

    # Calculate error for wavelet analysis
    wavelet_error = np.linalg.norm(np.array(original_wavelet) - np.array(reconstructed_wavelet))

    # Calculate error for Fourier features
    fourier_error = np.linalg.norm(np.array(original_fourier) - np.array(reconstructed_fourier))

    # Normalize errors before applying weights
    max_error = max(wavelet_error, dtw_mahalanobis_distance, fourier_error)
    if max_error == 0:  # if all errors are 0
        max_error = 1  # to avoid division by zero

    normalized_wavelet_error = wavelet_error / max_error
    normalized_dtw_mahalanobis_distance = dtw_mahalanobis_distance / max_error
    normalized_fourier_error = fourier_error / max_error

    # Calculate total error considering weights
    total_error = (weights['wavelet'] * normalized_wavelet_error +
                   weights['dtw_mahalanobis'] * normalized_dtw_mahalanobis_distance +
                   weights['fourier'] * normalized_fourier_error)

    return total_error
