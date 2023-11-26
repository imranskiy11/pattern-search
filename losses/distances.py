from scipy.spatial.distance import mahalanobis
from fastdtw import fastdtw
from typing import Any, Tuple

# Type definitions for points and covariance matrix
Point = Tuple[float, ...]  # Type for a point represented as a tuple of numbers
# Type for covariance matrix (can be specific to your implementation)
CovarianceMatrix = Any


# Function to calculate Mahalanobis distance between two points
def mahalanobis_distance(
        point1: Point, point2: Point, cov_matrix: CovarianceMatrix) -> float:
    """
    Calculate Mahalanobis distance between two points.

    Args:
    - point1 (Point): First point.
    - point2 (Point): Second point.
    - cov_matrix (CovarianceMatrix): Covariance matrix.

    Returns:
    - float: Mahalanobis distance between the points.
    """
    return mahalanobis(point1, point2, cov_matrix)


# Function to define Mahalanobis distance for fastdtw
def distance_function(
        x: Point, y: Point, your_covariance_matrix: CovarianceMatrix) -> float:
    """
    Define Mahalanobis distance for fastdtw.

    Args:
    - x (Point): First point.
    - y (Point): Second point.
    - your_covariance_matrix (CovarianceMatrix): Covariance matrix.

    Returns:
    - float: Mahalanobis distance between the points.
    """
    return mahalanobis_distance(x, y, your_covariance_matrix)


if __name__ == '__main__':
    # Your time series data (time_series1, time_series2)
    time_series1, time_series2 = None, None
    # Call fastdtw using Mahalanobis distance
    distance, path = fastdtw(
        time_series1, time_series2, dist=distance_function)
