import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def error_function(time_series1, time_series2):
    if len(time_series1) != len(time_series2):
        raise ValueError("Длины временных рядов должны быть одинаковыми")

    # Рассчитываем среднеквадратичную ошибку (MSE)
    mse = np.mean((time_series1 - time_series2) ** 2)
    return mse


def correlation_error(time_series1, time_series2):
    if len(time_series1) != len(time_series2):
        raise ValueError("Длины временных рядов должны быть одинаковыми")

    # Вычисляем корреляцию Пирсона между временными рядами
    correlation = np.corrcoef(time_series1, time_series2)[0, 1]
    # Переводим корреляцию в ошибку:
    # 1 - корреляция (чем выше корреляция, тем меньше ошибка)
    error = 1 - correlation
    return error


def dtw_error(time_series1, time_series2):
    # Вычисляем расстояние DTW между временными рядами
    distance, _ = fastdtw(time_series1, time_series2, dist=euclidean)
    return distance
