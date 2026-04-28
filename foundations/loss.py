import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        # Векторно: mean вместо sum/len, clip для стабильности
        y_pred = np.clip(y_pred, eps, 1 - eps)
        term1 = y_true * np.log(y_pred)
        term2 = (1 - y_true) * np.log(1 - y_pred)
        loss = -np.mean(term1 + term2)
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        # Векторно по сэмплам: clip + sum(axis=1) + mean
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return np.round(loss, 4)