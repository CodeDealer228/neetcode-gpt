import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_div(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)

        z = np.dot(x,w) + b
        y_hat = sigmoid(z)
        loss = (y_hat - y_true) ** 2

        dl_dyhat = (y_hat - y_true)
        dyhat_dz = sigmoid_div(z)
        dz_dw = x
        

        dl_dw = dl_dyhat * dyhat_dz * dz_dw
        dl_db = dl_dyhat * dyhat_dz
        return(np.round(dl_dw, 5), np.round(float(dl_db), 5))

