import numpy as np
from typing import Sequence

def moving_average(data: Sequence[float], window_size: int=100) -> np.ndarray:
    """
    Compute the moving average of a 1D sequence.

    Args:
        data: A sequence of numeric values.
        window_size: The number of elements over which to average/convolve.

    Returns:
        A numpy array of size len(data) - window_size + 1 containing the moving average values.
    """
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("window_size must be greater than 0 and an integer.")
    n = len(data)
    if window_size > n:
        if n == 0: # if empty
            return np.array([], dtype=float)
        mean = float(np.sum(data) / n)
        return np.full(n, mean, dtype=float)
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def decay(val: float, decay_rate: float, min_val: float) -> float:
    return max(val * decay_rate, min_val)