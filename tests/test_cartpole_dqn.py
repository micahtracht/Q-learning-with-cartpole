import numpy as np
import pytest

# We import *only* the helpers and constants — avoid pulling in the training loop
from CartPoleDQN import moving_average, epsilon_min, epsilon_decay, alpha_min, alpha_decay


def test_moving_average_simple():
    """
    Window=2 over a simple list [0,1,2,3,4].
    """
    data = [0, 1, 2, 3, 4]
    ma = moving_average(data, n=2)
    # (0+1)/2=0.5, (1+2)/2=1.5, (2+3)/2=2.5, (3+4)/2=3.5
    expected = np.array([0.5, 1.5, 2.5, 3.5])
    assert isinstance(ma, np.ndarray)
    assert np.allclose(ma, expected)


def test_moving_average_window_equals_length():
    """
    If window == len(data), it should return exactly 1 average.
    """
    data = [10, 20, 30]
    ma = moving_average(data, n=3)
    # single value: (10+20+30)/3 = 20.0
    assert ma.shape == (1,)
    assert np.allclose(ma, np.array([20.0]))


def test_moving_average_window_larger_than_data():
    """
    If window > len(data), numpy.convolve with 'valid' should make an empty array.
    """
    data = [1, 2]
    ma = moving_average(data, n=5)
    assert isinstance(ma, np.ndarray)
    assert ma.size == 0


@pytest.mark.parametrize("n,expected", [
    (1, pytest.approx(1.0)),
    (5, pytest.approx(0.992**5)), 
])
def test_epsilon_decay_math(n, expected):
    """
    Manually compute epsilon_n = max(epsilon_min, 1 * epsilon_decay**n).
    For small n this stays above epsilon_min.
    """
    eps0 = 1.0
    eps = eps0
    for _ in range(n):
        eps = max(epsilon_min, eps * epsilon_decay)
    assert eps == expected


def test_epsilon_decays_to_minimum():
    """If you decay enough times, epsilon bottoms out at epsilon_min."""
    eps = 1.0
    # apply decay 10,000 times
    for _ in range(100000):
        eps = max(epsilon_min, eps * epsilon_decay)
    assert eps == epsilon_min


@pytest.mark.parametrize("n,expected", [
    (1, pytest.approx(0.0001 * alpha_decay)),  # single decay step
    (10, pytest.approx(max(alpha_min, 0.0001 * (alpha_decay**10)))),
])
def test_alpha_decay_math(n, expected):
    """
    alpha_n = max(alpha_min, alpha0 * alpha_decay**n).
    """
    a0 = 0.0001
    a = a0
    for _ in range(n):
        a = max(alpha_min, a * alpha_decay)
    assert a == expected


def test_alpha_decays_to_minimum():
    """Alpha should bottom out at alpha_min after many decays (1,000,000 in this case)."""
    a = 0.0001
    for _ in range(1000000):
        a = max(alpha_min, a * alpha_decay)
    assert a == alpha_min
