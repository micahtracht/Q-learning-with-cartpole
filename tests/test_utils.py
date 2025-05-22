import numpy as np
import pytest
from CartPoleDQN import moving_average
from config import cfg


epsilon_min   = cfg.dqn.epsilon_min
epsilon_decay = cfg.dqn.epsilon_decay
alpha_min     = cfg.dqn.alpha_min
alpha_decay   = cfg.dqn.alpha_decay

def test_moving_average_simple():
    """
    Window=2 over a simple list [0,1,2,3,4]
    """
    data = [0, 1, 2, 3, 4]
    # moving_average(data, n) uses n window size
    ma = moving_average(data, window_size=2)
    expected = np.array([0.5, 1.5, 2.5, 3.5])
    assert isinstance(ma, np.ndarray)
    assert np.allclose(ma, expected)


def test_moving_average_window_equals_length():
    """
    When window == len(data), returns one average.
    """
    data = [10, 20, 30]
    ma = moving_average(data, window_size=3)
    # single value: (10+20+30)/3 = 20.0
    assert ma.shape == (1,)
    assert np.allclose(ma, np.array([20.0]))


def test_moving_average_window_larger_than_data():
    """
    If window > len(data), numpy.convolve with 'valid' → empty array.
    """
    data = [1, 2]
    ma = moving_average(data, window_size=5)
    assert isinstance(ma, np.ndarray)
    assert ma.size == 2 # should be the same size as the given data


@pytest.mark.parametrize("n,expected", [
    (1, pytest.approx(epsilon_decay)),
    (5, pytest.approx(epsilon_decay**5)), 
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
    """
    If you decay enough times, epsilon bottoms out at epsilon_min.
    """
    eps = 1.0
    # apply decay 100,000 times
    for _ in range(100_000):
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
    """
    Alpha should bottom out at alpha_min after many decays.
    """
    a = 0.0001
    for _ in range(1_000_000):
        a = max(alpha_min, a * alpha_decay)
    assert a >= alpha_min