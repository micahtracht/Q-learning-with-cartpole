import numpy as np
import pytest
from config import Config
# import the two functions from your script
from CartPoleTabular import clip_obs, moving_average, main
from config import cfg

def test_clip_obs_in_bounds():
    """
    Values inside [low, high] should be unchanged.
    """
    low = np.array([-1.0,  0.0, 2.0])
    high = np.array([ 1.0,  5.0, 4.0])
    obs = np.array([ 0.0,  2.5, 3.0])
    out = clip_obs(obs, low, high)
    assert np.allclose(out, obs)

def test_clip_obs_below_and_above():
    """
    Values below low clamp to low, above high clamp to high.
    """
    low = np.array([0.0, -1.0])
    high = np.array([2.0,  1.0])
    obs = np.array([-5.0,  5.0])
    expected = np.array([ 0.0,  1.0])
    out = clip_obs(obs, low, high)
    assert np.allclose(out, expected)

def test_clip_obs_broadcasting():
    """
    clip_obs should broadcast low/high to obs shape when needed.
    """
    low = np.array([0.0])
    high = np.array([1.0])
    obs = np.array([-1.0, 0.5, 2.0])
    expected = np.array([0.0, 0.5, 1.0])
    out = clip_obs(obs, low, high)
    assert np.allclose(out, expected)

def test_moving_average_simple():
    """
    A simple increasing list with window = 2. Tests  basic functionality.
    """
    data   = [0, 1, 2, 3, 4]
    result = moving_average(data, window_size=2)
    assert np.allclose(result, np.array([0.5, 1.5, 2.5, 3.5]))

def test_moving_average_window_equals_length():
    """
    When window == len(data), should return a single average (the mean of the data).
    """
    data   = [10, 20, 30]
    result = moving_average(data, window_size=3)
    assert np.allclose(result, np.array([(10 + 20 + 30) / 3]))

def test_moving_average_window_larger_than_data():
    """
    If window > len(data), then numpy.convolve returns an empty array if 'valid' is used.
    """
    data   = [1, 2]
    result = moving_average(data, window_size=5)
    assert isinstance(result, np.ndarray)
    assert result.size == 0

def test_main_smoke_runs_zero_episodes(monkeypatch):
    """
    Smoke‐test main(cfg) with 0 episodes: it should return immediately
    (and not raise) when cfg.episodes_tabular = 0.
    """
    cfg = Config()
    cfg.tabular.episodes = 0
    # Also override plt.show so we don't block
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    # Call main; no exception => pass
    main(cfg)