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

def test_main_smoke_runs_zero_episodes(monkeypatch):
    """
    Smoke‐test main(cfg) with 0 episodes: it should return immediately
    and not raise when cfg.episodes_tabular = 0.
    """
    cfg.tabular.episodes = 0
    # Also override plt.show so we don't block
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    # Call main. no exception = pass
    main(cfg)