import numpy as np
import pytest
from CartPoleDQN import moving_average, main
from config import cfg

def test_smoke_main_runs_zero_episodes(monkeypatch):
    """
    Smoke-test main(cfg) with 0 episodes: should import and return
    without error when episodes_dqn=0.
    """
    # Prevent plotting from blocking
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


    cfg.dqn.episodes = 0
    main(cfg) # if raises, test fails