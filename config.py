from dataclasses import dataclass
import numpy as np
import yaml
from pathlib import Path
@dataclass
class EnvConfig:
    env_id: str
    seed: int

@dataclass
class DQNConfig:
    buffer_size: int
    batch_size: int
    gamma: float
    alpha: float
    alpha_decay: float
    alpha_min: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    episodes: int
    target_update_freq: int
    max_steps: int
    window: int
    save_path: str
    warmup: int
    
@dataclass
class TabularConfig:
    n_bins: int
    n_angle_bins: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    gamma: float
    alpha: float
    alpha_decay: float
    alpha_min: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    episodes: int
    max_steps: int
    window: int

@dataclass
class Config:
    env: EnvConfig
    dqn: DQNConfig
    tabular: TabularConfig

# - loader -
def load_config(path: Path = None) -> Config:
    if path is None:
        path = Path(__file__).parent / 'config.yaml'
    raw = yaml.safe_load(path.read_text())
    
    env_cfg = EnvConfig(**raw['env'])
    dqn_cfg = DQNConfig(**raw['dqn'])
    
    # tabular section, convert lists into numpy ndarrays
    tab = raw['tabular']
    tabular_cfg = TabularConfig(
        n_bins = tab['n_bins'],
        n_angle_bins = tab['n_angle_bins'],
        lower_bounds = np.array(tab['lower_bounds'], dtype=float),
        upper_bounds = np.array(tab['upper_bounds'], dtype=float),
        gamma         = tab["gamma"],
        alpha         = tab["alpha"],
        alpha_decay   = tab["alpha_decay"],
        alpha_min     = tab["alpha_min"],
        epsilon       = tab["epsilon"],
        epsilon_decay = tab["epsilon_decay"],
        epsilon_min   = tab["epsilon_min"],
        episodes      = tab["episodes"],
        max_steps     = tab["max_steps"],
        window        = tab["window"],
    )
    return Config(env=env_cfg, dqn=dqn_cfg, tabular=tabular_cfg)
cfg = load_config()