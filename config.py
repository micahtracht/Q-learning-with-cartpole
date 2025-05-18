from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    # General
    env_id: str = 'CartPole-v1'
    seed: int = 42
    
    # DQN Hyperparameters
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma_dqn: float = 0.99
    alpha_dqn: float = 0.0001
    alpha_decay_dqn: float = 1
    alpha_min_dqn: float = 0
    epsilon_dqn: float = 1
    epsilon_min_dqn: float = 0.01
    epsilon_decay_dqn: float = 0.992
    episodes_dqn: int = 2_500
    target_update_freq: int = 10
    max_steps: int = 500
    window_dqn: int = 100
    save_path: str = 'dqn_cartpole_solved.pth'
    warmup: int = 1000
    
    # Tabular Q-learning hyperparameters
    n_bins: int = 10
    n_angle_bins: int = 20
    lower_bounds: np.ndarray = np.array([-4.8, -5.0, -0.418, -5.0])
    upper_bounds: np.ndarray = np.array([ 4.8,  5.0,  0.418,  5.0])
    gamma_tabular: float = 0.99
    alpha_tabular: float = 0.3
    alpha_decay_tabular: float = 0.999
    alpha_min_tabular: float = 0.005
    epsilon_tabular: float = 1
    epsilon_decay_tabular: float = 0.999
    epsilon_min_tabular: float = 0.01
    epssiodes_tabular: int = 10_000
    max_steps_tabular: int = 500
    window_tabular: int = 100
    
    