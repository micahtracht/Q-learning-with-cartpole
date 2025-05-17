from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = 'CartPole-v1'
    seed: int = 42
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    alpha: float = 0.0001
    alpha_decay: float = 1
    alpha_min: float = 0
    epsilon: float = 1
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.992
    target_update_freq: int = 10
    episodes: int = 2_500
    max_steps: int = 500
    window: int = 100
    save_path: str = 'dqn_cartpole_solved.pth'