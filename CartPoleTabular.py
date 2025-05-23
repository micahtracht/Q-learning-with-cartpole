import numpy as np
if not hasattr(np, 'bool8'): # fix annoying error with gym, this took me 30 minutes. Honestly could have been worse.
    np.bool8 = np.bool_
import gym
from discretizer import Discretizer
import matplotlib.pyplot as plt
from config import cfg, Config
from utils import moving_average, decay

# - Globals & Helpers -
def clip_obs(obs: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    '''
    Clips each feature of obs into range [low_i, high_i]. Important to avoid wrap arounds.
    
    Args:
        obs (np.ndarray): Array of observations, shape (n_features) or greater.
        low (np.ndarray): Array of lower bounds, same shape as obs.
        high (np.ndarray): Array of upper bounds, same shape as obs.
    
    Returns:
        np.ndarray: The clipped array of observations
    '''
    return np.clip(obs, low, high)

def main(cfg: Config, seed: int = cfg.env.seed):
    env = gym.make(cfg.env.env_id)
    env.observation_space.seed(cfg.env.seed)
    env.action_space.seed(cfg.env.seed)
    env.action_space.n = 2  # discrete two-action CartPole
    
    bins = [cfg.tabular.n_bins, cfg.tabular.n_bins, cfg.tabular.n_angle_bins, cfg.tabular.n_bins]
    Q = np.zeros(bins + [env.action_space.n], dtype=float)
    disc = Discretizer(bins_per_feature=bins, lower_bounds=cfg.tabular.lower_bounds, upper_bounds=cfg.tabular.upper_bounds)
    
    epsilon = cfg.tabular.epsilon
    alpha = cfg.tabular.alpha
    episode_rewards = []
    
    print('Starting tabular Q-learning')
    for episode in range(cfg.tabular.episodes):
        obs, _ = env.reset()
        obs = clip_obs(obs, cfg.tabular.lower_bounds, cfg.tabular.upper_bounds)
        state = disc.discretize(obs)
        total_reward = 0.0
        
        for _ in range(cfg.tabular.max_steps):
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_obs = clip_obs(next_obs, cfg.tabular.lower_bounds, cfg.tabular.upper_bounds)
                next_state = disc.discretize(next_obs)
                
                # Bellman update
                td_target = reward + cfg.tabular.gamma * np.max(Q[next_state])
                td_error = td_target - Q[state][action]
                Q[state][action] += alpha * td_error
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                
        # decay schedules
        alpha = decay(alpha, cfg.tabular.alpha_decay, cfg.tabular.alpha_min)
        epsilon = decay(epsilon, cfg.tabular.epsilon_decay, cfg.tabular.epsilon_min)
        
        episode_rewards.append(total_reward)
        if episode % 1000 == 0:
            print(f"Episode {episode},Reward={total_reward:.1f}, epsilon={epsilon:.3f}  alpha={alpha:.3f}")
    if len(episode_rewards) != 0:
        avg_reward = sum(episode_rewards)/(len(episode_rewards))
        print(f"Average reward over {cfg.tabular.episodes} episodes: {avg_reward:.2f}")
        plot_rewards(episode_rewards, cfg.tabular.window)


def plot_rewards(episode_rewards, window_tabular):
    smoothed = moving_average(episode_rewards, window_tabular)
    
    plt.subplot(1,2,2)
    plt.plot(smoothed, label=f"{window_tabular}-episode MA")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Smoothed Reward Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(cfg)
