import random
from typing import Sequence
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from dqn_agent import DQN
from replay_buffer import ReplayBuffer
from config import Config

# - Globals & Helpers -
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define config
cfg = Config()

random.seed(cfg.env.seed)
np.random.seed(cfg.env.seed)
torch.manual_seed(cfg.env.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg.env.seed)


def moving_average(data: Sequence[float], window_size: int=100):
    """
    Compute the moving average of a 1D sequence.

    Args:
        data: A sequence of numeric values.
        window_size: The number of elements over which to average/convolve.

    Returns:
        A numpy array of size len(data) - window_size + 1 containing the moving average values.
    """
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def decay(value: float, decay_rate: float, min_val: float) -> float:
    return max(min_val, value * decay_rate)

# - Main training loop -
def main(cfg: Config):
    epsilon = cfg.dqn.epsilon
    alpha = cfg.dqn.alpha
    # Set up environment
    env = gym.make(cfg.env.env_id)
    env.observation_space.seed(cfg.env.seed)
    env.action_space.seed(cfg.env.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Networks & optimizer
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    
    policy_net.to(device)
    target_net.to(device)
    
    policy_net.train()
    target_net.eval()
    
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    replay_buffer = ReplayBuffer(capacity=cfg.buffer_size)
    
    solved = False # only save once
    episode_rewards = []
    print('Started training.')
    
    for episode in range(cfg.dqn.episodes):
        state, _ = env.reset()
        total_reward = 0
        for step in range(cfg.dqn.max_steps):
            # eps greedy policy selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) #.unsqueeze(0) adds a batch dimension to the tensor since the DQN expects that. It tells us the number of examples we're getting at once - in this case 1 - hence why we add a 1.
                q_values = policy_net(state_tensor) # get all q values from the model
                action = torch.argmax(q_values).item() # select the best action according to the model
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # is the episode finished
            
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= cfg.batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(cfg.batch_size)
            
                # convert to tensors
                states_np = np.stack(states, axis=0).astype(np.float32)
                states = torch.from_numpy(states_np).to(device)
                actions = torch.tensor(actions, dtype=torch.int64, device = device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device = device).unsqueeze(1)
                next_states_np = np.stack(next_states, axis=0).astype(np.float32)
                next_states = torch.from_numpy(next_states_np).to(device)
                dones = torch.tensor(dones, dtype=torch.float32, device = device).unsqueeze(1)
                
                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)
                
                # max a' Q(s', a') from target net
                with torch.no_grad():
                    best_next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
                    next_q_values = target_net(next_states).gather(1, best_next_actions)
                    targets = rewards + cfg.dqn.gamma * (1 - dones) * next_q_values
                
                # loss & backprop
                if len(replay_buffer) > cfg.warmup:
                    loss = nn.MSELoss()(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()
            if done:
                break
        
        # Decay schedules
        epsilon = decay(epsilon, cfg.dqn.epsilon_decay, cfg.dqn.epsilon_min)
        alpha = decay(alpha, cfg.dqn.alpha_decay, cfg.dqn.alpha_min)
        
        # sync target net
        if episode % cfg.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(total_reward)
        
        # Logging & saving
        if episode % 10 == 0:
            recent = episode_rewards[-cfg.dqn.window:]
            recent_avg = sum(recent) / len(recent)
            print(f"Ep {episode}  R= {total_reward:.0f}, eps = {epsilon:.3f}  alpha = {alpha:.5f} recent avg = {recent_avg:.1f}")
            if not solved and recent_avg >= cfg.dqn.max_steps:
                solved = True
                torch.save(policy_net.state_dict(), cfg.save_path)
                print(f'Solved! Model saved to {cfg.save_path}')
        
    plot_rewards(episode_rewards, cfg.dqn.window)


def plot_rewards(episode_rewards, window_dqn):
    smoothed = moving_average(episode_rewards, window_dqn)
    
    plt.subplot(1,2,2)
    plt.plot(smoothed, label=f"{window_dqn}-episode MA")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Smoothed Reward Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(cfg)