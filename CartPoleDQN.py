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
import wandb

SEED = 42 # can anyone explain why this is the standard seed? I just see it everywhere
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED) # originally typed torch.manuel_seed(SEED), spent 10 mins scratching my head at the error. Unfortunately my auto-typo flagger did not catch 'manuel'.
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters for our DQN agent:
env_id = 'CartPole-v1'
buffer_size = 200000
batch_size = 256
gamma = 0.99 # 1 leads to stochastic instability
alpha, alpha_decay, alpha_min = 0.0001, 1, 0
target_update_freq = 10 # number of steps we take between theta_target <- theta
epsilon, epsilon_min, epsilon_decay = 1, 0.01, 0.992 
episodes = 10000
max_steps = 500 
env = gym.make(env_id)
env.observation_space.seed(SEED)
env.action_space.seed(SEED)
state_dim = env.observation_space.shape[0] # 4
action_dim = env.action_space.n # 2

# Define Q-net and target-net:
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
policy_net.to(device)
target_net.to(device)
policy_net.train()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # disable the gradient computation for the target net

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
replay_buffer = ReplayBuffer(capacity=buffer_size)

save_path = "dqn_cartpole_solved.pth"
solved = False # only save once
window = 100

# actual training loop
print('starting training')
episode_rewards = []
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
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
        
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
            # Now we do the actual updating of the model
            # Step 1: Convert everything to tensors so we can process it
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
                best_next_actions = policy_net(next_states).argmax(dim=1, keepdim=True) # Policy selects the best action (use keepdim to keep the dimensions 2d as .gather needs that later)
                next_q_values = target_net(next_states).gather(1, best_next_actions) # Target evaluates that action
                targets = rewards + gamma * (1 - dones) * next_q_values
            
            warmup = 1000
            if len(replay_buffer) > warmup and len(replay_buffer) > batch_size:
                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    alpha = max(alpha_min, alpha * alpha_decay)
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        recent = episode_rewards[-window:]
        recent_avg = sum(recent) / len(recent)
        print(f"Ep {episode}  R= {total_reward:.0f}, eps = {epsilon:.3f}  alpha = {alpha:.5f} recent avg = {recent_avg:.1f}")
        if not solved and recent_avg >= max_steps:
            solved = True
            torch.save(policy_net.state_dict(), save_path)
            print(f'Solved! Model saved to {save_path}')

def moving_average(data: Sequence[float], n: int=100):
    """
    Compute the moving average of a 1D sequence.

    Args:
        data: A sequence of numeric values.
        window_size: The number of elements over which to average/convolve.

    Returns:
        A numpy array of size len(data) - window_size + 1 containing the moving average values.
    """
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

smoothed = moving_average(episode_rewards, window)

plt.subplot(1,2,2)
plt.plot(smoothed, label=f"{window}-episode MA")
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.title("Smoothed Reward Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()