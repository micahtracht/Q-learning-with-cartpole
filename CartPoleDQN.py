import random
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from dqn_agent import DQN
from replay_buffer import ReplayBuffer
from config import cfg, Config
from utils import moving_average, decay

# - Globals & Helpers -
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# - Main training loop -
def run_one_experiment(cfg: Config, seed: int = cfg.env.seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    epsilon = cfg.dqn.epsilon
    alpha = cfg.dqn.alpha
    # Set up environment
    env = gym.make(cfg.env.env_id)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
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
    replay_buffer = ReplayBuffer(capacity=cfg.dqn.buffer_size)
    
    episode_solved_at = -1 # when the agent solves it
    
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
            
            if len(replay_buffer) >= cfg.dqn.batch_size and len(replay_buffer) >= cfg.dqn.warmup:
                states, actions, rewards, next_states, dones = replay_buffer.sample(cfg.dqn.batch_size)
            
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
                if len(replay_buffer) > cfg.dqn.warmup:
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
        if episode % cfg.dqn.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(total_reward)
        
        # Logging & saving
        if episode % 10 == 0:
            recent = episode_rewards[-cfg.dqn.window:]
            recent_avg = sum(recent) / len(recent)
            print(f"Ep {episode}  R= {total_reward:.0f}, eps = {epsilon:.3f}  alpha = {alpha:.5f} recent avg = {recent_avg:.1f}")
            if not solved and recent_avg >= cfg.dqn.max_steps:
                solved = True
                torch.save(policy_net.state_dict(), cfg.dqn.save_path)
                print(f'Solved! Model saved to {cfg.dqn.save_path}')
                episode_solved_at = episode
    return episode_rewards, episode_solved_at

def plot_average_rewards(all_rewards_lists: List[List[float]], num_runs: int, window_dqn: int = 100) -> None:
    if not all_rewards_lists:
        print('Nothing to plot.')
        return 
    
    min_len = min(len(r) for r in all_rewards_lists)
    processed = np.array([r[:min_len] for r in all_rewards_lists])
    

    ma_intermediate_results = []
    for run_data in processed:
        current_ma = moving_average(run_data, window_dqn)
        ma_intermediate_results.append(current_ma)
    
    smoothed = np.array(ma_intermediate_results)
    
    min_smoothed = min(len(s) for s in smoothed)
    final_smoothed = np.array([sr[:min_smoothed] for sr in smoothed])
    
    mean_smoothed = np.mean(final_smoothed, axis=0)
    std_smoothed = np.std(final_smoothed, axis=0)
    
    episodes_x = np.arange(min_smoothed)
    
    plt.plot(episodes_x, mean_smoothed, label=f"Mean Reward ({window_dqn}-ep MA)")
    plt.fill_between(episodes_x, mean_smoothed - std_smoothed, mean_smoothed + std_smoothed, alpha=0.3, label="Std Dev")
    
    plt.xlabel(f"Episode (Window: {window_dqn})")
    plt.ylabel("Average Reward")
    plt.title(f"DQN Smoothed Reward Curve (Avg over {num_runs} runs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rewards(episode_rewards: int, window_dqn: int) -> None:
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
    num_runs = 5
    seeds = [int(10000 * np.random.rand()) for i in range(num_runs)]
    
    all_episodes_rewards = []
    when_solved = []
    avg_run_rewards_list = []
    
    for i in range(num_runs):
        curr_run_seed = seeds[i]
        print(f"\nStarting Run {i+1}/{num_runs} with Seed: {curr_run_seed}")
        rewards_single_run, solve_time = run_one_experiment(cfg, curr_run_seed)
        all_episodes_rewards.append(rewards_single_run)
        if solve_time != -1:
            when_solved.append(solve_time)
        
        final_avg_reward = np.mean(rewards_single_run)
        avg_run_rewards_list.append(final_avg_reward)
    
    if len(when_solved) > 0:
        mean_solve_time = np.mean(when_solved)
        std_solve_time = np.std(when_solved)
        print(f"Episodes to solve (for solved runs): Mean={mean_solve_time:.2f}, StdDev={std_solve_time:.2f}")
        print(f"Solved episodes list: {when_solved}")
    else:
        print('No runs solved.')
    
    if avg_run_rewards_list:
        mean_final_reward = np.mean(avg_run_rewards_list)
        std_final_reward = np.std(avg_run_rewards_list)
        print(f"Average rewards across runs: Mean={mean_final_reward:.2f}, StdDev={std_final_reward:.2f}") #
        print(f"Individual run average rewards: {[float(f'{r:.2f}') for r in avg_run_rewards_list]}")
    
    print('all completed')
    plot_average_rewards(all_episodes_rewards, num_runs, cfg.dqn.window)
    
    