import numpy as np
if not hasattr(np, 'bool8'): # fix annoying error with gym, this took me 30 minutes. Honestly could have been worse.
    np.bool8 = np.bool_
import gym
from discretizer import Discretizer
import matplotlib.pyplot as plt
from typing import Sequence

# Create the environment using gym.make. This defines the place where the agent will trian, including:
# the reward signal, possible actions, the state, and more.
env = gym.make("CartPole-v1")

n_bins = 10
n_angle_bins = 20 # refine this further because the system is very sensitive to the angle.
bin_counts = [n_bins, n_bins, n_angle_bins, n_bins]
# goes in cart pos, cart vel, pole angle, pole angular vel
lower_bounds = np.array([-4.8, -5, -0.418, -5])
upper_bounds = np.array([4.8, 5, 0.418, 5])
# Create the state space, which is 10^4 x 2 = 20,000 states (n_bins^dim * num_actions)
Q = np.zeros(bin_counts + [env.action_space.n], dtype='float')
# make discretizer
disc = Discretizer(bins_per_feature=bin_counts, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

env.action_space.n = 2
alpha, alpha_min, alpha_decay = 0.3, 0.005, 0.999 # learning rate, how much do we update Q(S,A) each time. What to mult the learning rate by each time. Min value for learning rate.
gamma = 0.99 # discount factor for future rewards, if gamma = 1 we have instability in stochastic environments, but gamma should be near 1 because we care about future rewards too
epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.999 # the chance we take a random action to explore (exploration vs exploitation). The min value for epsilon (can't go below this). What to multiply epsilon by after each simulation
episodes = 10000 # number of simulations to run where it will learn
max_steps = 500 # max length of simulation, if we haven't failed by t=500 we've 'solved' it.

rewards = []

# Clips each feature down to [low_i, high_i] to avoid indexing errors or weird rap arounds, mostly with velocities.
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

# Compute moving average to smooth data for plotting
def moving_average(data: Sequence[float], window_size: int=100) -> np.ndarray:
    """
    Computes the moving average of a 1D sequence.

    Args:
        data (Sequence[float]): A sequence of numeric values.
        window_size (int): The size of the moving window.

    Returns:
        np.ndarray: Array of the moving average, length = len(data) - window_size + 1.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for episode in range(episodes):
    obs, _ = env.reset()
    obs = clip_obs(obs, lower_bounds, upper_bounds)
    state = disc.discretize(obs)
    total_reward = 0 
    done = False
    
    for _ in range(max_steps):
        # with probability epsilon, take a random action.
        if np.random.random() < epsilon:
            action = env.action_space.sample() # get a random action
        else:
            action = np.argmax(Q[state]) # do the action that has the best expected reward
        
        # Now that we've taken an action, we apply that to the environment and observe the results
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated # either of these ends the episode
        # discretize the new state so we can process it
        next_obs = clip_obs(next_obs, lower_bounds, upper_bounds)
        next_state = disc.discretize(next_obs)
        
        # Now we use the bellman equation to update the q values
        best_future_q = np.max(Q[next_state]) # This is max a' Q(s', a')
        td_target = reward + gamma * best_future_q # this is our target value
        td_error = td_target - Q[state][action] # this is our temporal difference (reality and what we believed)
        Q[state][action] += alpha * td_error # Update Q-val
        
        state = next_state # update our sate
        total_reward += reward
        
        if done:
            break
    
    alpha = max(alpha_min, alpha * alpha_decay)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    rewards.append(total_reward)
    if episode % 1000 == 0:
        print(f'Episode {episode}, total reward: {total_reward}, Epsilon: {epsilon:.3f}, Alpha: {alpha:.3f}')

def moving_average(data: Sequence[float], window_size: int = 100) -> np.ndarray:
    """
    Compute the moving average of a 1D sequence.

    Args:
        data: A sequence of numeric values.
        window_size: The number of elements over which to average/convolve.

    Returns:
        A numpy array of size len(data) - window_size + 1 containing the moving average values.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

print(f'average rewards: {sum(rewards)/len(rewards)}')

smoothed_rewards = moving_average(rewards)
plt.plot(smoothed_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Q-learning cartpole reward graph')
plt.grid()
plt.show()

later_rewards = smoothed_rewards[:5000]
plt.plot(later_rewards)
plt.xlabel('Episodes (+5000)')
plt.ylabel('Reward')
plt.title('later rewards in cartpole')
plt.grid()
plt.show()