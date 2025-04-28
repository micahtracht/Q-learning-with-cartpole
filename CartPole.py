import numpy as np
if not hasattr(np, 'bool8'): # fix annoying error with gym, this took me 30 minutes. Honestly could have been worse.
    np.bool8 = np.bool_
import gym
from Discretizer import Discretizer
import matplotlib.pyplot as plt

# Create the environment using gym.make. This defines the place where the agent will trian, including:
# the reward signal, possible actions, the state, and more.
env = gym.make("CartPole-v1")

n_bins = 10
# goes in cart pos, cart vel, pole angle, pole angular vel
lower_bounds = [-4.8, -5, -0.418, -5]
upper_bounds = [4.8, 5, 0.418, 5]

# Create the state space, which is 10^4 x 2 = 20,000 states (n_bins^dim * num_actions)
Q = np.zeros([n_bins] * 4 + [env.action_space.n])

# make discretizer
disc = Discretizer(bins_per_feature=n_bins, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

env.action_space.n = 2
alpha = 0.1 # learning rate, how much do we update Q(S,A) each time
gamma = 0.99 # discount factor for future rewards, if gamma = 1 we have instability in stochastic environments, but gamma should be near 1 because we care about future rewards too
epsilon = 1.0 # the chance we take a random action to explore (exploration vs exploitation)
epsilon_min = 0.01 # the min value for epsilon (can't go below this)
epsilon_decay = 0.999 # what to multiply epsilon by after each simulation
episodes = 10000 # number of simulations to run where it will learn
max_steps = 500 # max length of simulation, if we haven't failed by t=500 we've 'solved' it.

rewards = []

for episode in range(episodes):
    obs, _ = env.reset()
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
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    if episode % 1000 == 0:
        print(f'Episode {episode}, total reward: {total_reward}, Epsilon: {epsilon:.3f}')

def moving_average(data, window_size = 100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
smoothed_rewards = moving_average(rewards)
plt.plot(smoothed_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Q-learning cartpole reward graph')
plt.grid()
plt.show()


print(f'average rewards: {sum(rewards)/len(rewards)}')