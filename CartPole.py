import gym
import numpy as np
from discretizer import Discretizer

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