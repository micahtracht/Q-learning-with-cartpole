# Q-Learning on CartPole-v1

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)

This project implements **tabular Q-learning** to solve the classic control problem [CartPole-v1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) using **state discretization**, epsilon-greedy exploration, & Bellman updates.

Despite the continuous state space, Q-learning is made effective by discretizing the observation features into fixed bins and learning Q-values per discrete state–action pair.

10 bins are selected for the cart position, cart velocity, and pole velocity features, while 20 bins are selected for the pole angle feature since it is the most sensitive to small changes.

---

## Goal

Balance a pole vertically on a moving cart for as long as possible.  
Each timestep the pole remains upright earns a reward of `+1`.  
Episodes terminate when:
- The pole angle exceeds `+/- 12°`.
- The cart moves too far from center
- The episode reaches 500 steps (At which point the agent has 'solved' the episode)

---

## Key Concepts

**Q-Learning**      : Off-policy reinforcement learning algorithm that updates a Q-table using TD learning.
**Discretization**  : Continuous states (cart position, velocity, etc.) are mapped to discrete bins so Q-values can be indexed in a table. This is done using the discretizer class.
**ε-Greedy Policy** : With probability ε, take a random action. With probability 1−ε, take the best known action. 
**Bellman Update**  : Core learning rule: `Q[s][a] ← Q[s][a] + α (r + γ·max Q[s'] - Q[s][a])` (This is the Bellman equation)

---

## Hyperparameters

```python

alpha, alpha_min, alpha_decay = 0.3, 0.005, 0.999 # learning rate, how much do we update Q(S,A) each time. What to mult the learning rate by each time. Min value for learning rate.
gamma = 0.99 # discount factor for future rewards, if gamma = 1 we have instability in stochastic environments, but gamma should be near 1 because we care about future rewards too
epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.999 # the chance we take a random action to explore (exploration vs exploitation). The min value for epsilon (can't go below this). What to multiply epsilon by after each simulation
episodes = 10000 # number of simulations to run where it will learn
max_steps = 500 # max length of simulation, if we haven't failed by t=500 we've 'solved' it.

```

## Results

After ~2,500 - 4,500 episodes, the agent consistently achieves average rewards near 500 (at least 495).

TBD: ADD REWARD CURVE IMAGE

## Files

CartPole.py: Main training loop w/ Q-learning
discretizer.py: Discretizer class that maps the continuous space to discrete subspaces.
requirements.txt: The requirements needed to run it.
README.md: You're in the readme right now! This is the project description.

## How to run

```bash
pip install -r requirements.txt
python CartPole.py
```

## Coming soon
 - Deep Q-Network to use continuous variables.
 - More environments.
 - Add early stopping.


## Credits

Built by Micah Tracht. Inspired by OpenAI Gym's classic control suite (thanks the Gym library, I used it a lot, and it was really easy to learn!) and Sutton & Barto's RL textbook (Again, thanks, it's been an amazing resource for me!)
