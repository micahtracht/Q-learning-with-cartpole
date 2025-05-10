import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        '''
        Initializes the replay buffer.
        
        Args:
            Capacity (int): Max number of transitions stored in buffer.
        '''
        
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Stores a single transition in the buffer.

        Args:
            state (array): The current state
            action (int): The action taken
            reward (float): The reward received
            next_state (array): The next state after action
            done (bool): Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Samples a random batch of transitions.
        Args:
            batch_size (int): Number of batches to sample.
        Returns:
            A tuple of lists: batches of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(list, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)