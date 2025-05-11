import random
from collections import deque
from typing import Deque, Tuple, List, Sequence
class ReplayBuffer:
    buffer: Deque[Tuple[Sequence[float], int, float, Sequence[float], bool]]
    def __init__(self, capacity: int) -> None:
        '''
        Initializes the replay buffer.
        
        Args:
            Capacity (int): Max number of transitions stored in buffer.
        '''
        
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: Sequence[float], action: int, reward: float, next_state: Sequence[float], done: bool) -> None:
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
    
    def sample(self, batch_size: int) -> Tuple[List[Sequence[float]], List[int], List[float], List[Sequence[float]], List[bool]]:
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

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)