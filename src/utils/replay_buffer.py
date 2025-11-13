import os
from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions.
    """

    def __init__(self, capacity=1000000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Random sampling
        batch = random.sample(self.buffer, batch_size)

        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size
