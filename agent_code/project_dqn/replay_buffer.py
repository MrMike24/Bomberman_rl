import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        states_t = torch.tensor(np.array(state), dtype=torch.float32)
        actions_t = torch.tensor(action, dtype=torch.int64)
        rewards_t = torch.tensor(reward, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_state), dtype=torch.float32)
        dones_t = torch.tensor(done, dtype=torch.float32)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)
