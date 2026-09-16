import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer for Double DQN v2.
    Stores transitions (s, a, r, s', done) and samples mini-batches.
    """
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device="cpu"):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)
