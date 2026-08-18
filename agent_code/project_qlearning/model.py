import os
import pickle
import numpy as np
from .utils import ACTIONS, check_safe_moves, can_escape_bomb, get_bomb_danger_map


class QTableAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table represented as a dictionary mapping state tuple -> dict of action values
        self.q_table = {}

    def get_q_values(self, state):
        if state not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[state] = {a: np.random.uniform(-0.01, 0.01) for a in ACTIONS}
        return self.q_table[state]

    def choose_action(self, state, game_state=None, is_training=True):
        q_vals = self.get_q_values(state)

        # Safety checking helper
        safe_actions = None
        if game_state is not None:
            danger_map = get_bomb_danger_map(game_state['field'], game_state['bombs'], game_state['explosion_map'])
            safe_moves_dict = check_safe_moves(game_state, danger_map)
            bombs_left = game_state['self'][2]
            
            safe_actions = [a for a in ACTIONS if a in safe_moves_dict and safe_moves_dict[a]]
            if bombs_left and can_escape_bomb(game_state):
                safe_actions.append('BOMB')

        if is_training and np.random.rand() < self.epsilon:
            # Epsilon-greedy exploration: pick random safe action if available, else any action
            if safe_actions and len(safe_actions) > 0:
                return np.random.choice(safe_actions)
            return np.random.choice(ACTIONS)

        # Greedy choice: select action with max Q-value (preferring safe actions)
        if safe_actions and len(safe_actions) > 0:
            best_action = max(safe_actions, key=lambda a: q_vals[a])
        else:
            best_action = max(ACTIONS, key=lambda a: q_vals[a])

        return best_action

    def update(self, state, action, reward, next_state, done):
        q_vals = self.get_q_values(state)
        
        if done or next_state is None:
            target = reward
        else:
            next_q_vals = self.get_q_values(next_state)
            max_next_q = max(next_q_vals.values())
            target = reward + self.gamma * max_next_q

        q_vals[action] += self.alpha * (target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'gamma': self.gamma
            }, f)

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.epsilon)
                self.alpha = data.get('alpha', self.alpha)
                self.gamma = data.get('gamma', self.gamma)
            return True
        return False
