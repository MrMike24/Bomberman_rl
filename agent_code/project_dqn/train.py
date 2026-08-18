import os
import torch
import numpy as np
import events as e
from .features import state_to_features_dqn
from .model import DQNAgent
from .replay_buffer import ReplayBuffer
from .utils import ACTIONS, check_safe_moves, can_escape_bomb, count_crates_in_range

# Custom auxiliary events
CLOSER_TO_TARGET = 'CLOSER_TO_TARGET'
AWAY_FROM_TARGET = 'AWAY_FROM_TARGET'
SUICIDAL_BOMB_DROPPED = 'SUICIDAL_BOMB_DROPPED'
EFFECTIVE_BOMB_DROPPED = 'EFFECTIVE_BOMB_DROPPED'

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

REWARD_PROFILES = {
    'profile_sparse': {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.INVALID_ACTION: -0.1,
    },
    'profile_dense_nav': {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.2,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.1,
        CLOSER_TO_TARGET: 0.2,
        AWAY_FROM_TARGET: -0.2,
        SUICIDAL_BOMB_DROPPED: -3.0,
        EFFECTIVE_BOMB_DROPPED: 0.5,
    },
    'profile_combative_survival': {
        e.COIN_COLLECTED: 1.5,
        e.KILLED_OPPONENT: 6.0,
        e.KILLED_SELF: -8.0,
        e.GOT_KILLED: -6.0,
        e.CRATE_DESTROYED: 0.6,
        e.COIN_FOUND: 0.3,
        e.SURVIVED_ROUND: 2.0,
        e.INVALID_ACTION: -1.0,
        e.WAITED: -0.2,
        CLOSER_TO_TARGET: 0.3,
        AWAY_FROM_TARGET: -0.3,
        SUICIDAL_BOMB_DROPPED: -5.0,
        EFFECTIVE_BOMB_DROPPED: 1.0,
    }
}


def setup_training(self):
    """
    Setup DQN training variables, replay buffer, and model.
    """
    self.logger.info("Initializing DQN training setup...")
    self.reward_profile_name = getattr(self, 'reward_profile_name', 'profile_combative_survival')
    self.rewards = REWARD_PROFILES[self.reward_profile_name]

    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pt')

    dummy_field = np.zeros((17, 17), dtype=int)
    dummy_game_state = {
        'field': dummy_field,
        'explosion_map': np.zeros((17, 17), dtype=int),
        'bombs': [],
        'coins': [],
        'self': ('agent', 0, True, (1, 1)),
        'others': []
    }
    dummy_feat = state_to_features_dqn(dummy_game_state)
    feat_dim = len(dummy_feat)

    if not hasattr(self, 'model') or self.model is None:
        self.model = DQNAgent(input_dim=feat_dim)

    if self.model.load(self.model_filepath):
        self.logger.info(f"Loaded existing DQN model from {self.model_filepath}")

    if not hasattr(self, 'replay_buffer') or self.replay_buffer is None:
        self.replay_buffer = ReplayBuffer(capacity=50000)

    self.batch_size = getattr(self, 'batch_size', 64)
    self.target_update_freq = getattr(self, 'target_update_freq', 500)
    self.step_counter = 0
    self.total_rewards = 0.0


def reward_from_events(self, old_game_state, self_action, new_game_state, events):
    reward_sum = 0.0

    if old_game_state is not None and new_game_state is not None:
        old_feat = state_to_features_dqn(old_game_state)
        new_feat = state_to_features_dqn(new_game_state)

        if old_feat is not None and new_feat is not None:
            # Nearest target distance feature at index 2
            old_coin_dist = old_feat[2]
            new_coin_dist = new_feat[2]

            if new_coin_dist < old_coin_dist:
                events.append(CLOSER_TO_TARGET)
            elif new_coin_dist > old_coin_dist and old_coin_dist < 1.0:
                events.append(AWAY_FROM_TARGET)

        if self_action == 'BOMB':
            if not can_escape_bomb(old_game_state):
                events.append(SUICIDAL_BOMB_DROPPED)
            else:
                crates_in_range = count_crates_in_range(old_game_state['field'], old_game_state['self'][3])
                if crates_in_range > 0:
                    events.append(EFFECTIVE_BOMB_DROPPED)

    for event in events:
        if event in self.rewards:
            reward_sum += self.rewards[event]

    return reward_sum


def train_step(self):
    if len(self.replay_buffer) < self.batch_size:
        return

    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

    # Compute Q(s, a)
    q_eval = self.model.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target y = r + gamma * max Q_target(s')
    with torch.no_grad():
        q_next = self.model.target_net(next_states).max(1)[0]
        q_target = rewards + self.model.gamma * q_next * (1.0 - dones)

    loss = self.model.loss_fn(q_eval, q_target)

    self.model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 1.0)
    self.model.optimizer.step()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    old_state = state_to_features_dqn(old_game_state)
    new_state = state_to_features_dqn(new_game_state) if new_game_state is not None else np.zeros_like(old_state)
    action_idx = ACTION_TO_IDX.get(self_action, 5)

    reward = reward_from_events(self, old_game_state, self_action, new_game_state, events)
    self.total_rewards += reward

    self.replay_buffer.push(old_state, action_idx, reward, new_state, done=False)

    train_step(self)

    self.step_counter += 1
    if self.step_counter % self.target_update_freq == 0:
        self.model.update_target_net()


def end_of_round(self, last_game_state, last_action, events):
    last_state = state_to_features_dqn(last_game_state)
    dummy_next = np.zeros_like(last_state)
    action_idx = ACTION_TO_IDX.get(last_action, 5)

    reward = reward_from_events(self, last_game_state, last_action, None, events)
    self.total_rewards += reward

    self.replay_buffer.push(last_state, action_idx, reward, dummy_next, done=True)
    train_step(self)

    self.model.decay_epsilon()
    self.model.save(self.model_filepath)
    self.logger.info(f"DQN Round finished. Total Reward: {self.total_rewards:.2f}, Epsilon: {self.model.epsilon:.4f}")
    self.total_rewards = 0.0
