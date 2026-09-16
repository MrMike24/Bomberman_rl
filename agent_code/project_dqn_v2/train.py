import os
import torch
import numpy as np
import events as e
from .features import state_to_features_dqn_v2
from .model import DoubleDQNAgent
from .replay_buffer import ReplayBuffer
from .utils import (
    ACTIONS,
    check_safe_moves,
    can_escape_bomb,
    get_bomb_danger_map,
    count_crates_in_range,
    opponents_in_range,
    bfs_pathfinding
)

# Custom auxiliary events
CLOSER_TO_TARGET = 'CLOSER_TO_TARGET'
AWAY_FROM_TARGET = 'AWAY_FROM_TARGET'
ESCAPED_DANGER = 'ESCAPED_DANGER'
ENTERED_DANGER = 'ENTERED_DANGER'
SUICIDAL_BOMB_DROPPED = 'SUICIDAL_BOMB_DROPPED'
EFFECTIVE_BOMB_DROPPED = 'EFFECTIVE_BOMB_DROPPED'
USELESS_BOMB_DROPPED = 'USELESS_BOMB_DROPPED'

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
        e.COIN_COLLECTED: 1.5,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.CRATE_DESTROYED: 0.5,
        e.COIN_FOUND: 0.3,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.1,
        CLOSER_TO_TARGET: 0.3,
        AWAY_FROM_TARGET: -0.3,
        ESCAPED_DANGER: 0.5,
        ENTERED_DANGER: -1.0,
        SUICIDAL_BOMB_DROPPED: -3.0,
        EFFECTIVE_BOMB_DROPPED: 0.8,
    },
    'profile_combative_survival': {
        e.COIN_COLLECTED: 2.0,
        e.KILLED_OPPONENT: 8.0,
        e.KILLED_SELF: -10.0,
        e.GOT_KILLED: -6.0,
        e.CRATE_DESTROYED: 0.8,
        e.COIN_FOUND: 0.4,
        e.SURVIVED_ROUND: 2.5,
        e.INVALID_ACTION: -1.0,
        e.WAITED: -0.15,
        CLOSER_TO_TARGET: 0.35,
        AWAY_FROM_TARGET: -0.35,
        ESCAPED_DANGER: 1.0,
        ENTERED_DANGER: -1.5,
        SUICIDAL_BOMB_DROPPED: -6.0,
        EFFECTIVE_BOMB_DROPPED: 1.2,
        USELESS_BOMB_DROPPED: -0.5,
    }
}


def setup_training(self):
    """
    Initializes Double DQN training state, replay buffer, and active reward profile.
    """
    if hasattr(self, 'logger') and self.logger:
        self.logger.info("Initializing Double DQN v2 training setup...")

    self.reward_profile_name = getattr(self, 'reward_profile_name', 'profile_combative_survival')
    self.rewards = REWARD_PROFILES.get(self.reward_profile_name, REWARD_PROFILES['profile_combative_survival'])

    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pt')

    dummy_field = np.zeros((17, 17), dtype=int)
    dummy_game_state = {
        'field': dummy_field,
        'explosion_map': np.zeros((17, 17), dtype=int),
        'bombs': [],
        'coins': [],
        'self': ('agent', 0, True, (1, 1)),
        'others': [],
        'step': 0
    }
    dummy_feat = state_to_features_dqn_v2(dummy_game_state)
    feat_dim = len(dummy_feat)

    if not hasattr(self, 'model') or self.model is None:
        self.model = DoubleDQNAgent(input_dim=feat_dim)

    if self.model.load(self.model_filepath):
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Loaded existing DQN v2 model from {self.model_filepath}")

    if not hasattr(self, 'replay_buffer') or self.replay_buffer is None:
        self.replay_buffer = ReplayBuffer(capacity=50000)

    self.batch_size = getattr(self, 'batch_size', 64)
    self.step_counter = 0
    self.total_rewards = 0.0


def compute_true_target_dist(game_state):
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    coins = game_state['coins']
    _, _, _, (x, y) = game_state['self']
    others = game_state['others']
    other_xys = [xy for (_, _, _, xy) in others]

    cols, rows = field.shape
    bomb_xys = {xy for (xy, t) in bombs}
    free_mask = (field == 0) & (explosion_map == 0)
    for bxy in bomb_xys:
        free_mask[bxy] = False
    for oxy in other_xys:
        free_mask[oxy] = False

    if coins:
        _, dist = bfs_pathfinding(field, (x, y), coins, free_mask)
        if dist != -1:
            return dist, 'coin'

    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    if crate_tiles:
        _, dist = bfs_pathfinding(field, (x, y), crate_tiles, free_mask)
        if dist != -1:
            return dist, 'crate'

    if other_xys:
        _, dist = bfs_pathfinding(field, (x, y), other_xys, free_mask)
        if dist != -1:
            return dist, 'opponent'

    return -1, 'none'


def reward_from_events(self, old_game_state, self_action, new_game_state, events):
    reward_sum = 0.0

    if old_game_state is not None and new_game_state is not None:
        old_danger = get_bomb_danger_map(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
        new_danger = get_bomb_danger_map(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
        
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        # True distance progression
        old_dist, old_type = compute_true_target_dist(old_game_state)
        new_dist, new_type = compute_true_target_dist(new_game_state)

        if old_dist != -1 and new_dist != -1 and old_type == new_type:
            if new_dist < old_dist:
                events.append(CLOSER_TO_TARGET)
            elif new_dist > old_dist:
                events.append(AWAY_FROM_TARGET)

        # Danger changes
        if old_danger[old_pos] != 99 and new_danger[new_pos] == 99:
            events.append(ESCAPED_DANGER)
        elif old_danger[old_pos] == 99 and new_danger[new_pos] != 99:
            events.append(ENTERED_DANGER)

        # Bomb placement evaluation
        if self_action == 'BOMB':
            if not can_escape_bomb(old_game_state):
                events.append(SUICIDAL_BOMB_DROPPED)
            else:
                crates_hit = count_crates_in_range(old_game_state['field'], old_pos)
                opps_hit = opponents_in_range(old_pos, old_game_state['others'])
                if crates_hit > 0 or opps_hit > 0:
                    events.append(EFFECTIVE_BOMB_DROPPED)
                else:
                    events.append(USELESS_BOMB_DROPPED)

    for event in events:
        if event in self.rewards:
            reward_sum += self.rewards[event]

    return reward_sum


def train_step(self):
    if len(self.replay_buffer) < self.batch_size:
        return

    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, device=self.model.device)

    # 1. Q(s, a) from policy net
    q_eval = self.model.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # 2. Double DQN target:
    # a* = argmax_a Q_policy(s', a)
    # y = r + gamma * Q_target(s', a*) * (1 - d)
    with torch.no_grad():
        next_actions = self.model.policy_net(next_states).argmax(dim=1, keepdim=True)
        q_next_target = self.model.target_net(next_states).gather(1, next_actions).squeeze(1)
        q_target = rewards + self.model.gamma * q_next_target * (1.0 - dones)

    loss = self.model.loss_fn(q_eval, q_target)

    self.model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 1.0)
    self.model.optimizer.step()

    # Polyak soft target network update
    self.model.soft_update_target_net()


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    if old_game_state is None:
        return

    old_state = state_to_features_dqn_v2(old_game_state)
    new_state = state_to_features_dqn_v2(new_game_state) if new_game_state is not None else np.zeros_like(old_state)
    action_idx = ACTION_TO_IDX.get(self_action, 5)

    reward = reward_from_events(self, old_game_state, self_action, new_game_state, events)
    self.total_rewards += reward

    self.replay_buffer.push(old_state, action_idx, reward, new_state, done=False)
    train_step(self)
    self.step_counter += 1


def end_of_round(self, last_game_state, last_action, events):
    last_state = state_to_features_dqn_v2(last_game_state)
    dummy_next = np.zeros_like(last_state)
    action_idx = ACTION_TO_IDX.get(last_action, 5)

    reward = reward_from_events(self, last_game_state, last_action, None, events)
    self.total_rewards += reward

    self.replay_buffer.push(last_state, action_idx, reward, dummy_next, done=True)
    train_step(self)

    self.model.decay_epsilon()
    self.model.save(self.model_filepath)

    if hasattr(self, 'logger') and self.logger:
        self.logger.info(f"DQN v2 Round Finished. Total Reward: {self.total_rewards:.2f}, Epsilon: {self.model.epsilon:.4f}")
    self.total_rewards = 0.0
