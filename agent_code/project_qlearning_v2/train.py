import os
import events as e
from .features import state_to_features_qlearning_v2
from .model import QTableAgent
from .utils import (
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
    Initializes training variables, model instance, and active reward profile.
    """
    if hasattr(self, 'logger') and self.logger:
        self.logger.info("Initializing Q-Learning v2 training setup...")
    
    self.reward_profile_name = getattr(self, 'reward_profile_name', 'profile_combative_survival')
    self.rewards = REWARD_PROFILES.get(self.reward_profile_name, REWARD_PROFILES['profile_combative_survival'])

    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pkl')

    if not hasattr(self, 'model') or self.model is None:
        self.model = QTableAgent()

    if self.model.load(self.model_filepath):
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Loaded existing Q-table from {self.model_filepath}")

    self.round_rewards = []
    self.total_rewards = 0.0


def compute_true_target_dist(game_state):
    """
    Computes shortest true BFS distance to the most immediate objective.
    Priority: Coins -> Crates -> Opponents.
    """
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

    # 1. Coins
    if coins:
        _, dist = bfs_pathfinding(field, (x, y), coins, free_mask)
        if dist != -1:
            return dist, 'coin'

    # 2. Crates
    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    if crate_tiles:
        _, dist = bfs_pathfinding(field, (x, y), crate_tiles, free_mask)
        if dist != -1:
            return dist, 'crate'

    # 3. Opponents
    if other_xys:
        _, dist = bfs_pathfinding(field, (x, y), other_xys, free_mask)
        if dist != -1:
            return dist, 'opponent'

    return -1, 'none'


def reward_from_events(self, old_game_state, self_action, new_game_state, events):
    """
    Computes shaped scalar reward using true BFS potential differences and domain safety events.
    """
    reward_sum = 0.0

    if old_game_state is not None and new_game_state is not None:
        old_danger = get_bomb_danger_map(old_game_state['field'], old_game_state['bombs'], old_game_state['explosion_map'])
        new_danger = get_bomb_danger_map(new_game_state['field'], new_game_state['bombs'], new_game_state['explosion_map'])
        
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        # 1. True Distance Progress
        old_dist, old_type = compute_true_target_dist(old_game_state)
        new_dist, new_type = compute_true_target_dist(new_game_state)

        if old_dist != -1 and new_dist != -1 and old_type == new_type:
            if new_dist < old_dist:
                events.append(CLOSER_TO_TARGET)
            elif new_dist > old_dist:
                events.append(AWAY_FROM_TARGET)

        # 2. Danger transitions
        if old_danger[old_pos] != 99 and new_danger[new_pos] == 99:
            events.append(ESCAPED_DANGER)
        elif old_danger[old_pos] == 99 and new_danger[new_pos] != 99:
            events.append(ENTERED_DANGER)

        # 3. Bomb action assessment
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

    # Sum matching event rewards
    for event in events:
        if event in self.rewards:
            reward_sum += self.rewards[event]

    return reward_sum


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Step callback: Updates Q-table on each environment step.
    """
    if old_game_state is None:
        return

    old_state = state_to_features_qlearning_v2(old_game_state)
    new_state = state_to_features_qlearning_v2(new_game_state) if new_game_state is not None else None

    reward = reward_from_events(self, old_game_state, self_action, new_game_state, events)
    self.total_rewards += reward

    self.model.update(old_state, self_action, reward, new_state, done=False)


def end_of_round(self, last_game_state, last_action, events):
    """
    Round terminal callback: Performs final transition update, decays epsilon, saves checkpoint.
    """
    last_state = state_to_features_qlearning_v2(last_game_state)
    reward = reward_from_events(self, last_game_state, last_action, None, events)
    self.total_rewards += reward

    self.model.update(last_state, last_action, reward, None, done=True)
    self.model.decay_epsilon()
    self.model.save(self.model_filepath)

    if hasattr(self, 'logger') and self.logger:
        self.logger.info(f"Q-learning v2 Round Finished. Reward: {self.total_rewards:.2f}, Epsilon: {self.model.epsilon:.4f}")
    self.total_rewards = 0.0
