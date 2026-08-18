import os
import events as e
from .features import state_to_features_qlearning
from .model import QTableAgent
from .utils import check_safe_moves, can_escape_bomb, get_bomb_danger_map, count_crates_in_range

# Custom auxiliary events
CLOSER_TO_TARGET = 'CLOSER_TO_TARGET'
AWAY_FROM_TARGET = 'AWAY_FROM_TARGET'
SUICIDAL_BOMB_DROPPED = 'SUICIDAL_BOMB_DROPPED'
EFFECTIVE_BOMB_DROPPED = 'EFFECTIVE_BOMB_DROPPED'

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
    Initializes training variables, model instance, and active reward profile.
    """
    self.logger.info("Initializing Q-Learning training setup...")
    
    # Active reward profile (can be configured via environment or self attribute)
    self.reward_profile_name = getattr(self, 'reward_profile_name', 'profile_combative_survival')
    self.rewards = REWARD_PROFILES[self.reward_profile_name]

    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pkl')

    if not hasattr(self, 'model') or self.model is None:
        self.model = QTableAgent()

    # Load existing weights if available
    if self.model.load(self.model_filepath):
        self.logger.info(f"Loaded existing Q-table from {self.model_filepath}")
    else:
        self.logger.info("Created new Q-table model")

    self.round_rewards = []
    self.total_rewards = 0


def reward_from_events(self, old_game_state, self_action, new_game_state, events):
    """
    Computes total reward by combining standard environment events with custom shaped auxiliary events.
    """
    reward_sum = 0.0

    # Custom event detection
    if old_game_state is not None and new_game_state is not None:
        old_feat = state_to_features_qlearning(old_game_state)
        new_feat = state_to_features_qlearning(new_game_state)

        if old_feat is not None and new_feat is not None:
            # 1. Target distance progress (Coin / Crate / Opponent)
            old_target_dist = min([d for d in [old_feat[0], old_feat[1], old_feat[2]] if d > 0] or [99])
            new_target_dist = min([d for d in [new_feat[0], new_feat[1], new_feat[2]] if d > 0] or [99])

            if new_target_dist < old_target_dist:
                events.append(CLOSER_TO_TARGET)
            elif new_target_dist > old_target_dist and old_target_dist != 99:
                events.append(AWAY_FROM_TARGET)

        # 2. Bomb placement check
        if self_action == 'BOMB':
            if not can_escape_bomb(old_game_state):
                events.append(SUICIDAL_BOMB_DROPPED)
            else:
                crates_in_range = count_crates_in_range(old_game_state['field'], old_game_state['self'][3])
                if crates_in_range > 0:
                    events.append(EFFECTIVE_BOMB_DROPPED)

    # Accumulate rewards based on profile
    for event in events:
        if event in self.rewards:
            reward_sum += self.rewards[event]

    return reward_sum


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Callback after each game step during training.
    Executes Q-learning update.
    """
    if old_game_state is None:
        return

    old_state = state_to_features_qlearning(old_game_state)
    new_state = state_to_features_qlearning(new_game_state) if new_game_state is not None else None

    reward = reward_from_events(self, old_game_state, self_action, new_game_state, events)
    self.total_rewards += reward

    # Perform Q-learning update
    self.model.update(old_state, self_action, reward, new_state, done=False)


def end_of_round(self, last_game_state, last_action, events):
    """
    Callback after final step of round.
    Executes final transition update, decays epsilon, and saves model.
    """
    last_state = state_to_features_qlearning(last_game_state)
    reward = reward_from_events(self, last_game_state, last_action, None, events)
    self.total_rewards += reward

    self.model.update(last_state, last_action, reward, None, done=True)
    self.model.decay_epsilon()

    # Save model periodically
    self.model.save(self.model_filepath)
    self.logger.info(f"Round finished. Total Reward: {self.total_rewards:.2f}, Epsilon: {self.model.epsilon:.4f}")
    self.total_rewards = 0
