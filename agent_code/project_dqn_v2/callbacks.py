import os
import numpy as np
from .features import state_to_features_dqn_v2
from .model import DoubleDQNAgent
from .utils import ACTIONS, check_safe_moves, can_escape_bomb, get_bomb_danger_map

IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def setup(self):
    """
    Tournament setup callback called once before rounds begin.
    Loads trained weights into the Double DQN architecture.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pt')

    # Dynamically determine feature dimension from dummy state
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

    self.model = DoubleDQNAgent(input_dim=feat_dim)

    if os.path.exists(self.model_filepath):
        loaded = self.model.load(self.model_filepath)
        if loaded and hasattr(self, 'logger') and self.logger:
            self.logger.info(f"DQN v2 model loaded successfully from {self.model_filepath}")
    else:
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("No saved DQN v2 checkpoint found, initializing fresh weights.")


def act(self, game_state: dict) -> str:
    """
    Per-step action selection callback.
    """
    if game_state is None:
        return 'WAIT'

    state_features = state_to_features_dqn_v2(game_state)

    # Compute verified safe actions
    danger_map = get_bomb_danger_map(game_state['field'], game_state['bombs'], game_state['explosion_map'])
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    bombs_left = game_state['self'][2]

    safe_action_indices = [ACTION_TO_IDX[a] for a in ACTIONS if a in safe_moves_dict and safe_moves_dict[a]]
    if bombs_left and can_escape_bomb(game_state):
        safe_action_indices.append(ACTION_TO_IDX['BOMB'])

    action_idx = self.model.select_action(state_features, safe_action_indices=safe_action_indices, is_training=self.train)
    action = IDX_TO_ACTION.get(action_idx, 'WAIT')

    return action
