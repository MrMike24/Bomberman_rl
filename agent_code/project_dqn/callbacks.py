import os
import torch
import numpy as np
from .features import state_to_features_dqn
from .model import DQNAgent
from .utils import ACTIONS, check_safe_moves, can_escape_bomb, get_bomb_danger_map

IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def setup(self):
    """
    Setup callback called once before games start.
    Loads the trained DQN model parameters.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pt')

    # Construct dummy game_state to dynamically infer feature dimension
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

    self.model = DQNAgent(input_dim=feat_dim)

    if os.path.exists(self.model_filepath):
        loaded = self.model.load(self.model_filepath)
        if loaded and self.logger:
            self.logger.info(f"DQN model loaded successfully from {self.model_filepath}")
    else:
        if self.logger:
            self.logger.info("No saved DQN model found, starting with freshly initialized weights.")


def act(self, game_state: dict) -> str:
    """
    Called each step to choose an action.
    """
    if game_state is None:
        return 'WAIT'

    state_features = state_to_features_dqn(game_state)

    # Compute safe action indices
    danger_map = get_bomb_danger_map(game_state['field'], game_state['bombs'], game_state['explosion_map'])
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    bombs_left = game_state['self'][2]

    safe_action_indices = [ACTION_TO_IDX[a] for a in ACTIONS if a in safe_moves_dict and safe_moves_dict[a]]
    if bombs_left and can_escape_bomb(game_state):
        safe_action_indices.append(ACTION_TO_IDX['BOMB'])

    action_idx = self.model.select_action(state_features, safe_action_indices=safe_action_indices, is_training=self.train)
    action = IDX_TO_ACTION.get(action_idx, 'WAIT')

    return action
