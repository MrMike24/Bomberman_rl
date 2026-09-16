import numpy as np
import torch
from agent_code.project_qlearning.utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    can_escape_bomb
)
from agent_code.project_qlearning.features import state_to_features_qlearning
from agent_code.project_dqn.features import state_to_features_dqn

from agent_code.project_qlearning_v2.features import state_to_features_qlearning_v2
from agent_code.project_dqn_v2.features import state_to_features_dqn_v2
from agent_code.project_dqn_v2.model import DoubleDQNAgent

import agent_code.project_qlearning.callbacks as q1_cb
import agent_code.project_qlearning_v2.callbacks as q2_cb
import agent_code.project_dqn.callbacks as dqn1_cb
import agent_code.project_dqn_v2.callbacks as dqn2_cb


class DummySelf:
    def __init__(self, train=False):
        self.train = train
        self.logger = None


def create_dummy_game_state():
    field = np.zeros((17, 17), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    
    for x in range(2, 16, 2):
        for y in range(2, 16, 2):
            field[x, y] = -1

    field[2, 1] = 1

    explosion_map = np.zeros((17, 17), dtype=int)
    bombs = [((1, 3), 3)]
    coins = [(3, 1), (1, 5)]

    game_state = {
        'round': 1,
        'step': 10,
        'field': field,
        'explosion_map': explosion_map,
        'bombs': bombs,
        'coins': coins,
        'self': ('agent_1', 0, True, (1, 1)),
        'others': [('agent_2', 0, True, (15, 15))],
        'user_input': None
    }
    return game_state


def test_bomb_danger_map():
    game_state = create_dummy_game_state()
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    danger_map = get_bomb_danger_map(field, bombs, explosion_map)
    assert danger_map[1, 3] == 3
    assert danger_map[1, 1] == 3
    assert danger_map[15, 15] == 99


def test_bfs_pathfinding():
    game_state = create_dummy_game_state()
    field = game_state['field']
    free_mask = (field == 0)

    next_tile, dist = bfs_pathfinding(field, (1, 1), [(3, 1)], free_mask)
    assert next_tile is not None
    assert dist > 0


def test_can_escape_bomb():
    game_state = create_dummy_game_state()
    can_escape = can_escape_bomb(game_state, (1, 1))
    assert isinstance(can_escape, bool)


def test_qlearning_feature_extraction():
    game_state = create_dummy_game_state()
    feat = state_to_features_qlearning(game_state)
    assert len(feat) == 6
    dir_coin, dir_crate, dir_opp, dir_safety, safe_moves, bomb_state = feat
    assert len(safe_moves) == 4
    assert 0 <= bomb_state <= 3


def test_dqn_feature_extraction():
    game_state = create_dummy_game_state()
    feat = state_to_features_dqn(game_state)
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.ndim == 1
    assert len(feat) >= 25


def test_qlearning_v2_feature_extraction():
    game_state = create_dummy_game_state()
    feat = state_to_features_qlearning_v2(game_state)
    assert isinstance(feat, tuple)
    assert len(feat) == 8
    dir_target, dist_bucket, target_type, danger_status, dir_safety, safe_moves, bomb_feas, in_dead = feat
    assert 0 <= dir_target <= 4
    assert 0 <= dist_bucket <= 4
    assert 0 <= target_type <= 3
    assert 0 <= danger_status <= 2
    assert 0 <= dir_safety <= 4
    assert len(safe_moves) == 4
    assert 0 <= bomb_feas <= 4
    assert in_dead in (0, 1)


def test_dqn_v2_feature_extraction():
    game_state = create_dummy_game_state()
    feat = state_to_features_dqn_v2(game_state)
    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert feat.shape == (38,)
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()


def test_dqn_v2_double_dqn_step():
    agent = DoubleDQNAgent(input_dim=38, num_actions=6)
    dummy_state = np.random.randn(38).astype(np.float32)
    action = agent.select_action(dummy_state, safe_action_indices=[0, 1, 2], is_training=False)
    assert action in [0, 1, 2]

    # Test soft update
    old_target_p = [p.clone() for p in agent.target_net.parameters()]
    agent.soft_update_target_net()
    for p_old, p_new in zip(old_target_p, agent.target_net.parameters()):
        assert p_new.shape == p_old.shape


def test_callbacks_act_all_models():
    game_state = create_dummy_game_state()
    
    for mod_name, mod in [("q1", q1_cb), ("q2", q2_cb), ("dqn1", dqn1_cb), ("dqn2", dqn2_cb)]:
        self_obj = DummySelf(train=False)
        mod.setup(self_obj)
        action = mod.act(self_obj, game_state)
        assert action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], f"{mod_name} returned invalid action {action}"
