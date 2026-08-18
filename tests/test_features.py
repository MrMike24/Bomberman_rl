import numpy as np
from agent_code.project_qlearning.utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    check_safe_moves,
    can_escape_bomb,
    count_crates_in_range
)
from agent_code.project_qlearning.features import state_to_features_qlearning
from agent_code.project_dqn.features import state_to_features_dqn


def create_dummy_game_state():
    field = np.zeros((17, 17), dtype=int)
    # Add stone walls
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    
    # Add some inner stone walls
    for x in range(2, 16, 2):
        for y in range(2, 16, 2):
            field[x, y] = -1

    # Add a crate at (2, 1)
    field[2, 1] = 1

    explosion_map = np.zeros((17, 17), dtype=int)
    bombs = [((1, 3), 3)]  # Bomb at (1,3) exploding in 3 steps
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

    # Bomb is at (1,3) with timer 3
    assert danger_map[1, 3] == 3
    # Tile (1,1) is in blast ray of (1,3) (distance 2)
    assert danger_map[1, 1] == 3
    # Tile (15,15) should be safe (99)
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
    # At (1,1), agent drops bomb. Can it escape to safe ground?
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
