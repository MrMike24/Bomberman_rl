import numpy as np
from .utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    check_safe_moves,
    can_escape_bomb,
    count_crates_in_range,
    ACTION_TO_DELTA
)


def state_to_features_qlearning(game_state):
    """
    Transforms raw game_state dictionary into a compact discrete state tuple for Tabular Q-Learning.
    
    Tuple components:
    1. dir_coin: Direction to nearest coin (0: NONE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
    2. dir_crate: Direction to nearest crate (0: NONE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
    3. dir_opponent: Direction to nearest opponent (0: NONE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
    4. dir_safety: Direction to nearest safe tile if in danger (0: SAFE/NONE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT)
    5. safe_moves: Tuple of 4 booleans for (UP, DOWN, LEFT, RIGHT)
    6. bomb_state: Discrete integer (0: No bomb, 1: Bomb ready but unsafe escape, 2: Bomb ready & safe & target nearby, 3: Bomb ready & safe & no target)
    """
    if game_state is None:
        return None

    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = game_state['bombs']
    coins = game_state['coins']
    _, score, bombs_left, (x, y) = game_state['self']
    others = game_state['others']
    other_xys = [xy for (_, _, _, xy) in others]

    cols, rows = field.shape

    # 1. Danger map
    danger_map = get_bomb_danger_map(field, bombs, explosion_map)

    # 2. Walkable mask (field == 0, no explosion, no bomb, no opponent)
    bomb_xys = {xy for (xy, t) in bombs}
    free_mask = (field == 0) & (explosion_map == 0)
    for bxy in bomb_xys:
        free_mask[bxy] = False
    for oxy in other_xys:
        free_mask[oxy] = False

    # 3. Pathfinding to targets
    def action_code_from_tile(next_tile):
        if next_tile is None:
            return 0
        dx = next_tile[0] - x
        dy = next_tile[1] - y
        if (dx, dy) == (0, -1): return 1  # UP
        if (dx, dy) == (0, 1):  return 2  # DOWN
        if (dx, dy) == (-1, 0): return 3  # LEFT
        if (dx, dy) == (1, 0):  return 4  # RIGHT
        return 0

    # Nearest Coin
    coin_next, _ = bfs_pathfinding(field, (x, y), coins, free_mask)
    dir_coin = action_code_from_tile(coin_next)

    # Nearest Crate
    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    crate_next, _ = bfs_pathfinding(field, (x, y), crate_tiles, free_mask)
    dir_crate = action_code_from_tile(crate_next)

    # Nearest Opponent
    opp_next, _ = bfs_pathfinding(field, (x, y), other_xys, free_mask)
    dir_opp = action_code_from_tile(opp_next)

    # Direction to Safety (if danger_map[x,y] != 99)
    dir_safety = 0
    if danger_map[x, y] != 99:
        safe_tiles = [(sx, sy) for sx in range(1, cols - 1) for sy in range(1, rows - 1) if danger_map[sx, sy] == 99 and free_mask[sx, sy]]
        safe_next, _ = bfs_pathfinding(field, (x, y), safe_tiles, free_mask)
        dir_safety = action_code_from_tile(safe_next)

    # 4. Neighboring safety (UP, DOWN, LEFT, RIGHT)
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    safe_moves = (
        safe_moves_dict.get('UP', False),
        safe_moves_dict.get('DOWN', False),
        safe_moves_dict.get('LEFT', False),
        safe_moves_dict.get('RIGHT', False)
    )

    # 5. Bomb state
    crates_in_range = count_crates_in_range(field, (x, y))
    opp_in_range = any(abs(ox - x) + abs(oy - y) <= 3 for (ox, oy) in other_xys)
    has_target = (crates_in_range > 0 or opp_in_range)

    if not bombs_left:
        bomb_state = 0
    else:
        can_escape = can_escape_bomb(game_state, (x, y))
        if not can_escape:
            bomb_state = 1
        elif has_target:
            bomb_state = 2
        else:
            bomb_state = 3

    return (dir_coin, dir_crate, dir_opp, dir_safety, safe_moves, bomb_state)
