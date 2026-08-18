import numpy as np
from .utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    check_safe_moves,
    can_escape_bomb,
    count_crates_in_range
)


def state_to_features_dqn(game_state):
    """
    Transforms raw game_state dictionary into a normalized 1D float numpy vector for PyTorch Deep Q-Network.
    
    Feature Vector Layout (~25 features):
    - [0:3]   Nearest coin relative vector (dx/17, dy/17, reachable_dist/17)
    - [3:6]   Nearest crate relative vector (dx/17, dy/17, reachable_dist/17)
    - [6:9]   Nearest opponent relative vector (dx/17, dy/17, reachable_dist/17)
    - [9:12]  Nearest safe tile relative vector if in danger (dx/17, dy/17, reachable_dist/17)
    - [12:17] 5 action safety indicators (UP, DOWN, LEFT, RIGHT, WAIT) -> 1.0 if safe else 0.0
    - [17]    Current position in danger (1.0 if danger else 0.0)
    - [18]    Bomb available (1.0 if True else 0.0)
    - [19]    Bomb drop escape safe (1.0 if safe else 0.0)
    - [20]    Crates in blast range (count / 4.0)
    - [21]    Opponents in blast range (count / 3.0)
    - [22:31] 3x3 local grid surrounding agent (-1: wall, 0: free, 1: crate, 2: danger/bomb)
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
    features = []

    # 1. Danger map
    danger_map = get_bomb_danger_map(field, bombs, explosion_map)

    # 2. Walkable mask
    bomb_xys = {xy for (xy, t) in bombs}
    free_mask = (field == 0) & (explosion_map == 0)
    for bxy in bomb_xys:
        free_mask[bxy] = False
    for oxy in other_xys:
        free_mask[oxy] = False

    # Target pathfinding helper
    def get_target_vec(targets):
        if not targets:
            return [0.0, 0.0, 1.0]
        next_tile, dist = bfs_pathfinding(field, (x, y), targets, free_mask)
        if next_tile is None:
            return [0.0, 0.0, 1.0]
        dx = (next_tile[0] - x) / 17.0
        dy = (next_tile[1] - y) / 17.0
        d_norm = dist / 17.0
        return [dx, dy, d_norm]

    # Nearest Coin, Crate, Opponent
    features.extend(get_target_vec(coins))

    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    features.extend(get_target_vec(crate_tiles))

    features.extend(get_target_vec(other_xys))

    # Nearest Safe Tile (if in danger)
    if danger_map[x, y] != 99:
        safe_tiles = [(sx, sy) for sx in range(1, cols - 1) for sy in range(1, rows - 1) if danger_map[sx, sy] == 99 and free_mask[sx, sy]]
        features.extend(get_target_vec(safe_tiles))
    else:
        features.extend([0.0, 0.0, 0.0])

    # 5 Action Safety Indicators
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    for act in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']:
        features.append(1.0 if safe_moves_dict.get(act, False) else 0.0)

    # In danger boolean
    features.append(1.0 if danger_map[x, y] != 99 else 0.0)

    # Bomb status & escape safety
    features.append(1.0 if bombs_left else 0.0)
    can_escape = can_escape_bomb(game_state, (x, y)) if bombs_left else False
    features.append(1.0 if can_escape else 0.0)

    # Crates & Opponents in blast range
    crates_count = count_crates_in_range(field, (x, y))
    features.append(min(crates_count / 4.0, 1.0))

    opp_count = sum(1 for (ox, oy) in other_xys if abs(ox - x) + abs(oy - y) <= 3)
    features.append(min(opp_count / 3.0, 1.0))

    # 3x3 Local board patch
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                features.append(-1.0)
            elif field[nx, ny] == -1:
                features.append(-1.0)
            elif field[nx, ny] == 1:
                features.append(1.0)
            elif danger_map[nx, ny] != 99:
                features.append(2.0)
            else:
                features.append(0.0)

    return np.array(features, dtype=np.float32)
