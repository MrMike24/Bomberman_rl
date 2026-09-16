import numpy as np
from .utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    check_safe_moves,
    can_escape_bomb,
    count_escape_routes,
    is_dead_end_tile,
    count_crates_in_range,
    opponents_in_range
)


def state_to_features_dqn_v2(game_state):
    """
    Transforms raw game_state into a 38-dimensional normalized float32 vector for Double DQN v2.
    
    Feature Layout (38 dimensions):
    - [0:3]   Nearest coin relative vector: (dx/17, dy/17, reachable_dist/17)
    - [3:6]   Nearest crate relative vector: (dx/17, dy/17, reachable_dist/17)
    - [6:9]   Nearest opponent relative vector: (dx/17, dy/17, reachable_dist/17)
    - [9:12]  Nearest safe tile relative vector if in danger: (dx/17, dy/17, reachable_dist/17)
    - [12:17] 5 action safety indicators: (UP, DOWN, LEFT, RIGHT, WAIT) -> 1.0 if safe else 0.0
    - [17]    Current position in danger: (1.0 if danger else 0.0)
    - [18]    Danger timer normalized: (timer / 4.0 if in danger else 1.0)
    - [19]    Bomb available: (1.0 if True else 0.0)
    - [20]    Bomb drop escape safe: (1.0 if safe else 0.0)
    - [21]    Escape route diversity: (min(escape_routes / 4.0, 1.0))
    - [22]    Current position in dead-end corridor: (1.0 if True else 0.0)
    - [23]    Crates in blast range: (min(crates / 4.0, 1.0))
    - [24]    Opponents in blast range: (min(opponents / 3.0, 1.0))
    - [25]    Opponent trapped in dead-end: (1.0 if True else 0.0)
    - [26:35] 3x3 local grid surrounding agent: (-1.0: wall, 0.5: crate, 1.0: opponent, -0.5: bomb/danger, 0.0: free)
    - [35]    Step progression: ((400 - step) / 400.0)
    - [36]    Coins remaining ratio: (min(len(coins) / 50.0, 1.0))
    - [37]    Active opponents ratio: (len(others) / 3.0)
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
    step = game_state.get('step', 0)

    cols, rows = field.shape
    features = []

    # 1. Danger map
    danger_map = get_bomb_danger_map(field, bombs, explosion_map)

    # 2. Free tiles mask
    bomb_xys = {xy for (xy, t) in bombs}
    free_mask = (field == 0) & (explosion_map == 0)
    for bxy in bomb_xys:
        free_mask[bxy] = False
    for oxy in other_xys:
        free_mask[oxy] = False

    # Pathfinding vector helper
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

    # [0:3] Nearest Coin
    features.extend(get_target_vec(coins))

    # [3:6] Nearest Crate
    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    features.extend(get_target_vec(crate_tiles))

    # [6:9] Nearest Opponent
    features.extend(get_target_vec(other_xys))

    # [9:12] Nearest Safe Tile (if in danger)
    curr_danger = danger_map[x, y]
    if curr_danger != 99:
        safe_tiles = [(sx, sy) for sx in range(1, cols - 1) for sy in range(1, rows - 1)
                      if danger_map[sx, sy] == 99 and free_mask[sx, sy]]
        features.extend(get_target_vec(safe_tiles))
    else:
        features.extend([0.0, 0.0, 0.0])

    # [12:17] 5 Action Safety Indicators
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    for act in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']:
        features.append(1.0 if safe_moves_dict.get(act, False) else 0.0)

    # [17] In Danger Flag
    features.append(1.0 if curr_danger != 99 else 0.0)

    # [18] Danger Timer Normalized
    features.append(min(curr_danger / 4.0, 1.0) if curr_danger != 99 else 1.0)

    # [19] Bomb Available
    features.append(1.0 if bombs_left else 0.0)

    # [20] Bomb Escape Safe
    can_escape = can_escape_bomb(game_state, (x, y)) if bombs_left else False
    features.append(1.0 if can_escape else 0.0)

    # [21] Escape Route Diversity
    escape_exits = count_escape_routes(game_state, (x, y)) if bombs_left else 0
    features.append(min(escape_exits / 4.0, 1.0))

    # [22] Current Position Dead-End Corridor
    in_dead_end = is_dead_end_tile(field, (x, y), free_mask)
    features.append(1.0 if in_dead_end else 0.0)

    # [23] Crates in Blast Range
    crates_count = count_crates_in_range(field, (x, y))
    features.append(min(crates_count / 4.0, 1.0))

    # [24] Opponents in Blast Range
    opp_count = opponents_in_range((x, y), others)
    features.append(min(opp_count / 3.0, 1.0))

    # [25] Opponent Trapped in Dead-End
    opp_trapped = any(is_dead_end_tile(field, (ox, oy), free_mask) for (ox, oy) in other_xys)
    features.append(1.0 if opp_trapped else 0.0)

    # [26:35] 3x3 Local Semantic Patch
    other_set = set(other_xys)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                features.append(-1.0)  # Boundary wall
            elif field[nx, ny] == -1:
                features.append(-1.0)  # Stone wall
            elif field[nx, ny] == 1:
                features.append(0.5)   # Crate
            elif (nx, ny) in other_set:
                features.append(1.0)   # Opponent
            elif danger_map[nx, ny] != 99:
                features.append(-0.5)  # Bomb / explosion threat
            else:
                features.append(0.0)   # Free tile

    # [35] Step Progression Normalized
    features.append(max(0.0, (400.0 - step) / 400.0))

    # [36] Coins Remaining Ratio
    features.append(min(len(coins) / 50.0, 1.0))

    # [37] Active Opponents Ratio
    features.append(len(others) / 3.0)

    return np.array(features, dtype=np.float32)
