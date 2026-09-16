import numpy as np
from .utils import (
    get_bomb_danger_map,
    bfs_pathfinding,
    check_safe_moves,
    can_escape_bomb,
    is_dead_end_tile,
    count_crates_in_range,
    opponents_in_range,
    ACTION_TO_DELTA
)


def action_code_from_tile(current_pos, next_tile):
    """
    Converts coordinate delta (next_tile - current_pos) into integer action code:
    0: NONE, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT
    """
    if next_tile is None:
        return 0
    x, y = current_pos
    dx = next_tile[0] - x
    dy = next_tile[1] - y
    if (dx, dy) == (0, -1): return 1  # UP
    if (dx, dy) == (0, 1):  return 2  # DOWN
    if (dx, dy) == (-1, 0): return 3  # LEFT
    if (dx, dy) == (1, 0):  return 4  # RIGHT
    return 0


def distance_to_bucket(dist):
    """
    Discretizes true BFS distance into coarse buckets:
    0: adjacent (1 step)
    1: near (2-3 steps)
    2: medium (4-6 steps)
    3: far (>=7 steps)
    4: unreachable / none (-1)
    """
    if dist < 0:
        return 4
    if dist == 1:
        return 0
    if 2 <= dist <= 3:
        return 1
    if 4 <= dist <= 6:
        return 2
    return 3


def state_to_features_qlearning_v2(game_state):
    """
    Transforms raw game_state into an expressive, discrete, manageable tuple for Q-Learning v2.
    
    Components:
    1. dir_primary_target (0..4)
    2. dist_target_bucket (0..4)
    3. target_type (0: COIN, 1: CRATE, 2: OPPONENT, 3: NONE)
    4. danger_status (0: SAFE, 1: DANGER_IMMINENT <=2, 2: DANGER_DELAYED 3-4)
    5. dir_safety (0..4)
    6. safe_moves: (UP_safe, DOWN_safe, LEFT_safe, RIGHT_safe)
    7. bomb_feasibility (0: NO_BOMB, 1: UNSAFE, 2: SAFE_CRATES, 3: SAFE_OPP, 4: SAFE_USELESS)
    8. in_dead_end (0 or 1)
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

    # 2. Free tiles mask (walkable: field == 0, no explosion, no bomb, no other agent)
    bomb_xys = {xy for (xy, t) in bombs}
    free_mask = (field == 0) & (explosion_map == 0)
    for bxy in bomb_xys:
        free_mask[bxy] = False
    for oxy in other_xys:
        free_mask[oxy] = False

    # 3. Target prioritization & true pathfinding
    # Target 1: Coins
    coin_next, coin_dist = bfs_pathfinding(field, (x, y), coins, free_mask)

    # Target 2: Crates
    crate_tiles = [(cx, cy) for cx in range(1, cols - 1) for cy in range(1, rows - 1) if field[cx, cy] == 1]
    crate_next, crate_dist = bfs_pathfinding(field, (x, y), crate_tiles, free_mask)

    # Target 3: Opponents
    opp_next, opp_dist = bfs_pathfinding(field, (x, y), other_xys, free_mask)

    # Determine primary target by priority and availability
    if coin_dist != -1:
        primary_next, primary_dist, target_type = coin_next, coin_dist, 0  # COIN
    elif crate_dist != -1:
        primary_next, primary_dist, target_type = crate_next, crate_dist, 1  # CRATE
    elif opp_dist != -1:
        primary_next, primary_dist, target_type = opp_next, opp_dist, 2  # OPPONENT
    else:
        primary_next, primary_dist, target_type = None, -1, 3  # NONE

    dir_primary_target = action_code_from_tile((x, y), primary_next)
    dist_target_bucket = distance_to_bucket(primary_dist)

    # 4. Danger & Safety direction
    curr_danger = danger_map[x, y]
    if curr_danger == 99:
        danger_status = 0
        dir_safety = 0
    elif curr_danger <= 2:
        danger_status = 1  # IMMINENT
        safe_tiles = [(sx, sy) for sx in range(1, cols - 1) for sy in range(1, rows - 1)
                      if danger_map[sx, sy] == 99 and free_mask[sx, sy]]
        safe_next, _ = bfs_pathfinding(field, (x, y), safe_tiles, free_mask)
        dir_safety = action_code_from_tile((x, y), safe_next)
    else:
        danger_status = 2  # DELAYED
        safe_tiles = [(sx, sy) for sx in range(1, cols - 1) for sy in range(1, rows - 1)
                      if danger_map[sx, sy] == 99 and free_mask[sx, sy]]
        safe_next, _ = bfs_pathfinding(field, (x, y), safe_tiles, free_mask)
        dir_safety = action_code_from_tile((x, y), safe_next)

    # 5. Safe directional moves
    safe_moves_dict = check_safe_moves(game_state, danger_map)
    safe_moves = (
        safe_moves_dict.get('UP', False),
        safe_moves_dict.get('DOWN', False),
        safe_moves_dict.get('LEFT', False),
        safe_moves_dict.get('RIGHT', False)
    )

    # 6. Strategic Bomb Feasibility
    if not bombs_left:
        bomb_feasibility = 0
    else:
        can_escape = can_escape_bomb(game_state, (x, y))
        if not can_escape:
            bomb_feasibility = 1  # UNSAFE
        else:
            crates_cnt = count_crates_in_range(field, (x, y))
            opp_cnt = opponents_in_range((x, y), others)
            if crates_cnt > 0:
                bomb_feasibility = 2  # CRATES
            elif opp_cnt > 0:
                bomb_feasibility = 3  # OPPONENT
            else:
                bomb_feasibility = 4  # SAFE BUT USELESS

    # 7. Dead-end corridor detection
    in_dead_end = 1 if is_dead_end_tile(field, (x, y), free_mask) else 0

    return (
        dir_primary_target,
        dist_target_bucket,
        target_type,
        danger_status,
        dir_safety,
        safe_moves,
        bomb_feasibility,
        in_dead_end
    )
