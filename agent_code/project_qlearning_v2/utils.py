import numpy as np
from collections import deque

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
ACTION_TO_DELTA = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'WAIT': (0, 0)
}
DELTA_TO_ACTION = {v: k for k, v in ACTION_TO_DELTA.items() if k != 'WAIT'}


def get_bomb_danger_map(field, bombs, explosion_map, bomb_power=3):
    """
    Computes a 2D grid representing danger countdown timers for all arena tiles.
    - 0: Active explosion tile currently damaging agents.
    - 1 to 4: Bomb explosion will engulf tile in that many ticks.
    - 99: Completely safe tile (no imminent blast).
    """
    cols, rows = field.shape
    danger_map = np.full((cols, rows), 99, dtype=int)

    # Active explosions (damage active right now)
    danger_map[explosion_map > 0] = 0

    # Active bombs
    for (xb, yb), t in bombs:
        danger_map[xb, yb] = min(danger_map[xb, yb], t)
        
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            for dist in range(1, bomb_power + 1):
                nx, ny = xb + dx * dist, yb + dy * dist
                if not (0 <= nx < cols and 0 <= ny < rows):
                    break
                if field[nx, ny] == -1:  # Indestructible stone wall blocks explosion
                    break
                danger_map[nx, ny] = min(danger_map[nx, ny], t)
                if field[nx, ny] == 1:   # Destructible crate stops blast propagation beyond it
                    break

    return danger_map


def bfs_pathfinding(field, start, targets, free_tiles_mask):
    """
    Computes shortest path from start to closest reachable target using BFS.
    Returns:
        next_tile: (x, y) coordinate of the first move step towards closest target (or start if already there).
        distance: Integer path length (-1 if no target reachable).
    """
    if not targets or len(targets) == 0:
        return None, -1

    targets_set = set(targets)
    if start in targets_set:
        return start, 0

    cols, rows = field.shape
    frontier = deque([start])
    parent_dict = {start: None}
    dist_dict = {start: 0}
    found_target = None

    while frontier:
        current = frontier.popleft()
        if current in targets_set:
            found_target = current
            break

        cx, cy = current
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if 0 <= nx < cols and 0 <= ny < rows:
                if free_tiles_mask[nx, ny] and neighbor not in parent_dict:
                    parent_dict[neighbor] = current
                    dist_dict[neighbor] = dist_dict[current] + 1
                    frontier.append(neighbor)

    if found_target is None:
        return None, -1

    curr = found_target
    while parent_dict[curr] is not None and parent_dict[curr] != start:
        curr = parent_dict[curr]

    return curr, dist_dict[found_target]


def check_safe_moves(game_state, danger_map):
    """
    Evaluates safety of 5 cardinal/stationary moves (UP, DOWN, LEFT, RIGHT, WAIT).
    A move is deemed safe if:
    1. The destination tile is walkable (field == 0, no bomb, no explosion, no other agent).
    2. The destination tile has danger timer == 99 (completely safe).
    """
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    _, _, _, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = {xy for (xy, t) in bombs}
    others_xys = {xy for (_, _, _, xy) in game_state['others']}

    cols, rows = field.shape
    safe_actions = {}

    for action, (dx, dy) in ACTION_TO_DELTA.items():
        nx, ny = x + dx, y + dy
        if not (0 <= nx < cols and 0 <= ny < rows):
            safe_actions[action] = False
            continue

        if action == 'WAIT':
            is_walkable = True
            is_safe = (danger_map[x, y] == 99)
        else:
            is_walkable = (field[nx, ny] == 0 and explosion_map[nx, ny] == 0 and 
                           (nx, ny) not in bomb_xys and (nx, ny) not in others_xys)
            is_safe = is_walkable and (danger_map[nx, ny] == 99)

        safe_actions[action] = is_safe

    return safe_actions


def can_escape_bomb(game_state, bomb_pos=None):
    """
    Evaluates whether dropping a bomb at bomb_pos leaves at least one reachable safe tile
    within the 4-tick bomb countdown timer.
    """
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = list(game_state['bombs'])
    _, _, _, (x, y) = game_state['self']

    if bomb_pos is None:
        bomb_pos = (x, y)

    simulated_bombs = bombs + [(bomb_pos, 4)]
    sim_danger_map = get_bomb_danger_map(field, simulated_bombs, explosion_map)

    cols, rows = field.shape
    frontier = deque([(x, y, 0)])
    visited = {(x, y)}

    while frontier:
        cx, cy, dist = frontier.popleft()
        if sim_danger_map[cx, cy] == 99:
            return True

        if dist >= 3:
            continue

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in visited and field[nx, ny] == 0 and (nx, ny) != bomb_pos:
                    visited.add((nx, ny))
                    frontier.append((nx, ny, dist + 1))

    return False


def is_dead_end_tile(field, pos, free_mask):
    """
    Determines if pos is a dead-end corridor (<= 1 walkable exit).
    """
    x, y = pos
    cols, rows = field.shape
    walkable_exits = 0
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows and free_mask[nx, ny]:
            walkable_exits += 1
    return walkable_exits <= 1


def count_crates_in_range(field, pos, bomb_power=3):
    """
    Counts crates in line of sight within blast radius.
    """
    cols, rows = field.shape
    x, y = pos
    crate_count = 0

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        for dist in range(1, bomb_power + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if not (0 <= nx < cols and 0 <= ny < rows):
                break
            if field[nx, ny] == -1:
                break
            if field[nx, ny] == 1:
                crate_count += 1
                break

    return crate_count


def opponents_in_range(pos, others, bomb_power=3):
    """
    Counts opponents within Manhattan distance <= bomb_power.
    """
    x, y = pos
    count = 0
    for (_, _, _, (ox, oy)) in others:
        if abs(ox - x) + abs(oy - y) <= bomb_power:
            count += 1
    return count
