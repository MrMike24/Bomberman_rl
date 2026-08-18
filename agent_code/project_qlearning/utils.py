from collections import deque
import numpy as np

# Actions mapping
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
    Computes a 2D grid of danger timers for all tiles on the board.
    
    Values:
    - 0 to 4: tile will be/is engulfed by explosion in that many steps (0 = explosion active now).
    - 99: safe tile (no explosion or bomb threat).
    """
    cols, rows = field.shape
    danger_map = np.full((cols, rows), 99, dtype=int)

    # Active explosions (timer 0)
    danger_map[explosion_map > 0] = 0

    # Active bombs
    for (xb, yb), t in bombs:
        danger_map[xb, yb] = min(danger_map[xb, yb], t)
        
        # Raycast in 4 directions
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            for dist in range(1, bomb_power + 1):
                nx, ny = xb + dx * dist, yb + dy * dist
                if not (0 <= nx < cols and 0 <= ny < rows):
                    break
                # Stone wall blocks explosion completely
                if field[nx, ny] == -1:
                    break
                
                danger_map[nx, ny] = min(danger_map[nx, ny], t)
                
                # Crate catches explosion but blocks behind it
                if field[nx, ny] == 1:
                    break

    return danger_map


def bfs_pathfinding(field, start, targets, free_tiles_mask, logger=None):
    """
    Finds the shortest path from start to closest target using BFS over free_tiles_mask.
    
    Returns:
    - next_tile: coordinate (x,y) of the first step on shortest path to closest target (or start if at target/no path).
    - distance: integer shortest path distance (-1 if unreachable).
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

    # Backtrack to find first step after start
    curr = found_target
    while parent_dict[curr] is not None and parent_dict[curr] != start:
        curr = parent_dict[curr]

    return curr, dist_dict[found_target]


def check_safe_moves(game_state, danger_map):
    """
    Evaluates safety of neighbors for actions: UP, DOWN, LEFT, RIGHT, WAIT.
    A move is safe if:
    1. Tile is walkable (field == 0, not blocked by wall/crate/bomb/other agent).
    2. Tile danger timer is 99 (completely safe) or > 1 step away so agent can move through if needed.
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

        # Check walkability
        is_walkable = True
        if action == 'WAIT':
            is_walkable = True
        else:
            if field[nx, ny] != 0 or explosion_map[nx, ny] > 0 or (nx, ny) in bomb_xys or (nx, ny) in others_xys:
                is_walkable = False

        if not is_walkable:
            safe_actions[action] = False
            continue

        # Check danger
        tile_danger = danger_map[nx, ny]
        # Fully safe if danger_map == 99
        is_safe = (tile_danger == 99)
        safe_actions[action] = is_safe

    return safe_actions


def can_escape_bomb(game_state, bomb_pos=None):
    """
    Simulates dropping a bomb at bomb_pos (default current position) and checks
    if there exists at least one safe tile reachable within 3 steps before explosion.
    """
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    bombs = list(game_state['bombs'])
    _, _, _, (x, y) = game_state['self']

    if bomb_pos is None:
        bomb_pos = (x, y)

    # Add simulated bomb with countdown 4
    simulated_bombs = bombs + [(bomb_pos, 4)]
    sim_danger_map = get_bomb_danger_map(field, simulated_bombs, explosion_map)

    cols, rows = field.shape
    # BFS to find reachable tile with sim_danger_map == 99 within 3 steps
    frontier = deque([(x, y, 0)])
    visited = {(x, y)}

    while frontier:
        cx, cy, dist = frontier.popleft()
        if sim_danger_map[cx, cy] == 99:
            return True  # Found safe tile!

        if dist >= 3:
            continue

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if (nx, ny) not in visited and field[nx, ny] == 0 and (nx, ny) != bomb_pos:
                    visited.add((nx, ny))
                    frontier.append((nx, ny, dist + 1))

    return False


def count_crates_in_range(field, pos, bomb_power=3):
    """
    Counts destructible crates in blast radius if bomb is dropped at pos.
    """
    cols, rows = field.shape
    x, y = pos
    crate_count = 0

    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        for dist in range(1, bomb_power + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if not (0 <= nx < cols and 0 <= ny < rows):
                break
            if field[nx, ny] == -1:  # Wall stops ray
                break
            if field[nx, ny] == 1:   # Crate found and ray stops
                crate_count += 1
                break

    return crate_count
