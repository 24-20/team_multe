"""BFS pathfinding utilities shared across bot implementations."""
from collections import deque


def bfs(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
    width: int,
    height: int,
    occupied: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return shortest path from start to goal, or [] if unreachable.

    Path includes start but excludes goal (goal may be a shelf/wall).
    If goal itself is walkable, path includes goal.
    """
    if start == goal:
        return [start]

    blocked = walls | (occupied or set())
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        cx, cy = path[-1]

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            pos = (nx, ny)

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if pos in visited:
                continue

            new_path = path + [pos]

            if pos == goal:
                return new_path

            # Don't enter walls unless it's the goal (shelves are walls we stand adjacent to)
            if pos in blocked:
                continue

            visited.add(pos)
            queue.append(new_path)

    return []


def next_step_towards(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: set[tuple[int, int]],
    width: int,
    height: int,
    occupied: set[tuple[int, int]] | None = None,
) -> str | None:
    """Return the action string to move one step towards goal, or None if already there / unreachable."""
    path = bfs(start, goal, walls, width, height, occupied)
    if len(path) < 2:
        return None
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    return {(1, 0): "move_right", (-1, 0): "move_left", (0, 1): "move_down", (0, -1): "move_up"}[
        (dx, dy)
    ]


def adjacent_cells(pos: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
