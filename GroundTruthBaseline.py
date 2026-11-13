
import heapq
import json
import math

import numpy as np
from ai2thor.controller import Controller

SCENE_NAME    = "FloorPlan201"
GRID_SIZE     = 0.25        # 1 cell = 0.25 m; the robot is treated as occupying exactly one cell

OUTPUT_JSON   = "ground_truth.json"
PLATFORM      = None        # Linux + Vulkan only, to avoid opening a Unity window:
                            #   from ai2thor.platform import CloudRendering; PLATFORM = CloudRendering

ROBOT_LOCATION = -2
EXIT_LOCATION  = -1
FREE_CELL      = 0
BLOCKED_CELL   = 1

CARDINAL_MOVES = {
    "up":    (-1,  0),  # +Z (North)
    "right": ( 0,  1),  # +X (East)
    "down":  ( 1,  0),  # -Z (South)
    "left":  ( 0, -1),  # -X (West)
}
MOVE_TO_YAW = {"up": 0, "right": 90, "down": 180, "left": 270}


def build_ground_truth_grid(reachable, objects, agent_pos, grid_size=GRID_SIZE):
    """Every reachable cell is FREE, everything else in the bounding box is BLOCKED."""
    to_cell = lambda p: (round(p["x"] / grid_size), round(p["z"] / grid_size))
    reach_xz = [to_cell(p) for p in reachable]
    obj_xz = [to_cell(o["position"]) for o in objects]

    if len(set(reach_xz)) != len(reach_xz):
        print("WARNING: two reachable positions quantise to the same cell. "
              "Check that the spawn position is a multiple of GRID_SIZE.")

    # Same bounds as the vision file (including object positions) so (row, col) indices line up
    all_x = [c[0] for c in reach_xz + obj_xz]
    all_z = [c[1] for c in reach_xz + obj_xz]
    min_x, max_x = min(all_x) - 1, max(all_x) + 1
    min_z, max_z = min(all_z) - 1, max(all_z) + 1

    def to_rc(gx, gz):
        return int(max_z - gz), int(gx - min_x)

    grid = np.full((max_z - min_z + 1, max_x - min_x + 1), BLOCKED_CELL, dtype=int)
    for gx, gz in reach_xz:
        grid[to_rc(gx, gz)] = FREE_CELL

    # Exit rule identical to the vision file: farthest reachable cell (Euclidean) from the spawn
    far = max(reachable, key=lambda p: math.hypot(p["x"] - agent_pos["x"], p["z"] - agent_pos["z"]))
    exit_rc = to_rc(*to_cell(far))
    rob_rc = to_rc(*to_cell(agent_pos))
    if grid[rob_rc] == BLOCKED_CELL:
        print("WARNING: spawn cell is not in the reachable set (quantisation mismatch).")

    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc] = ROBOT_LOCATION
    return grid, min_x, max_z, rob_rc, exit_rc


def astar(grid, start, goal):
    rows, cols = grid.shape

    def h(r, c):
        return abs(r - goal[0]) + abs(c - goal[1])

    pq = [(h(*start), 0, start)]
    cost = {start: 0}
    came_from = {}

    while pq:
        _, g, curr = heapq.heappop(pq)
        if curr == goal:
            path = [curr]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return path[::-1]
        if g > cost[curr]:
            continue
        for dr, dc in CARDINAL_MOVES.values():
            nxt = (curr[0] + dr, curr[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt] == BLOCKED_CELL:
                continue
            ng = g + 1
            if ng < cost.get(nxt, float("inf")):
                cost[nxt] = ng
                came_from[nxt] = curr
                heapq.heappush(pq, (ng + h(*nxt), ng, nxt))
    return None





# =====================================================================
# 4. MAIN
# =====================================================================
def main():
    kwargs = dict(
        scene=SCENE_NAME,
        gridSize=GRID_SIZE,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        width=100,
        height=100,
    )
    if PLATFORM is not None:
        kwargs["platform"] = PLATFORM

    controller = Controller(**kwargs)
    try:
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata["actionReturn"]
        objects = event.metadata["objects"]
        agent_pos = event.metadata["agent"]["position"]

        grid, min_x, max_z, start_rc, exit_rc = build_ground_truth_grid(reachable, objects, agent_pos)
        path = astar(grid, start_rc, exit_rc)
        if path is None:
            print(f"No path from {start_rc} to {exit_rc} in the ground-truth grid.")
            return

        steps = len(path) - 1
        manhattan = abs(start_rc[0] - exit_rc[0]) + abs(start_rc[1] - exit_rc[1])
        result = {
            "scene": SCENE_NAME,
            "grid_size": GRID_SIZE,
            "grid_shape": [int(v) for v in grid.shape],
            "num_reachable_cells": len(reachable),
            "start_rc": [int(v) for v in start_rc],
            "exit_rc": [int(v) for v in exit_rc],
            "optimal_steps": steps,
            "optimal_metres": steps * GRID_SIZE,
            "manhattan_lower_bound": manhattan,
            "path_rc": [[int(r), int(c)] for r, c in path],
        }



        with open(OUTPUT_JSON, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Scene {SCENE_NAME} | grid {grid.shape[0]}x{grid.shape[1]} | {len(reachable)} reachable cells")
        print(f"Start {start_rc} -> Exit {exit_rc}")
        print(f"Optimal path: {steps} steps = {steps * GRID_SIZE:.2f} m "
              f"(Manhattan lower bound {manhattan}, detour x{steps / max(manhattan, 1):.2f})")

        print(f"Saved {OUTPUT_JSON}")
    finally:
        controller.stop()



def compare_to_ground_truth(actual_steps, start_rc, exit_rc, reached, path=OUTPUT_JSON):
    """
    actual_steps = number of SUCCESSFUL MoveAhead calls the vision robot made (backtracking included).
    Returns SPL = reached * optimal / max(actual, optimal), the standard embodied-navigation metric.
    """
    with open(path) as f:
        gt = json.load(f)

    if [int(v) for v in start_rc] != gt["start_rc"] or [int(v) for v in exit_rc] != gt["exit_rc"]:
        print("WARNING: start/exit differ from the ground-truth run, so this comparison is not apples-to-apples.")

    opt = gt["optimal_steps"]
    if actual_steps < opt:
        print("WARNING: vision path is shorter than the optimum, which is impossible on the same grid. "
              "Check start/exit and the cell quantisation.")
    spl = (opt / max(actual_steps, opt)) if reached else 0.0

    print("---- Path length vs ground truth ----")
    print(f"Ground truth : {opt} steps ({gt['optimal_metres']:.2f} m)")
    print(f"Vision       : {actual_steps} steps ({actual_steps * gt['grid_size']:.2f} m) | reached exit: {reached}")
    print(f"Extra steps  : {actual_steps - opt} ({100.0 * (actual_steps - opt) / opt:.0f}% longer)")
    print(f"SPL          : {spl:.3f}")
    return spl


if __name__ == "__main__":
    main()
