import heapq
import math
import cv2
import numpy as np

ROBOT_LOCATION = -2 # for the occupancy grid we have unique grid squares as robot and exit
EXIT_LOCATION = -1
FREE_CELL = 0 # free cells factor into the A* path
BLOCKED_CELL = 1
UNKNOWN_CELL = 2

CARDINAL_MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MOVE_TO_YAW = { # move to yaw mapping for the robot to turn in the right direction
    # this doesnt consider vertical movement its with respect to 2d grid
    "up": 0,
    "right": 90,
    "down": 180,
    "left": 270,
} # here this dictionary is not used


# =====================================================================
# 4. 4-WAY MANHATTAN A* WITH SOFT OBSTACLE CLEARANCE
# =====================================================================
# A* Pathfinding algorithm to find the next move for the robot to reach the exit in a grid
# The grid takes into account the robot size already. This is done using binary dilation python library
# So we can assume the robot is a point in the grid and can move to any free cell

def next_move(grid: np.ndarray):
    # the robot size is 1x1 so we can treat it as a point in the grid
    rows, cols = grid.shape
    robot_pts = np.argwhere(grid == ROBOT_LOCATION)
    exit_pts  = np.argwhere(grid == EXIT_LOCATION)

    if robot_pts.size == 0 or exit_pts.size == 0:
        return None, [] # we return none for no path

    # get the robot and exit locations as tuples of Python ints
    # we have 1 robot and 1 exit in the grid so we can just take the first one
    start, goal = tuple(int(v) for v in robot_pts[0]), tuple(int(v) for v in exit_pts[0])

    # soft clearance: discourages hugging walls on a 0.25 m grid without hard-blocking 0.5 m doorways
    near_blocked = cv2.dilate((grid == BLOCKED_CELL).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    # heuristic is h(n) = Manhattan distance from the current cell to the exit cell
    # heuristic can never overestimate the cost to reach the goal, so we use Manhattan distance
    heuristic = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    # priority queue is a list of tuples (estimated total cost, cost to reach, coordinate)
    pq = [(heuristic(start), 0.0, start)]
    # cost_to_reach is a dictionary that maps coordinates to the cost to reach that coordinate from the start coordinate
    # came_from is a dictionary that maps coordinates to the coordinate that we came from to reach that coordinate
    cost_to_reach, came_from = {start: 0.0}, {}

    while pq:
        # pop the coordinate with the lowest estimated total cost from the priority queue
        # the first variable is the estimated total cost, we don't need it
        _, g_cost, curr = heapq.heappop(pq)

        if curr == goal:
            # we have reached the exit, reconstruct the path
            path = [curr]
            while path[-1] != start:
                # reconstruct the path by following the came_from dictionary from the exit to the start
                path.append(came_from[path[-1]])
            path.reverse()

            if len(path) < 2:
                # if the path is only 1 step long, we are already at the exit, so there is no next move
                return None, path

            # here we calculate the delta between the first step and the start coordinate to determine the move name
            # the name of the move is used to turn the robot in the right direction to reach the exit
            delta = (path[1][0] - start[0], path[1][1] - start[1])
            return next(k for k, v in CARDINAL_MOVES.items() if v == delta), path

        if g_cost > cost_to_reach[curr]:
            # there exists a better path, skip this one
            continue

        for dr, dc in CARDINAL_MOVES.values():
            # calculate the neighbor coordinate by adding the delta to the current coordinate
            nb = (curr[0] + dr, curr[1] + dc)

            # check if the neighbor coordinate is within the grid bounds and not blocked
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid[nb] == BLOCKED_CELL:
                continue

            # calculate the cost to reach the neighbor coordinate
            # small penalty (+0.25) is added if the neighbor cell is unknown, to encourage exploration of known free cells first
            step = 1.25 if grid[nb] == UNKNOWN_CELL else 1.0
            # additional penalty (+0.90) is added if the cell is adjacent to a blocked cell (unless it is the goal)
            if near_blocked[nb] and nb != goal:
                step += 0.90

            # if the tentative cost to reach the neighbor coordinate is less than the previously recorded cost, update the cost and path
            if g_cost + step < cost_to_reach.get(nb, float("inf")):
                cost_to_reach[nb] = g_cost + step
                came_from[nb] = curr
                # we push the neighbor coordinate onto the priority queue with its estimated total cost and tentative cost
                # priority queue is a min-heap, so the coordinate with the lowest estimated total cost will be popped first
                heapq.heappush(pq, (g_cost + step + heuristic(nb), g_cost + step, nb))

    return None, [] # the function returns None if there is no path to the exit, and an empty list for the path
