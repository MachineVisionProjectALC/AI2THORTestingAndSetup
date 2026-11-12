



import heapq
import math
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

DIAGONAL_MOVES = {
    "up, right": (-1, 1),
    "down, right": (1, 1),
    "down, left": (1, -1),
    "up, left": (-1, -1),
}

ALL_MOVES = {**CARDINAL_MOVES, **DIAGONAL_MOVES}
# concatenate the two dictionaries to get all possible moves for the robot

MOVE_TO_YAW = { # move to yaw mapping for the robot to turn in the right direction
    #this doesnt consider vertical movement its with respect to 2d grid
    "up": 0,
    "up, right": 45,
    "right": 90,
    "down, right": 135,
    "down": 180,
    "down, left": 225,
    "left": 270,
    "up, left": 315,
} # here this dictionary is not used



#A* Pathfinding algorithm to find the next move for the robot to reach the exit in a grid
#The grid takes into account the robot size already. This is done using binary dilation python library
#So we can assume the robot is a point in the grid and can move to any free cell

def next_move(grid: np.ndarray, allow_diagonal: bool = True): # allow diagonal movement for the robot
    #the robot size is 1x1 so we can treat it as a point in the grid
    rows, cols = grid.shape
    robot_indices = np.argwhere(grid == ROBOT_LOCATION)
    exit_indices = np.argwhere(grid == EXIT_LOCATION)

    if (robot_indices.size == 0 or exit_indices.size == 0):
        return None, [] # we return none for no path

    start_coord = tuple(robot_indices[0].tolist()) # get the robot location as a tuple
    exit_coords = {tuple(exit_point) for exit_point in exit_indices.tolist()} # get the exit locations as a set of tuples
    #exit_indices[0].tolist()
    # we have 1 robot and 1 exit in the grid so we can just take the first one
    available_moves = ALL_MOVES if allow_diagonal else CARDINAL_MOVES
    move_name_by_delta = {delta: name for name, delta in available_moves.items()}
    # move name by delta is a dictionary that maps the delta of the move to the name of the move
    # the name of the move is used to turn the robot in the right direction to reach the exit
    # the delta of the move is used to calculate the cost of the move in the A* algorithm

    def calculate_heuristic(coord): # heuristic is h(n) = min distance from the current cell to any exit cell
        #heuristic can never overestimate the cost to reach the goal, so we use the minimum distance to any exit cell as the heuristic
        row, col = coord # coord is (row,col)
        min_distance = math.inf
        for exit_row, exit_col in exit_coords: # if there are multiple exits 
            dx = abs(row - exit_row) # 
            dy = abs(col - exit_col)
            if allow_diagonal:
                dist = max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
                #formula: https://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#diagonal-distance
                #D=1
                '''
                function heuristic(node) =
                dx = abs(node.x - goal.x)
                dy = abs(node.y - goal.y)
                return D * max(dx, dy) + (D2-D) * min(dx, dy).'''
            else:
                dist = dx + dy
            min_distance = min(min_distance, dist)
        return min_distance

    priority_queue = [(calculate_heuristic(start_coord), 0.0, start_coord)]
    # priority queue is a list of tuples (estimated total cost, cost to reach, coordinate)
    cost_to_reach = {start_coord: 0.0}
    #cost to reach is a dictionary that maps coordinates to the cost to reach that coordinate from the start coordinate
    came_from = {}
    # came from is a dictionary that maps coordinates to the coordinate that we came from to reach that coordinate

    while priority_queue:
        _, current_cost, current_coord = heapq.heappop(priority_queue)
        # pop the coordinate with the lowest estimated total cost from the priority queue
        #the first variable is the estimated total cost, we don't need it 

        if current_coord in exit_coords:
            # we have reached an exit, reconstruct the path
            path = [current_coord]
            while path[-1] != start_coord:
                path.append(came_from[path[-1]])
                # reconstruct the path by following the came_from dictionary from the exit to the start
            path.reverse()

            if len(path) < 2:
                # if the path is only 1 step long, we are already at the exit, so there is no next move
                return None, path

            first_step = path[1]
            
            step_delta = (first_step[0] - start_coord[0], first_step[1] - start_coord[1])
            # here we calculate the delta between the first step and the start coordinate to determine the move name
            return move_name_by_delta.get(step_delta), path
        # move name is the name of the move to take to reach the exit, path is the list of coordinates from the start to the exit

        if current_cost > cost_to_reach[current_coord]:
            continue
        # there exists a better path, skip this one

        for delta_row, delta_col in available_moves.values():
            # calculate the neighbor coordinate by adding the delta to the current coordinate
            neighbor_coord = (current_coord[0] + delta_row, current_coord[1] + delta_col)
            n_row, n_col = neighbor_coord
            # check if the neighbor coordinate is within the grid bounds and not blocked

            if not (0 <= n_row < rows and 0 <= n_col < cols) or grid[neighbor_coord] == BLOCKED_CELL:
                continue
            # check for diagonal movement and apply lenient corner-cutting rules
            # Lenient corner-cutting: only block diagonal if both orthogonal neighbors are blocked
            # for instance, if moving from (1,1) to (2,2), we check (1,2) and (2,1). If both are blocked, we cannot move diagonally.
            # prevents getting stuck near noisy stereo edges


            #May or may not want to not use lenient corner cutting as a safety measure. We can use binary dilation for the occupancy grid
            #this will inflate the size of obstacles
            #with good enough grid resolution should be irrelevant if this is turned on or off
            # Turning it off now - replace and with or

            
            if delta_row != 0 and delta_col != 0:
                ortho_1 = (current_coord[0] + delta_row, current_coord[1])
                ortho_2 = (current_coord[0], current_coord[1] + delta_col)
                if grid[ortho_1] == BLOCKED_CELL or grid[ortho_2] == BLOCKED_CELL:
                    continue
            # calculate the cost to reach the neighbor coordinate
            unknown_penalty = 0.15 if grid[neighbor_coord] == UNKNOWN_CELL else 0.0
            # small penalty is added to the cost if the neighbor cell is unknown, to encourage exploration of known free cells first
            step_cost = math.hypot(delta_row, delta_col) + unknown_penalty
            tentative_cost = current_cost + step_cost
            # if the tentative cost to reach the neighbor coordinate is less than the previously recorded cost, update the cost and path
            if tentative_cost < cost_to_reach.get(neighbor_coord, math.inf):
                cost_to_reach[neighbor_coord] = tentative_cost
                came_from[neighbor_coord] = current_coord
                estimated_total = tentative_cost + calculate_heuristic(neighbor_coord)
                
                heapq.heappush(priority_queue, (estimated_total, tentative_cost, neighbor_coord))
                # we push the neighbor coordinate onto the priority queue with its estimated total cost and tentative cost
                #priority queue is a min-heap, so the coordinate with the lowest estimated total cost will be popped first

    return None, [] ## the function returns None if there is no path to the exit, and an empty list for the path
