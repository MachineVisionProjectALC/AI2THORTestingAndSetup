import heapq
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ai2thor.controller import Controller
from nav_metrics import MetricsTracker                                   
# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
SCENE_NAME      = "FloorPlan_Train12_5"
GRID_SIZE       = 0.25   # Matches ProcTHOR's native 0.25m doorway/furniture alignment
CAMERA_HEIGHT   = 1.575  # AI2-THOR standing camera height above floor (m)
BASELINE        = 0.12   # 12 cm horizontal stereo baseline (right shift)
IMG_W, IMG_H    = 300, 300
WORK_H          = 300
FOV             = 90.0   # Vertical FOV in degrees (Unity convention)
MAX_RANGE       = 2.2    # Cut off at 2.2m to prevent z^2 stereo depth bleed into empty cells
BUMP_LOCK       = 4.0    # Log-odds written ONLY by a physical collision

# Relative goal offset from spawn in the robot's initial local frame:
# (meters_right_of_spawn, meters_ahead_of_spawn) — zero Unity floorplan knowledge
GOAL_OFFSET_M   = (2.00, 2.50)
HALF_GRID_CELLS = 40     # 81x81 local grid (+/- 10.0m around spawn) — prevents out-of-bounds crash

# Anti-phantom obstacle classification thresholds
OBS_MIN_H       = 0.22   # m above floor (ignores rugs, thresholds, and floor disparity noise)
OBS_MAX_H       = 1.50   # m above floor (ignores ceiling/door headers)
FLOOR_BAND      = 0.16   # +/- m around floor plane counted as walkable floor
MIN_OBS_POINTS  = 85     # Requires solid surface cluster (rejects sparse edge bleed)
HIT_LOGODDS     = 0.60   # Requires 2+ consistent frames (>= 1.0) to mark BLOCKED

ROBOT_LOCATION  = -2
EXIT_LOCATION   = -1
FREE_CELL       = 0
BLOCKED_CELL    = 1
UNKNOWN_CELL    = 2

CARDINAL_MOVES = {
    "up":    (-1,  0),  # +Z (Local Forward at spawn)
    "right": ( 0,  1),  # +X (Local Right at spawn)
    "down":  ( 1,  0),  # -Z (Local Backward at spawn)
    "left":  ( 0, -1),  # -X (Local Left at spawn)
}

MOVE_TO_YAW = {
    "up":    0,
    "right": 90,
    "down":  180,
    "left":  270,
}


# =====================================================================
# 1. HARDWARE RIG SIMULATION (STEREO CAMERA PAIR ONLY)
# =====================================================================
def sync_stereo_rig(controller: Controller, baseline: float = BASELINE) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates a rigid physical stereo camera pair bolted to the robot head.
    Returns ONLY raw (rgb_left, rgb_right) frames — zero pose or room metadata is leaked.
    """
    meta = controller.last_event.metadata
    cam_pos = meta["cameraPosition"]
    agent_rot = meta["agent"]["rotation"]
    horizon = meta["agent"].get("cameraHorizon", 0.0)

    yaw_rad = math.radians(agent_rot["y"])
    rx = cam_pos["x"] + baseline * math.cos(yaw_rad)
    rz = cam_pos["z"] - baseline * math.sin(yaw_rad)
    ry = cam_pos["y"]

    event = controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        position=dict(x=rx, y=ry, z=rz),
        rotation=dict(x=horizon, y=agent_rot["y"], z=0.0),
        fieldOfView=FOV
    )
    return event.frame, event.third_party_camera_frames[0]


def init_environment(render_depth=False):
    """Enable simulator depth for evaluation with render_depth=True."""
    controller = Controller(
        scene=SCENE_NAME,
        gridSize=GRID_SIZE,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=render_depth,  # Optional ground truth; stereo uses RGB only.
        width=IMG_W,
        height=IMG_H,
        fieldOfView=FOV,
    )

    # Tilt head down 15 degrees so the floor right in front of the robot is visible
    controller.step(action="LookDown", degrees=15)

    meta = controller.last_event.metadata
    cam_pos = meta["cameraPosition"]
    yaw_rad = math.radians(meta["agent"]["rotation"]["y"])
    horizon = meta["agent"].get("cameraHorizon", 15.0)

    controller.step(
        action="AddThirdPartyCamera",
        position=dict(
            x=cam_pos["x"] + BASELINE * math.cos(yaw_rad),
            y=cam_pos["y"],
            z=cam_pos["z"] - BASELINE * math.sin(yaw_rad),
        ),
        rotation=dict(
            x=horizon,
            y=meta["agent"]["rotation"]["y"],
            z=0.0,
        ),
        fieldOfView=FOV
    )

    # Pure local grid centered at spawn (0, 0) — zero GetReachablePositions or objects metadata
    size = 2 * HALF_GRID_CELLS + 1
    grid     = np.full((size, size), UNKNOWN_CELL, dtype=int)
    log_odds = np.zeros((size, size), dtype=np.float32)

    rob_rc = (HALF_GRID_CELLS, HALF_GRID_CELLS)
    goal_dx, goal_dz = GOAL_OFFSET_M
    exit_r = int(np.clip(HALF_GRID_CELLS - round(goal_dz / GRID_SIZE), 1, size - 2))
    exit_c = int(np.clip(HALF_GRID_CELLS + round(goal_dx / GRID_SIZE), 1, size - 2))
    exit_rc = (exit_r, exit_c)

    log_odds[rob_rc] = -4.0
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION

    return controller, grid, log_odds, rob_rc, exit_rc, float(horizon)


def threshold_grid(
    grid: np.ndarray,
    log_odds: np.ndarray,
    visited_free: set,
    rob_rc: tuple[int, int],
    exit_rc: tuple[int, int]
):
    """Rebuilds discrete planning grid from log-odds with safe bounds clamping."""
    rows, cols = grid.shape
    grid[:] = UNKNOWN_CELL
    grid[log_odds <= -0.4] = FREE_CELL
    grid[log_odds >= 1.0] = BLOCKED_CELL
    if visited_free:
        valid_vis = [(r, c) for (r, c) in visited_free if 0 <= r < rows and 0 <= c < cols]
        if valid_vis:
            vr, vc = zip(*valid_vis)
            grid[list(vr), list(vc)] = FREE_CELL
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION


# =====================================================================
# 2. CLASSICAL VISION: SIFT + EPIPOLAR + SGBM + PITCH-CORRECTED 3D
# =====================================================================
class ClassicalStereoMapper:
    def __init__(self, src_w: int, src_h: int, pitch_deg: float = 15.0, fov=FOV, baseline=BASELINE, work_h=WORK_H):
        self.H = int(work_h)
        self.W = int(round(src_w * self.H / src_h))
        self.baseline = baseline
        self.pitch_rad = math.radians(pitch_deg)  # Positive = looking down

        fov_rad = math.radians(fov)
        self.fy = (self.H / 2.0) / math.tan(fov_rad / 2.0)
        self.fx = self.fy
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

        u = np.arange(self.W, dtype=np.float32)
        v = np.arange(self.H, dtype=np.float32)
        self.uu, self.vv = np.meshgrid(u, v)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.03)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.num_disp = 64
        block_size = 7
        self.sgbm_left = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=self.num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size**2,
            P2=32 * 1 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=120,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        self.sgbm_right = cv2.StereoSGBM_create(
            minDisparity=-self.num_disp,
            numDisparities=self.num_disp,
            blockSize=block_size,
            P1=8 * 1 * block_size**2,
            P2=32 * 1 * block_size**2,
            uniquenessRatio=15,
            speckleWindowSize=120,
            speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_HH
        )

    @classmethod
    def from_frame(cls, frame: np.ndarray, **kw):
        h, w = frame.shape[:2]
        return cls(src_w=w, src_h=h, **kw)

    def to_work(self, img: np.ndarray) -> np.ndarray:
        if img.shape[0] == self.H and img.shape[1] == self.W:
            return np.ascontiguousarray(img)
        return np.ascontiguousarray(cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA))

    def compute_clean_disparity(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray:
        disp_l = self.sgbm_left.compute(gray_l, gray_r).astype(np.float32) / 16.0
        disp_r = -self.sgbm_right.compute(gray_r, gray_l).astype(np.float32) / 16.0
        disp_l = cv2.medianBlur(disp_l, 5)

        u_right = np.clip(np.round(self.uu - disp_l).astype(int), 0, self.W - 1)
        v_idx   = self.vv.astype(int)
        lr_diff = np.abs(disp_l - disp_r[v_idx, u_right])

        grad_x = np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 0, 1, ksize=3))
        texture = np.hypot(grad_x, grad_y)

        valid_border = self.uu >= self.num_disp
        clean_mask = (
            valid_border &
            (disp_l > 2.0) &
            (disp_l < (self.num_disp - 4)) &
            (lr_diff <= 1.0) &
            (texture > 14.0)
        )

        clean_disp = np.zeros_like(disp_l)
        clean_disp[clean_mask] = disp_l[clean_mask]
        return clean_disp

    def match_sift_epipolar(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray:
        kp_l, desc_l = self.sift.detectAndCompute(gray_l, None)
        kp_r, desc_r = self.sift.detectAndCompute(gray_r, None)

        if desc_l is None or desc_r is None or len(kp_l) < 4 or len(kp_r) < 4:
            return np.empty((0, 3), dtype=np.float32)

        raw_matches = self.matcher.knnMatch(desc_l, desc_r, k=2)
        pts_3d = []
        for m_pair in raw_matches:
            if len(m_pair) < 2:
                continue
            m, n = m_pair
            if m.distance < 0.72 * n.distance:
                u_l, v_l = kp_l[m.queryIdx].pt
                u_r, v_r = kp_r[m.trainIdx].pt
                if abs(v_l - v_r) <= 1.0:
                    disp = u_l - u_r
                    if 2.0 < disp < (self.num_disp - 4):
                        z = (self.fx * self.baseline) / disp
                        if 0.30 <= z <= MAX_RANGE:
                            x = (u_l - self.cx) * z / self.fx
                            y = -(v_l - self.cy) * z / self.fy
                            pts_3d.append((x, y, z))
        return np.array(pts_3d, dtype=np.float32) if pts_3d else np.empty((0, 3), dtype=np.float32)

    def update_grid_from_stereo(
        self,
        grid: np.ndarray,
        log_odds: np.ndarray,
        visited_free: set,
        rob_rc: tuple[int, int],
        exit_rc: tuple[int, int],
        local_yaw: int,
        rgb_left: np.ndarray,
        rgb_right: np.ndarray,
        max_range: float = MAX_RANGE
    ):
        gray_l = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_left), cv2.COLOR_RGB2GRAY))
        gray_r = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_right), cv2.COLOR_RGB2GRAY))

        disp = self.compute_clean_disparity(gray_l, gray_r)
        valid = disp > 0

        z_cam = np.zeros_like(disp)
        z_cam[valid] = (self.fx * self.baseline) / disp[valid]
        valid &= (z_cam >= 0.30) & (z_cam <= max_range)

        x_cam = (self.uu[valid] - self.cx) * z_cam[valid] / self.fx
        y_cam = -(self.vv[valid] - self.cy) * z_cam[valid] / self.fy
        z_pts = z_cam[valid]

        pts_dense = np.stack([x_cam, y_cam, z_pts], axis=-1)
        pts_sift  = self.match_sift_epipolar(gray_l, gray_r)
        pts_cam   = np.vstack([pts_dense] + [pts_sift] * 3) if len(pts_sift) > 0 else pts_dense

        if len(pts_cam) == 0:
            return disp, pts_sift

        # Undo downward camera pitch to get horizontal forward (z_horiz) and vertical (y_level)
        cos_p, sin_p = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
        x_level = pts_cam[:, 0]
        y_level = pts_cam[:, 1] * cos_p - pts_cam[:, 2] * sin_p
        z_horiz = pts_cam[:, 1] * sin_p + pts_cam[:, 2] * cos_p

        # Transform into self-tracked local world coordinates
        rob_x = (rob_rc[1] - HALF_GRID_CELLS) * GRID_SIZE
        rob_z = (HALF_GRID_CELLS - rob_rc[0]) * GRID_SIZE

        yaw_rad = math.radians(local_yaw)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        x_local = rob_x + x_level * cos_y + z_horiz * sin_y
        z_local = rob_z - x_level * sin_y + z_horiz * cos_y
        h_above_floor = CAMERA_HEIGHT + y_level

        # Dynamic floor height calibration from seen floor points
        floor_cands = h_above_floor[(h_above_floor > -0.18) & (h_above_floor < 0.18)]
        floor_offset = float(np.median(floor_cands)) if len(floor_cands) > 40 else 0.0
        h_rel = h_above_floor - floor_offset

        obs_mask   = (h_rel >= OBS_MIN_H) & (h_rel <= OBS_MAX_H)
        floor_mask = np.abs(h_rel) < FLOOR_BAND

        rows, cols = grid.shape
        obs_counts   = np.zeros((rows, cols), dtype=int)
        floor_counts = np.zeros((rows, cols), dtype=int)

        obs_r = np.round(HALF_GRID_CELLS - (z_local[obs_mask] / GRID_SIZE)).astype(int)
        obs_c = np.round(HALF_GRID_CELLS + (x_local[obs_mask] / GRID_SIZE)).astype(int)
        ok_obs = (obs_r >= 0) & (obs_r < rows) & (obs_c >= 0) & (obs_c < cols)
        np.add.at(obs_counts, (obs_r[ok_obs], obs_c[ok_obs]), 1)

        flr_r = np.round(HALF_GRID_CELLS - (z_local[floor_mask] / GRID_SIZE)).astype(int)
        flr_c = np.round(HALF_GRID_CELLS + (x_local[floor_mask] / GRID_SIZE)).astype(int)
        ok_flr = (flr_r >= 0) & (flr_r < rows) & (flr_c >= 0) & (flr_c < cols)
        np.add.at(floor_counts, (flr_r[ok_flr], flr_c[ok_flr]), 1)

        # Log-odds update
        mutable = np.ones((rows, cols), dtype=bool)
        mutable[rob_rc]  = False
        mutable[exit_rc] = False
        if visited_free:
            valid_vis = [(r, c) for (r, c) in visited_free if 0 <= r < rows and 0 <= c < cols]
            if valid_vis:
                vr, vc = zip(*valid_vis)
                mutable[list(vr), list(vc)] = False

        strong_obs = obs_counts >= MIN_OBS_POINTS
        hit = mutable & strong_obs
        log_odds[hit] = np.minimum(log_odds[hit] + HIT_LOGODDS, 3.0)

        # Aggressively clear any cell where floor is observed and few obstacle points exist
        seen_floor = mutable & ~strong_obs & (floor_counts >= 5) & (obs_counts <= 8) & (log_odds < BUMP_LOCK)
        log_odds[seen_floor] = np.maximum(log_odds[seen_floor] - 0.85, -4.0)

        # Clear 3-cell wide forward corridor up to the first real obstacle
        yaw_vectors = {
            0:   ((-1,  0), ( 0,  1)),
            90:  (( 0,  1), ( 1,  0)),
            180: (( 1,  0), ( 0, -1)),
            270: (( 0, -1), (-1,  0)),
        }
        (fr, fc), (rr_v, rc_v) = yaw_vectors[local_yaw]
        for lat in (-1, 0, 1):
            for step_d in range(1, 7):
                nr = rob_rc[0] + fr * step_d + rr_v * lat
                nc = rob_rc[1] + fc * step_d + rc_v * lat
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if strong_obs[nr, nc] or obs_counts[nr, nc] >= 15 or log_odds[nr, nc] >= 1.0:
                    break
                if mutable[nr, nc]:
                    log_odds[nr, nc] = max(log_odds[nr, nc] - 0.55, -4.0)

        threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc)
        return disp, pts_sift


def scan_all_directions_stereo(
    controller: Controller,
    stereo_mapper: ClassicalStereoMapper,
    grid: np.ndarray,
    log_odds: np.ndarray,
    visited_free: set,
    rob_rc: tuple[int, int],
    exit_rc: tuple[int, int],
    local_yaw: int
):
    last_disp, last_sift = None, None
    for _ in range(4):
        controller.step(action="RotateRight")
        local_yaw = (local_yaw + 90) % 360
        rgb_left, rgb_right = sync_stereo_rig(controller)

        last_disp, last_sift = stereo_mapper.update_grid_from_stereo(
            grid=grid,
            log_odds=log_odds,
            visited_free=visited_free,
            rob_rc=rob_rc,
            exit_rc=exit_rc,
            local_yaw=local_yaw,
            rgb_left=rgb_left,
            rgb_right=rgb_right
        )
    return last_disp, last_sift, local_yaw


# =====================================================================
# 3. 4-WAY MANHATTAN A* WITH SOFT OBSTACLE CLEARANCE
# =====================================================================
def next_move(grid: np.ndarray):
    rows, cols = grid.shape
    robot_pts = np.argwhere(grid == ROBOT_LOCATION)
    exit_pts  = np.argwhere(grid == EXIT_LOCATION)

    if robot_pts.size == 0 or exit_pts.size == 0:
        return None, []

    start = tuple(int(v) for v in robot_pts[0])
    exits = [tuple(int(v) for v in p) for p in exit_pts]

    # Soft clearance cost: discourages hugging walls/furniture on a 0.25m grid
    # without hard-blocking narrow 0.50m doorways
    blocked_mask = (grid == BLOCKED_CELL)
    near_blocked = np.zeros_like(blocked_mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r0, r1 = max(0, dr), rows + min(0, dr)
            c0, c1 = max(0, dc), cols + min(0, dc)
            near_blocked[r0:r1, c0:c1] |= blocked_mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]

    def heuristic(r, c):
        return min(abs(r - er) + abs(c - ec) for er, ec in exits)

    pq = [(heuristic(*start), 0.0, start)]
    cost_to_reach = {start: 0.0}
    came_from = {}

    while pq:
        _, g_cost, curr = heapq.heappop(pq)

        if curr in exits:
            path = [curr]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()

            if len(path) < 2:
                return None, path

            dr = path[1][0] - start[0]
            dc = path[1][1] - start[1]
            for name, delta in CARDINAL_MOVES.items():
                if delta == (dr, dc):
                    return name, path

        if g_cost > cost_to_reach[curr]:
            continue

        for name, (dr, dc) in CARDINAL_MOVES.items():
            nr, nc = curr[0] + dr, curr[1] + dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols) or grid[neighbor] == BLOCKED_CELL:
                continue

            step_cost = 1.25 if grid[neighbor] == UNKNOWN_CELL else 1.0
            if near_blocked[neighbor] and neighbor not in exits:
                step_cost += 0.90  # Keeps 0.20m robot centered away from corners

            new_cost = g_cost + step_cost
            if new_cost < cost_to_reach.get(neighbor, float("inf")):
                cost_to_reach[neighbor] = new_cost
                came_from[neighbor] = curr
                heapq.heappush(pq, (new_cost + heuristic(nr, nc), new_cost, neighbor))

    return None, []


# =====================================================================
# 4. SELF-TRACKED MOTION EXECUTION (BOUNDS-SAFE)
# =====================================================================
def execute_move(
    controller: Controller,
    move_name: str,
    grid: np.ndarray,
    log_odds: np.ndarray,
    rob_rc: tuple[int, int],
    exit_rc: tuple[int, int],
    local_yaw: int
):
    rows, cols = grid.shape
    target_yaw = MOVE_TO_YAW[move_name]
    diff = (target_yaw - local_yaw) % 360

    if diff == 90:
        controller.step(action="RotateRight")
    elif diff == 180:
        controller.step(action="RotateRight")
        controller.step(action="RotateRight")
    elif diff == 270:
        controller.step(action="RotateLeft")
    local_yaw = target_yaw

    dr, dc = CARDINAL_MOVES[move_name]
    nr, nc = rob_rc[0] + dr, rob_rc[1] + dc

    # Guard against walking off the edge of the allocated local grid
    if not (0 <= nr < rows and 0 <= nc < cols):
        return False, rob_rc, local_yaw

    event = controller.step(action="MoveAhead")
    if event.metadata["lastActionSuccess"]:
        grid[rob_rc] = FREE_CELL
        log_odds[rob_rc] = -4.0
        new_rc = (nr, nc)
        grid[new_rc] = ROBOT_LOCATION
        return True, new_rc, local_yaw
    else:
        if (nr, nc) != exit_rc:
            log_odds[nr, nc] = BUMP_LOCK
            grid[nr, nc] = BLOCKED_CELL
        return False, rob_rc, local_yaw


# =====================================================================
# 5. LIVE 3-PANEL VISUALIZER (AUTO-CROPPED TO EXPLORED REGION)
# =====================================================================
class LiveStereoPlotter:
    def __init__(self, grid: np.ndarray, work_w: int, work_h: int):
        plt.ion()
        self.fig, (self.ax_grid, self.ax_rgb, self.ax_disp) = plt.subplots(1, 3, figsize=(16, 5.5))
        self.cmap = ListedColormap([
            "#2b83ba",  # -2: Robot (Blue)
            "#fdae61",  # -1: Exit (Orange)
            "#ffffff",  #  0: Free (White)
            "#2c3e50",  #  1: Blocked (Dark Slate)
            "#bdc3c7"   #  2: Unknown (Light Gray)
        ])
        self.im_grid = self.ax_grid.imshow(grid, cmap=self.cmap, vmin=-2.5, vmax=2.5)
        self.path_line, = self.ax_grid.plot([], [], color="#00a86b", linewidth=1.8, marker=".", markersize=4,
                                            label="A* Planned Path")
        self.taken_line, = self.ax_grid.plot([], [], color="#d62728", linewidth=2, marker=".", markersize=4,
                                             label="Path Taken")

        self.ax_grid.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        self.ax_grid.grid(which="minor", color="#95a5a6", linestyle="-", linewidth=0.35)
        self.ax_grid.legend(loc="upper left")

        cbar = self.fig.colorbar(self.im_grid, ax=self.ax_grid, ticks=[-2, -1, 0, 1, 2], shrink=0.75)
        cbar.ax.set_yticklabels(["-2: Robot", "-1: Exit", "0: Free", "1: Blocked", "2: Unknown"])

        self.im_rgb = self.ax_rgb.imshow(np.zeros((work_h, work_w, 3), dtype=np.uint8))
        self.ax_rgb.axis("off")
        self.ax_rgb.set_title("Left Head Camera (RGB)")

        self.im_disp = self.ax_disp.imshow(np.zeros((work_h, work_w), dtype=np.float32), cmap="plasma", vmin=0, vmax=64)
        self.ax_disp.axis("off")
        self.ax_disp.set_title("Classical SGBM Disparity")
        plt.tight_layout()

    def update(self, grid, path, rgb_frame, disp_frame, step_idx, status, taken_path=()):
        self.im_grid.set_data(grid)

        # Auto-zoom plot window around non-unknown cells so the large 81x81 grid stays readable
        known_r, known_c = np.nonzero(grid != UNKNOWN_CELL)
        if len(known_r) > 0:
            pad = 4
            self.ax_grid.set_xlim(max(-0.5, known_c.min() - pad), min(grid.shape[1] - 0.5, known_c.max() + pad))
            self.ax_grid.set_ylim(min(grid.shape[0] - 0.5, known_r.max() + pad), max(-0.5, known_r.min() - pad))

        if path:
            self.path_line.set_data([p[1] for p in path], [p[0] for p in path])
        else:
            self.path_line.set_data([], [])
        self.taken_line.set_data([p[1] for p in taken_path], [p[0] for p in taken_path])

        self.ax_grid.set_title(f"Local Stereo Occupancy Grid (Step {step_idx})\n{status}")
        if rgb_frame is not None:
            self.im_rgb.set_data(rgb_frame)
        if disp_frame is not None:
            self.im_disp.set_data(np.clip(disp_frame, 0, 64))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)
if __name__ == "__main__":
    controller, occ_grid, log_odds, rob_rc, exit_rc, pitch_deg = init_environment()

    left0, right0 = sync_stereo_rig(controller)
    stereo_mapper = ClassicalStereoMapper.from_frame(left0, pitch_deg=pitch_deg)
    plotter = LiveStereoPlotter(occ_grid, stereo_mapper.W, stereo_mapper.H)

    # [METRICS] ------------------------------------------------------------
    metrics = MetricsTracker(method_name="classical_stereo", scene=SCENE_NAME,
                             grid_size=GRID_SIZE, half_grid=HALF_GRID_CELLS,
                             depth_range=(0.30, MAX_RANGE))
    metrics.attach(controller, GOAL_OFFSET_M)
    metrics.start(baseline_s=3.0)          # records idle hardware draw first
    sync_stereo_rig            = metrics.wrap("stereo_capture", sync_stereo_rig)
    scan_all_directions_stereo = metrics.wrap("scan_total", scan_all_directions_stereo)
    next_move                  = metrics.wrap_planner(next_move)
    execute_move               = metrics.wrap_executor(execute_move)
    metrics.wrap_mapper(stereo_mapper)
    plotter.update             = metrics.wrap("plotting", plotter.update)
    # ----------------------------------------------------------------------

    local_yaw      = 0
    visited_vision = set()
    visited_free   = {rob_rc}
    taken_path     = [rob_rc]
    last_disp      = None

    retries = 0
    max_steps = 150
    for step in range(max_steps):
        metrics.begin_step(step, occ_grid)                                   # [METRICS]

        if rob_rc not in visited_vision:
            last_disp, _, local_yaw = scan_all_directions_stereo(
                controller, stereo_mapper, occ_grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw
            )
            visited_vision.add(rob_rc)

        move_name, path = next_move(occ_grid)
        frame_work = stereo_mapper.to_work(controller.last_event.frame)
        plotter.update(
            occ_grid, path, frame_work, last_disp, step,
            f"Robot: {rob_rc} -> Goal: {exit_rc} | Next: {move_name}", taken_path
        )

        if move_name is None:
            if retries < 6:
                retries += 1
                metrics.note("replan_retry")                                 # [METRICS]
                weak = (log_odds > 0) & (log_odds < BUMP_LOCK)
                log_odds[weak] *= 0.35
                threshold_grid(occ_grid, log_odds, visited_free, rob_rc, exit_rc)
                print(f"[Step {step}] No path: decaying uncertain obstacles (retry {retries}/6)")
                continue
            print(f"[Step {step}] No valid path found!")
            break
        retries = 0

        success, rob_rc, local_yaw = execute_move(
            controller, move_name, occ_grid, log_odds, rob_rc, exit_rc, local_yaw
        )
        if success:
            visited_free.add(rob_rc)
            taken_path.append(rob_rc)
        print(f"[Step {step:02d}] Move: {move_name:<5} | Success: {str(success):<5} | Cell: {rob_rc} | Steps left: {len(path)-1}")

        if rob_rc == exit_rc:
            print(f"[Step {step}] Reached goal cell {rob_rc}!")
            plotter.update(occ_grid, [], frame_work, last_disp, step, "GOAL REACHED!", taken_path)
            break

        if step == max_steps - 1:
            plotter.update(occ_grid, [], frame_work, last_disp, step, "STEP LIMIT REACHED", taken_path)

    # [METRICS] ------------------------------------------------------------
    metrics.finish(success=(rob_rc == exit_rc), final_rc=rob_rc,
                   exit_rc=exit_rc, grid=occ_grid, max_steps=max_steps)
    # ----------------------------------------------------------------------

    plt.ioff()
    plt.show()
    controller.stop()
