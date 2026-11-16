import heapq
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ai2thor.controller import Controller

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
SCENE_NAME     = "FloorPlan_Train12_5"
GRID_SIZE      = 0.25   # 0.25m step = 1 grid cell
ROBOT_RADIUS   = 0.125  # metres; planner clearance, not simulator collision size
BASELINE       = 0.05  # 5 cm horizontal stereo baseline (right shift)
IMG_W, IMG_H   = 300, 300   # what we ASK the simulator for (it may not honour this, see ClassicalStereoMapper)
WORK_H         = 300        # height the vision pipeline actually runs at (width follows the real aspect ratio)
FOV            = 90.0       # vertical FOV in degrees (Unity convention)
MAX_RANGE      = 2.5        # metres; stereo depth noise grows with z^2, so don't trust far points
BUMP_LOCK      = 4.0        # log-odds written ONLY by a physical bump; vision evidence is capped below this

# Obstacle classification. At 2 m a 0.5 px disparity error moves a floor point by ~0.1 m, so the old 0.12 m
# obstacle threshold sat inside the floor noise, and 12 points per cell is nothing for a dense 580x300 cloud.
OBS_MIN_H      = 0.30       # m above floor: lower than this is treated as floor noise
OBS_MAX_H      = 1.65
FLOOR_BAND     = 0.15       # +/- m around the floor plane counted as floor
MIN_OBS_POINTS = 60         # points in one cell, in one frame, to count as an obstacle hit
HIT_LOGODDS    = 0.7        # one hit is NOT enough to block (blocked at >= 1.0): needs 2 consistent frames

ROBOT_LOCATION = -2
EXIT_LOCATION  = -1
FREE_CELL      = 0
BLOCKED_CELL   = 1
UNKNOWN_CELL   = 2

CARDINAL_MOVES = {
    "up":    (-1,  0),  # +Z (North)
    "right": ( 0,  1),  # +X (East)
    "down":  ( 1,  0),  # -Z (South)
    "left":  ( 0, -1),  # -X (West)
}

MOVE_TO_YAW = {
    "up":    0,
    "right": 90,
    "down":  180,
    "left":  270,
}


# =====================================================================
# 1. ENVIRONMENT & STEREO THIRD-PARTY CAMERA SETUP
# =====================================================================
def get_right_cam_pose(event_meta: dict, baseline: float = BASELINE) -> dict:
    """Uses AI2-THOR's exact `cameraPosition` metadata so there is zero height offset error."""
    cam_pos = event_meta["cameraPosition"]
    agent_rot = event_meta["agent"]["rotation"]
    horizon = event_meta["agent"].get("cameraHorizon", 0.0)

    yaw_rad = math.radians(agent_rot["y"])
    rx = cam_pos["x"] + baseline * math.cos(yaw_rad)
    rz = cam_pos["z"] - baseline * math.sin(yaw_rad)
    ry = cam_pos["y"]

    return dict(
        position=dict(x=rx, y=ry, z=rz),
        rotation=dict(x=horizon, y=agent_rot["y"], z=0.0)
    )


def sync_stereo_pair(controller: Controller) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = controller.last_event.metadata
    cam_pose = get_right_cam_pose(meta, BASELINE)

    event = controller.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        position=cam_pose["position"],
        rotation=cam_pose["rotation"],
        fieldOfView=FOV
    )
    return event.frame, event.third_party_camera_frames[0], event.metadata


def init_environment(render_depth=False):
    # Ground-truth depth is optional and is never used by the stereo mapper.
    controller = Controller(
        scene=SCENE_NAME,
        gridSize=GRID_SIZE,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=render_depth,
        makeAgentsVisible=True,
        width=IMG_W,
        height=IMG_H,
        fieldOfView=FOV,          # FIX: was `fov=`, which is not an AI2-THOR parameter
    )

    # Attach the right-shifted stereo camera
    cam_pose = get_right_cam_pose(controller.last_event.metadata, BASELINE)
    controller.step(
        action="AddThirdPartyCamera",
        position=cam_pose["position"],
        rotation=cam_pose["rotation"],
        fieldOfView=FOV
    )

    # Use GetReachablePositions ONLY to size the grid and pick a reachable exit
    event = controller.step(action="GetReachablePositions")
    reachable = event.metadata["actionReturn"]
    objects   = event.metadata["objects"]

    reach_xz = [(round(p["x"] / GRID_SIZE), round(p["z"] / GRID_SIZE)) for p in reachable]
    obj_xz   = [(round(o["position"]["x"] / GRID_SIZE), round(o["position"]["z"] / GRID_SIZE)) for o in objects]

    xs = [p[0] for p in reach_xz]
    zs = [p[1] for p in reach_xz]
    min_x, max_x = min(xs + [p[0] for p in obj_xz]) - 1, max(xs + [p[0] for p in obj_xz]) + 1
    min_z, max_z = min(zs + [p[1] for p in obj_xz]) - 1, max(zs + [p[1] for p in obj_xz]) + 1

    height = (max_z - min_z) + 1
    width  = (max_x - min_x) + 1

    # Start with entire grid as UNKNOWN_CELL (2) and log-odds = 0.0
    grid     = np.full((height, width), UNKNOWN_CELL, dtype=int)
    log_odds = np.zeros((height, width), dtype=np.float32)

    def to_rc(gx, gz):
        return int(max_z - gz), int(gx - min_x)

    # FIX: exit = farthest REACHABLE cell from the start (the old fallback sat outside the room)
    pos = event.metadata["agent"]["position"]
    far = max(reachable, key=lambda p: math.hypot(p["x"] - pos["x"], p["z"] - pos["z"]))
    exit_rc = to_rc(round(far["x"] / GRID_SIZE), round(far["z"] / GRID_SIZE))
    grid[exit_rc] = EXIT_LOCATION

    # Sim truth, used ONLY for the false-blocked overlay/report, never for planning
    reach_mask = np.zeros(grid.shape, dtype=bool)
    for gx, gz in reach_xz:
        reach_mask[to_rc(gx, gz)] = True

    # Place Robot (-2) at default spawn position
    rob_rc = to_rc(round(pos["x"] / GRID_SIZE), round(pos["z"] / GRID_SIZE))
    grid[rob_rc] = ROBOT_LOCATION

    return controller, grid, log_odds, min_x, max_z, rob_rc, exit_rc, reach_mask



def threshold_grid(grid: np.ndarray, log_odds: np.ndarray, visited_free: set):
    """
    Rewrite `grid` (in place) from log-odds. Robot/exit markers and cells the robot has stood on survive.
    This stores measured occupancy; the planner applies clearance separately.
    """
    special = (grid == ROBOT_LOCATION) | (grid == EXIT_LOCATION)
    saved = grid[special].copy()

    grid[:] = UNKNOWN_CELL
    grid[log_odds <= -0.5] = FREE_CELL
    grid[log_odds >= 1.0] = BLOCKED_CELL
    if visited_free:
        vr, vc = zip(*visited_free)
        grid[list(vr), list(vc)] = FREE_CELL
    grid[special] = saved


# =====================================================================
# 2. CLASSICAL VISION: SIFT + EPIPOLAR + SGBM + 3D FLOOR FILTER
# =====================================================================
class ClassicalStereoMapper:
    def __init__(self, src_w: int, src_h: int, fov=FOV, baseline=BASELINE, work_h=WORK_H):
        """
        src_w/src_h = the ACTUAL frame size coming out of the simulator.
        The pipeline runs on a resized copy that keeps the aspect ratio (square pixels), so the
        tuned parameters (disparity range, thresholds) behave the same whatever the source size.
        Unity's fieldOfView is VERTICAL, so fy comes from the work height and fx == fy.
        """
        self.src_w, self.src_h = src_w, src_h
        self.H = int(work_h)
        self.W = int(round(src_w * self.H / src_h))
        self.baseline = baseline

        fov_rad = math.radians(fov)
        self.fy = (self.H / 2.0) / math.tan(fov_rad / 2.0)
        self.fx = self.fy
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0

        u = np.arange(self.W, dtype=np.float32)
        v = np.arange(self.H, dtype=np.float32)
        self.uu, self.vv = np.meshgrid(u, v)   # shape (H, W): always matches the work frames

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.03)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.num_disp = 48
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    @classmethod
    def from_frame(cls, frame: np.ndarray, **kw):
        h, w = frame.shape[:2]
        return cls(src_w=w, src_h=h, **kw)

    def to_work(self, img: np.ndarray) -> np.ndarray:
        """Resize any frame to the work resolution (no-op if it already matches)."""
        if img.shape[0] == self.H and img.shape[1] == self.W:
            return np.ascontiguousarray(img)
        return np.ascontiguousarray(cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA))

    def compute_clean_disparity(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray:
        """SGBM disparity with Left-Right consistency checking, median filtering and a texture gate."""
        assert gray_l.shape == gray_r.shape == (self.H, self.W), \
            f"gray frames {gray_l.shape}/{gray_r.shape} != mapper work size {(self.H, self.W)}"

        disp_l = self.sgbm_left.compute(gray_l, gray_r).astype(np.float32) / 16.0
        disp_r = -self.sgbm_right.compute(gray_r, gray_l).astype(np.float32) / 16.0

        disp_l = cv2.medianBlur(disp_l, 5)

        # Left-Right consistency: disp_l(u, v) must agree with disp_r(u - d, v)
        u_right = np.clip(np.round(self.uu - disp_l).astype(int), 0, self.W - 1)
        v_idx   = self.vv.astype(int)
        lr_diff = np.abs(disp_l - disp_r[v_idx, u_right])

        # Require local texture so blank drywall doesn't hallucinate depth
        grad_x = np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(gray_l, cv2.CV_32F, 0, 1, ksize=3))
        # Stereo matches a window, so assess texture over that window too. A
        # per-pixel threshold discards the interiors of otherwise valid matches.
        texture = cv2.boxFilter(np.hypot(grad_x, grad_y), -1, (7, 7))

        # Left border where the right camera has no overlap
        valid_border = self.uu >= self.num_disp

        clean_mask = (
            valid_border &
            (disp_l > 1.5) &
            (disp_l < (self.num_disp - 4)) &
            (disp_r[v_idx, u_right] > 0) &
            (lr_diff <= 1.5) &
            (texture > 8.0)
        )

        clean_disp = np.zeros_like(disp_l)
        clean_disp[clean_mask] = disp_l[clean_mask]
        return clean_disp

    def match_sift_epipolar(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray:
        """Sparse 3D points (N, 3) in the camera frame from rectified epipolar SIFT matches."""
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
                    if 1.5 < disp < (self.num_disp - 4):
                        z = (self.fx * self.baseline) / disp
                        if 0.35 <= z <= 4.0:
                            x = (u_l - self.cx) * z / self.fx
                            y = -(v_l - self.cy) * z / self.fy
                            pts_3d.append((x, y, z))
        return np.array(pts_3d, dtype=np.float32) if pts_3d else np.empty((0, 3), dtype=np.float32)

    def update_grid_from_stereo(
        self,
        grid: np.ndarray,
        log_odds: np.ndarray,
        visited_free: set,
        min_x: int,
        max_z: int,
        event_meta: dict,
        rgb_left: np.ndarray,
        rgb_right: np.ndarray,
        max_range: float = MAX_RANGE
    ):
        # Whatever size the simulator sent, bring both frames to the work resolution first
        gray_l = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_left), cv2.COLOR_RGB2GRAY))
        gray_r = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_right), cv2.COLOR_RGB2GRAY))

        # 1. Clean L-R checked disparity + sparse SIFT epipolar anchors
        disp = self.compute_clean_disparity(gray_l, gray_r)
        valid = disp > 0

        z_cam = np.zeros_like(disp)
        z_cam[valid] = (self.fx * self.baseline) / disp[valid]
        valid &= (z_cam >= 0.35) & (z_cam <= max_range)

        x_cam = (self.uu[valid] - self.cx) * z_cam[valid] / self.fx
        y_cam = -(self.vv[valid] - self.cy) * z_cam[valid] / self.fy
        z_pts = z_cam[valid]

        pts_dense = np.stack([x_cam, y_cam, z_pts], axis=-1)
        pts_sift  = self.match_sift_epipolar(gray_l, gray_r)
        if len(pts_sift) > 0:
            pts_sift_r = pts_sift[pts_sift[:, 2] <= max_range]
        else:
            pts_sift_r = pts_sift

        # Weight SIFT landmarks 4x so sharp obstacle edges register strongly
        if len(pts_sift_r) > 0:
            pts_cam = np.vstack([pts_dense] + [pts_sift_r] * 4)
        else:
            pts_cam = pts_dense

        if len(pts_cam) == 0:
            return disp, pts_sift

        # 2. Camera points -> AI2-THOR world coordinates
        cam_pos = event_meta["cameraPosition"]
        yaw_deg = round(event_meta["agent"]["rotation"]["y"]) % 360
        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # Forward (+Z_cam) = (sin_y, cos_y), Right (+X_cam) = (cos_y, -sin_y)
        x_world = cam_pos["x"] + pts_cam[:, 0] * cos_y + pts_cam[:, 2] * sin_y
        z_world = cam_pos["z"] - pts_cam[:, 0] * sin_y + pts_cam[:, 2] * cos_y
        y_world = cam_pos["y"] + pts_cam[:, 1]

        # 3. Floor offset correction (rugs / carpet)
        floor_cands = y_world[(y_world > -0.15) & (y_world < 0.15)]
        floor_y = float(np.median(floor_cands)) if len(floor_cands) > 50 else 0.0
        h_above_floor = y_world - floor_y

        # 4. Obstacle vs floor points
        obs_mask   = (h_above_floor >= OBS_MIN_H) & (h_above_floor <= OBS_MAX_H)
        floor_mask = np.abs(h_above_floor) < FLOOR_BAND

        rows, cols = grid.shape
        obs_counts   = np.zeros((rows, cols), dtype=int)
        floor_counts = np.zeros((rows, cols), dtype=int)

        obs_r = np.round(max_z - (z_world[obs_mask] / GRID_SIZE)).astype(int)
        obs_c = np.round((x_world[obs_mask] / GRID_SIZE) - min_x).astype(int)
        ok = (obs_r >= 0) & (obs_r < rows) & (obs_c >= 0) & (obs_c < cols)
        np.add.at(obs_counts, (obs_r[ok], obs_c[ok]), 1)

        flr_r = np.round(max_z - (z_world[floor_mask] / GRID_SIZE)).astype(int)
        flr_c = np.round((x_world[floor_mask] / GRID_SIZE) - min_x).astype(int)
        ok = (flr_r >= 0) & (flr_r < rows) & (flr_c >= 0) & (flr_c < cols)
        np.add.at(floor_counts, (flr_r[ok], flr_c[ok]), 1)

        # 5. Log-odds update (vectorised). Never touch robot/exit cells or cells the robot has stood on.
        special = (grid == ROBOT_LOCATION) | (grid == EXIT_LOCATION)
        mutable = ~special
        if visited_free:
            vr, vc = zip(*visited_free)
            mutable[list(vr), list(vc)] = False

        strong_obs = obs_counts >= MIN_OBS_POINTS

        hit = mutable & strong_obs
        log_odds[hit] = np.minimum(log_odds[hit] + HIT_LOGODDS, 3.5)

        # Floor seen with no obstacle points -> evidence for free. Applies to ANY vision evidence, so a false
        # obstacle can be undone (only a physical bump, log-odds == BUMP_LOCK, is permanent).
        seen_floor = mutable & ~strong_obs & (floor_counts >= 8) & (obs_counts <= 2) & (log_odds < BUMP_LOCK)
        log_odds[seen_floor] = np.maximum(log_odds[seen_floor] - 0.8, -4.0)

        # 6. Clear a narrow cone ahead of the robot up to the first obstacle
        rob_r, rob_c = (int(v) for v in np.argwhere(grid == ROBOT_LOCATION)[0])
        yaw_vectors = {
            0:   ((-1,  0), ( 0,  1)),
            90:  (( 0,  1), ( 1,  0)),
            180: (( 1,  0), ( 0, -1)),
            270: (( 0, -1), (-1,  0)),
        }
        (fr, fc), (rr, rc) = yaw_vectors[yaw_deg]

        for lat in (-1, 0, 1):
            for step_d in range(1, 7):
                nr = rob_r + fr * step_d + rr * lat
                nc = rob_c + fc * step_d + rc * lat
                if not (0 <= nr < rows and 0 <= nc < cols):
                    break
                if strong_obs[nr, nc] or obs_counts[nr, nc] >= 5 or log_odds[nr, nc] >= 1.0:
                    break
                if mutable[nr, nc]:
                    log_odds[nr, nc] = max(log_odds[nr, nc] - 0.6, -4.0)

        # 7. Threshold log-odds into the planning grid
        threshold_grid(grid, log_odds, visited_free)

        return disp, pts_sift

    def disparity_for_display(self, rgb_left: np.ndarray, rgb_right: np.ndarray) -> np.ndarray:
        """Compute disparity from the same stereo pair shown in the live view."""
        gray_l = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_left), cv2.COLOR_RGB2GRAY))
        gray_r = self.clahe.apply(cv2.cvtColor(self.to_work(rgb_right), cv2.COLOR_RGB2GRAY))
        return self.compute_clean_disparity(gray_l, gray_r)


def scan_all_directions_stereo(
    controller: Controller,
    stereo_mapper: ClassicalStereoMapper,
    grid: np.ndarray,
    log_odds: np.ndarray,
    visited_free: set,
    min_x: int,
    max_z: int
):
    last_disp, last_sift = None, None
    for _ in range(4):
        controller.step(action="RotateRight")
        rgb_left, rgb_right, event_meta = sync_stereo_pair(controller)

        last_disp, last_sift = stereo_mapper.update_grid_from_stereo(
            grid=grid,
            log_odds=log_odds,
            visited_free=visited_free,
            min_x=min_x,
            max_z=max_z,
            event_meta=event_meta,
            rgb_left=rgb_left,
            rgb_right=rgb_right
        )
    return last_disp, last_sift


# =====================================================================
# 3. 4-WAY MANHATTAN A* PATHFINDER
# =====================================================================
def obstacle_clearance_mask(grid: np.ndarray, radius: float = ROBOT_RADIUS) -> np.ndarray:
    """Block robot centers whose circular footprint overlaps an occupied square."""
    if radius < 0:
        raise ValueError("Robot radius must be non-negative")
    blocked = grid == BLOCKED_CELL
    clearance = blocked.copy()
    rows, cols = grid.shape
    reach = math.ceil((radius + GRID_SIZE / 2) / GRID_SIZE)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            dx = max(abs(dc) * GRID_SIZE - GRID_SIZE / 2, 0.0)
            dz = max(abs(dr) * GRID_SIZE - GRID_SIZE / 2, 0.0)
            if dx * dx + dz * dz >= radius * radius or abs(dr) >= rows or abs(dc) >= cols:
                continue
            src_r = slice(max(0, -dr), min(rows, rows - dr))
            src_c = slice(max(0, -dc), min(cols, cols - dc))
            dst_r = slice(max(0, dr), min(rows, rows + dr))
            dst_c = slice(max(0, dc), min(cols, cols + dc))
            clearance[dst_r, dst_c] |= blocked[src_r, src_c]
    return clearance


def next_move(grid: np.ndarray, robot_radius: float = ROBOT_RADIUS):
    rows, cols = grid.shape
    robot_pts = np.argwhere(grid == ROBOT_LOCATION)
    exit_pts  = np.argwhere(grid == EXIT_LOCATION)

    if robot_pts.size == 0 or exit_pts.size == 0:
        return None, []

    start = tuple(int(v) for v in robot_pts[0])
    exits = [tuple(int(v) for v in p) for p in exit_pts]
    clearance = obstacle_clearance_mask(grid, robot_radius)

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

            if not (0 <= nr < rows and 0 <= nc < cols) or clearance[neighbor]:
                continue

            step_cost = 1.25 if grid[neighbor] == UNKNOWN_CELL else 1.0
            new_cost = g_cost + step_cost

            if new_cost < cost_to_reach.get(neighbor, float("inf")):
                cost_to_reach[neighbor] = new_cost
                came_from[neighbor] = curr
                heapq.heappush(pq, (new_cost + heuristic(nr, nc), new_cost, neighbor))

    return None, []


# =====================================================================
# 4. PHYSICAL STEP EXECUTION WITH COLLISION LOCK
# =====================================================================
def execute_move(
    controller: Controller,
    move_name: str,
    grid: np.ndarray,
    log_odds: np.ndarray,
    min_x: int,
    max_z: int
):
    target_yaw = MOVE_TO_YAW[move_name]
    curr_yaw = round(controller.last_event.metadata["agent"]["rotation"]["y"]) % 360
    diff = (target_yaw - curr_yaw) % 360

    if diff == 90:
        controller.step(action="RotateRight")
    elif diff == 180:
        controller.step(action="RotateRight")
        controller.step(action="RotateRight")
    elif diff == 270:
        controller.step(action="RotateLeft")

    rob_r, rob_c = (int(v) for v in np.argwhere(grid == ROBOT_LOCATION)[0])
    event = controller.step(action="MoveAhead")

    if event.metadata["lastActionSuccess"]:
        grid[rob_r, rob_c] = FREE_CELL
        log_odds[rob_r, rob_c] = -4.0  # verified walkable cell

        pos = event.metadata["agent"]["position"]
        new_r = int(max_z - round(pos["z"] / GRID_SIZE))
        new_c = int(round(pos["x"] / GRID_SIZE) - min_x)
        grid[new_r, new_c] = ROBOT_LOCATION
        return True, (new_r, new_c)
    else:
        # Lock the physical collision into log-odds so stereo rays cannot clear it
        dr, dc = CARDINAL_MOVES[move_name]
        br, bc = rob_r + dr, rob_c + dc
        if 0 <= br < grid.shape[0] and 0 <= bc < grid.shape[1]:
            if grid[br, bc] != EXIT_LOCATION:
                log_odds[br, bc] = BUMP_LOCK
                grid[br, bc] = BLOCKED_CELL
        return False, (rob_r, rob_c)


def false_blocked_mask(grid: np.ndarray, reach_mask: np.ndarray) -> np.ndarray:
    """Cells the robot marked BLOCKED although the simulator says the robot can stand there."""
    return (grid == BLOCKED_CELL) & reach_mask


def report_false_blocked(grid, reach_mask, rob_rc, step):
    fb = false_blocked_mask(grid, reach_mask)
    n_blocked, n_reach = int((grid == BLOCKED_CELL).sum()), int(reach_mask.sum())
    msg = (f"    [truth] blocked={n_blocked}  FALSE-BLOCKED={int(fb.sum())} "
           f"({100.0 * fb.sum() / max(n_reach, 1):.0f}% of the {n_reach} standable cells)")
    if fb.any():
        rr, cc = np.nonzero(fb)
        d = np.hypot(rr - rob_rc[0], cc - rob_rc[1]) * GRID_SIZE
        msg += f" | nearest {d.min():.2f} m, farthest {d.max():.2f} m from robot (sensor range {MAX_RANGE} m)"
    print(msg)


# =====================================================================
# 5. LIVE 3-PANEL VISUALIZER (GRID + RGB + SGBM DISPARITY) & MAIN LOOP
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
        self.im_false = self.ax_grid.imshow(np.ma.masked_all(grid.shape), cmap=ListedColormap(["#e74c3c"]),
                                            vmin=0, vmax=1, alpha=0.6)
        self.path_line, = self.ax_grid.plot([], [], color="#00a86b", linewidth=1.5, marker=".", markersize=3,
                                            label="A* Path (green) | red tint = false-blocked")

        self.ax_grid.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        self.ax_grid.grid(which="minor", color="#95a5a6", linestyle="-", linewidth=0.5)
        self.ax_grid.legend(loc="upper left")

        cbar = self.fig.colorbar(self.im_grid, ax=self.ax_grid, ticks=[-2, -1, 0, 1, 2], shrink=0.75)
        cbar.ax.set_yticklabels(["-2: Robot", "-1: Exit", "0: Free", "1: Blocked", "2: Unknown"])

        # Sized from the mapper's work resolution so set_data() never changes the image shape
        self.im_rgb = self.ax_rgb.imshow(np.zeros((work_h, work_w, 3), dtype=np.uint8))
        self.ax_rgb.axis("off")
        self.ax_rgb.set_title("Left Head Camera (RGB)")

        disparity_cmap = plt.get_cmap("plasma").copy()
        disparity_cmap.set_bad("#808080")  # no reliable match, not a measured distance
        self.im_disp = self.ax_disp.imshow(np.ma.masked_all((work_h, work_w)), cmap=disparity_cmap, vmin=1.5, vmax=48)
        self.ax_disp.axis("off")
        self.ax_disp.set_title("SGBM Disparity (gray = no match)")
        self.fig.colorbar(self.im_disp, ax=self.ax_disp, shrink=0.75, label="Disparity (pixels)")
        plt.tight_layout()

    def update(self, grid, path, rgb_frame, disp_frame, step_idx, status, false_mask=None):
        self.im_grid.set_data(grid)
        if false_mask is not None:
            self.im_false.set_data(np.ma.masked_where(~false_mask, np.ones(false_mask.shape)))
        if path:
            self.path_line.set_data([p[1] for p in path], [p[0] for p in path])
        else:
            self.path_line.set_data([], [])

        self.ax_grid.set_title(f"Stereo Occupancy Grid (Step {step_idx})\n{status}")
        if rgb_frame is not None:
            self.im_rgb.set_data(rgb_frame)
        if disp_frame is not None:
            self.im_disp.set_data(np.ma.masked_less_equal(disp_frame, 0))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)


if __name__ == "__main__":
    controller, occ_grid, log_odds, min_x, max_z, rob_rc, exit_rc, reach_mask = init_environment()

    # Build the mapper from the REAL frame size, not from the constants we asked for
    left0, right0, _ = sync_stereo_pair(controller)
    print(f"Requested {IMG_W}x{IMG_H}; simulator returned left {left0.shape[1]}x{left0.shape[0]}, "
          f"right {right0.shape[1]}x{right0.shape[0]}")
    if left0.shape[:2] != right0.shape[:2]:
        print("WARNING: left/right frame sizes differ; the right frame will be resized to match.")
    stereo_mapper = ClassicalStereoMapper.from_frame(left0)
    print(f"Vision pipeline runs at {stereo_mapper.W}x{stereo_mapper.H} (fx=fy={stereo_mapper.fx:.1f})")

    plotter = LiveStereoPlotter(occ_grid, stereo_mapper.W, stereo_mapper.H)

    print(f"Cell {GRID_SIZE} m, planner robot radius {ROBOT_RADIUS} m (simulator collision size unchanged)")

    visited_vision = set()
    visited_free   = {rob_rc}  # cells the robot has physically stood on (guaranteed FREE)
    last_disp = None

    retries = 0
    max_steps = 120
    for step in range(max_steps):
        if rob_rc not in visited_vision:
            last_disp, _ = scan_all_directions_stereo(
                controller, stereo_mapper, occ_grid, log_odds, visited_free, min_x, max_z
            )
            visited_vision.add(rob_rc)

        move_name, path = next_move(occ_grid)
        # Always display a synchronized pair, including when revisiting a cell.
        display_left, display_right, _ = sync_stereo_pair(controller)
        frame_work = stereo_mapper.to_work(display_left)
        last_disp = stereo_mapper.disparity_for_display(display_left, display_right)
        plotter.update(
            occ_grid, path, frame_work, last_disp, step,
            f"Robot: {rob_rc} -> Exit: {exit_rc} | Next: {move_name}",
            false_mask=false_blocked_mask(occ_grid, reach_mask),
        )
        report_false_blocked(occ_grid, reach_mask, rob_rc, step)

        if move_name is None:
            # Boxed in by uncertain vision obstacles: forget them (never bump-locked ones) and retry
            if retries < 5:
                retries += 1
                weak = (log_odds > 0) & (log_odds < BUMP_LOCK)
                log_odds[weak] *= 0.4
                threshold_grid(occ_grid, log_odds, visited_free)
                print(f"[Step {step}] No path: forgetting uncertain obstacles (retry {retries}/5)")
                continue
            print(f"[Step {step}] No valid path found!")
            break
        retries = 0

        success, rob_rc = execute_move(controller, move_name, occ_grid, log_odds, min_x, max_z)
        if success:
            visited_free.add(rob_rc)  # lock verified walkable cell forever
        print(f"[Step {step:02d}] Move: {move_name:<5} | Success: {str(success):<5} | Pos: {rob_rc} | Steps left: {len(path)-1}")

        if rob_rc == exit_rc:
            print(f"[Step {step}] Reached exit at {rob_rc}!")
            display_left, display_right, _ = sync_stereo_pair(controller)
            frame_work = stereo_mapper.to_work(display_left)
            last_disp = stereo_mapper.disparity_for_display(display_left, display_right)
            plotter.update(occ_grid, [], frame_work, last_disp, step, "GOAL REACHED!")
            break

    plt.ioff()
    plt.show()
    controller.stop()
