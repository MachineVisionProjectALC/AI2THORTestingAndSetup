"""
roomtesting.py -- stereo -> 2D occupancy grid -> A*, in AI2-THOR / ProcTHOR (single file).

Two vision pipelines share ONE back end (log-odds grid, A*, dead-reckoning odometry, scoring), so the
runs differ ONLY in how the images are turned into per-cell evidence:

  classical : LAB+CLAHE -> SGBM disparity (texture-gated) -> RANSAC floor plane -> height-threshold evidence
  deep      : RAFT-Stereo disparity
              -> [Depth Anything V2 metric mono depth fills the holes stereo cannot see]      (mono_fill)
              -> [SegFormer-ADE20K floor mask picks the points the floor plane is fitted to]  (seg_ground)
              -> [semantic class + height decide obstacle / floor evidence per pixel]         (seg_evidence)
  Each [bracketed] module has its own switch; see METHODS for the ablation ladder.

PERFECT INFORMATION STILL USED (state in the report) -- identical for both methods:
  * discrete odometry: MoveAhead == exactly one cell, Rotate* == exactly 90 deg (dead reckoning)
  * goal given as a relative (right, ahead) offset from spawn
  * camera intrinsics / baseline (calibration, fine)
PERFECT INFORMATION REMOVED:
  * camera pitch and camera height: estimated per frame from the stereo points (RANSAC plane fit)
    (USE_RANSAC_GROUND=False falls back to the known values = the oracle, for comparison).
Ground truth (reachable positions) is used ONLY for the error panel and the final score.

Deep-side setup (classical needs none of it; torch/transformers are imported lazily):
  pip install torch "transformers>=4.45"
  git clone https://github.com/princeton-vl/RAFT-Stereo      (next to this file; or set RAFT_STEREO_DIR)
  put raftstereo-realtime.pth in RAFT-Stereo/models/         (or set RAFT_STEREO_CKPT)
  SegFormer and Depth Anything V2 weights download from Hugging Face on first run.
"""
import heapq
import math
import re
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ai2thor.controller import Controller

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
SCENE_NAME      = "FloorPlan_Train6_5"
GRID_SIZE       = 0.25
IMG_W, IMG_H    = 300, 300
FOV             = 90.0     # vertical FOV (Unity convention); square image so horizontal is the same
BASELINE        = 0.12
LOOK_DOWN_DEG   = 30       # AI2-THOR's default maximum -> floor blind spot = h / tan(30+45) = 0.42 m
CAMERA_HEIGHT   = 1.576    # ONLY used when USE_RANSAC_GROUND = False (oracle) and for the final printout
MIN_RANGE       = 0.2
MAX_RANGE       = 2.2      # beyond this the stereo depth noise (grows as z^2) swamps the height thresholds
GOAL_OFFSET_M   = (10.00, 10.50)   # (meters right, meters ahead) of spawn
HALF_GRID_CELLS = 40       # 81x81 local grid (+/-10 m)
MAX_STEPS       = 300
ROBOT_RADIUS_M  = 0.20     # evaluation only: GT reachable cells are robot-CENTRE positions (C-space)

# ---- ablation switches ----
USE_CLAHE         = True   # classical only: per-image tiles differ between L and R, so this may hurt matching
USE_RANSAC_GROUND = True

# ---- which pipelines to run, in order (keys of METHODS, defined below the mappers) ----
RUN_METHODS = ("classical", "deep")

# ---- classical stereo ----
NUM_DISP        = 96 # 64
BLOCK           = 7
MIN_TEXTURE     = 10.0     # window-mean |dI/dx|; tune on Train scenes, report on Val
DISP_SIGMA_PX   = 0.4      # assumed disparity noise -> height noise sigma_h(z) = z^2/(f*b) * sigma_d

# ---- RANSAC ground plane (no known pitch / height) ----
RANSAC_ITERS      = 200
RANSAC_ROW_FRAC   = 0.5    # candidates come from the lower half of the image ...
RANSAC_MAX_Z      = 1.8    # ... and are not too far (far floor is noisy)
RANSAC_MAX_PTS    = 2000
RANSAC_MIN_INL    = 150
MIN_UP            = 0.75   # floor normal must be mostly along camera-up (camera tilted down < ~40 deg).
                           # Loose design prior: it rejects wall planes, it does not give the exact pitch.
HEIGHT_RANGE      = (0.3, 3.0)   # plausible camera height [m] for an accepted plane
PLANE_GATE_DEG    = 5.0    # camera mount is rigid -> floor plane in camera coords is constant across frames;
PLANE_GATE_M      = 0.15   # fits are blended (EMA) if consistent, ignored if not (reset after 5 in a row)
PLANE_EMA         = 0.3

# ---- point classification (heights above the estimated floor plane) ----
OBS_MIN_H       = 0.12
OBS_MAX_H       = 1.50
FLOOR_BAND_MIN  = 0.08
K_OBS, K_FLOOR  = 3.0, 2.5   # obstacle: h > K_OBS*sigma_h ; floor: |h| < K_FLOOR*sigma_h
AREA_OBS_M2     = 0.004      # observed obstacle surface per cell (~6% of a cell) -> counts as a hit
AREA_FLOOR_M2   = 0.006      # observed floor per cell (~10% of a cell) -> counts as seen floor

# ---- Bayesian log-odds inverse sensor model (Thrun ch. 9 / OctoMap), updated once per cell per frame ----
def _logit(p): return math.log(p / (1.0 - p))
P_HIT, P_FLOOR, P_RAY, P_CLAMP = 0.82, 0.36, 0.45, 0.97
P_OCC_THR, P_FREE_THR, P_STRICT = 0.80, 0.30, 0.95   # two hits cross P_OCC_THR; one floor view crosses P_FREE_THR
L_HIT, L_FLOOR, L_RAY = _logit(P_HIT), _logit(P_FLOOR), _logit(P_RAY)
L_MAX, L_OCC, L_FREE, L_STRICT = _logit(P_CLAMP), _logit(P_OCC_THR), _logit(P_FREE_THR), _logit(P_STRICT)
BUMP_LOCK       = 4.0    # log-odds written ONLY by a physical collision (> L_MAX, never touched by vision)
VISIT_LOCK      = -4.0   # log-odds of cells the robot has stood on

# ---- deep vision models (only read by DeepStereoMapper) ----
RAFT_PRESET       = "realtime"   # "realtime" (fast) | "middlebury" | "eth3d" (slower, 32 iters)
RAFT_ITERS        = None         # None -> preset default
SEG_MODEL_ID      = "nvidia/segformer-b2-finetuned-ade-512-512"          # b0..b5 exist; bigger = slower
SEG_INPUT         = 512                                                   # short-side resize fed to SegFormer
SEG_FLOOR_NAMES   = ("floor", "rug")        # ADE20K classes treated as walkable ("carpet" is the rug class)
SEG_CEILING_NAMES = ("ceiling",)            # never obstacle evidence
MONO_MODEL_ID     = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"   # Small / Base / Large
MONO_INPUT        = 518                                                   # short-side resize (multiple of 14)

DEEP_STRONG_OBS_H     = 0.30   # a point this far above the floor ALWAYS counts as obstacle (segmentation cannot veto it)
DEEP_FLOOR_MAX_Z      = 1.8    # semantic floor evidence is trusted further out than the geometric one (1.4 m)
DEEP_MONO_WEIGHT      = 0.5    # a mono-filled pixel counts half as much as a stereo pixel in the cell evidence
DEEP_MONO_REL_SIGMA   = 0.04   # mono depth error floor: 4% of range (stereo model is z^2-based)
DEEP_MONO_ALIGN_MIN_PX = 200   # min stereo/mono overlap needed to re-estimate the per-frame mono scale
MONO_VMAX         = 3.0        # plot range of the mono-depth panel [m]

ROBOT_LOCATION  = -2
EXIT_LOCATION   = -1
FREE_CELL       = 0
BLOCKED_CELL    = 1
UNKNOWN_CELL    = 2

CARDINAL_MOVES = {
    "up":    (-1,  0),  # +Z (local forward at spawn)
    "right": ( 0,  1),  # +X
    "down":  ( 1,  0),
    "left":  ( 0, -1),
}
MOVE_TO_YAW = {"up": 0, "right": 90, "down": 180, "left": 270}

# Perception stages whose time is summed into "perception_ms_per_frame". The classical mapper reports
# disparity + mapping only (its ground-plane fit is inside "mapping"); the deep mapper splits it out.
PERCEPTION_STAGES = ("disparity", "mono_depth", "segmentation", "fusion", "ground", "mapping")

TIMES = defaultdict(list)   # stage -> seconds per call


def _no_sync():
    pass


GPU_SYNC = _no_sync         # run_navigation() points this at torch.cuda.synchronize for the deep mapper


@contextmanager
def timed(stage):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        GPU_SYNC()          # make sure queued GPU kernels are finished before the clock stops
        TIMES[stage].append(time.perf_counter() - t0)


# =====================================================================
# 1. HARDWARE RIG SIMULATION (STEREO CAMERA PAIR ONLY)
# =====================================================================
def sync_stereo_rig(controller: Controller, baseline: float = BASELINE):
    """Rigid stereo pair: left = agent camera, right = third-party camera `baseline` to the right."""
    meta = controller.last_event.metadata
    cam_pos = meta["cameraPosition"]
    agent_rot = meta["agent"]["rotation"]
    horizon = meta["agent"].get("cameraHorizon", 0.0)

    yaw_rad = math.radians(agent_rot["y"])
    rx = cam_pos["x"] + baseline * math.cos(yaw_rad)
    rz = cam_pos["z"] - baseline * math.sin(yaw_rad)

    with timed("capture"):
        event = controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=dict(x=rx, y=cam_pos["y"], z=rz),
            rotation=dict(x=horizon, y=agent_rot["y"], z=0.0),
            fieldOfView=FOV,
        )
    return event.frame, event.third_party_camera_frames[0]


def init_environment(render_depth=False):
    controller = Controller(
        scene=SCENE_NAME, gridSize=GRID_SIZE, snapToGrid=True, rotateStepDegrees=90,
        renderDepthImage=render_depth,   # only for the optional depth-error check; slows the simulator
        width=IMG_W, height=IMG_H, fieldOfView=FOV,
    )
    controller.step(action="LookDown", degrees=LOOK_DOWN_DEG)

    meta = controller.last_event.metadata
    cam_pos = meta["cameraPosition"]
    yaw_rad = math.radians(meta["agent"]["rotation"]["y"])
    horizon = meta["agent"].get("cameraHorizon", 0.0)
    if abs(horizon - LOOK_DOWN_DEG) > 1e-3:
        print(f"[warn] LookDown gave horizon {horizon}, asked for {LOOK_DOWN_DEG}")

    controller.step(
        action="AddThirdPartyCamera",
        position=dict(x=cam_pos["x"] + BASELINE * math.cos(yaw_rad), y=cam_pos["y"],
                      z=cam_pos["z"] - BASELINE * math.sin(yaw_rad)),
        rotation=dict(x=horizon, y=meta["agent"]["rotation"]["y"], z=0.0),
        fieldOfView=FOV,
    )

    size = 2 * HALF_GRID_CELLS + 1
    grid     = np.full((size, size), UNKNOWN_CELL, dtype=int)
    log_odds = np.zeros((size, size), dtype=np.float32)
    rob_rc = (HALF_GRID_CELLS, HALF_GRID_CELLS)
    goal_dx, goal_dz = GOAL_OFFSET_M
    exit_rc = (int(np.clip(HALF_GRID_CELLS - round(goal_dz / GRID_SIZE), 1, size - 2)),
               int(np.clip(HALF_GRID_CELLS + round(goal_dx / GRID_SIZE), 1, size - 2)))
    log_odds[rob_rc] = VISIT_LOCK
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION
    return controller, grid, log_odds, rob_rc, exit_rc, float(horizon)


def threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc, strict=False):
    """Discrete planning grid from log-odds. strict=True: BLOCKED only if very confident (no-path fallback)."""
    rows, cols = grid.shape
    grid[:] = UNKNOWN_CELL
    grid[log_odds <= L_FREE] = FREE_CELL
    grid[log_odds >= (L_STRICT if strict else L_OCC)] = BLOCKED_CELL
    if visited_free:
        vr, vc = zip(*[(r, c) for (r, c) in visited_free if 0 <= r < rows and 0 <= c < cols])
        grid[list(vr), list(vc)] = FREE_CELL
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION


# =====================================================================
# 2. GEOMETRY PRIMITIVES
# =====================================================================
def ransac_plane(P, sig, rng):
    """
    Plane n.p + d = 0 (unit n with n_y >= MIN_UP, i.e. 'up' in camera coords): height of p above it = n.p + d,
    and d is the camera height. 3-point hypotheses, depth-aware inlier tolerance, SVD refit on the inliers.
    Returns (n, d) or None.
    """
    i = rng.integers(0, len(P), (RANSAC_ITERS, 3))
    nrm = np.cross(P[i[:, 1]] - P[i[:, 0]], P[i[:, 2]] - P[i[:, 0]])
    ln = np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(ln, 1e-9)[:, None]
    nrm *= np.where(nrm[:, 1] < 0, -1.0, 1.0)[:, None]
    off = -np.einsum("ij,ij->i", nrm, P[i[:, 0]])
    tol = 0.02 + 0.5 * sig
    dist = np.abs(P @ nrm.T + off)                                           # (N, iters)
    score = (dist < tol[:, None]).sum(0) * ((ln > 1e-6) & (nrm[:, 1] >= MIN_UP))
    k = int(np.argmax(score))
    if score[k] < RANSAC_MIN_INL:
        return None
    inl = P[dist[:, k] < tol]
    ctr = inl.mean(0)
    n = np.linalg.svd(inl - ctr, full_matrices=False)[2][-1]
    n = n if n[1] >= 0 else -n
    d = float(-n @ ctr)
    ok = n[1] >= MIN_UP and HEIGHT_RANGE[0] <= d <= HEIGHT_RANGE[1]
    return (n, d) if ok else None


# =====================================================================
# 3.A CLASSICAL VISION: SGBM + RANSAC GROUND + AREA-WEIGHTED EVIDENCE
#     (also the shared back end that the deep mapper inherits)
# =====================================================================
class ClassicalStereoMapper:
    def __init__(self, src_w: int, src_h: int, pitch_deg=None, fov=FOV, baseline=BASELINE, seed=0):
        """pitch_deg is used ONLY if USE_RANSAC_GROUND is False (oracle plane)."""
        self.H, self.W = src_h, src_w
        self.baseline = baseline
        self.oracle_pitch = pitch_deg
        self.fy = (self.H / 2.0) / math.tan(math.radians(fov) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.fb = self.fx * baseline
        self.d_lo, self.d_hi = self.fb / MAX_RANGE, min(self.fb / MIN_RANGE, NUM_DISP - 4)
        self.rng = np.random.default_rng(seed)
        self.plane, self.bad_fits = None, 0
        self.last = {}                                    # diagnostics of the last frame (shown in the plot title)
        self.vis = {}                                     # extra plot panels (the deep mapper fills this)
        self.cuda_sync = _no_sync

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK,
            P1=8 * 3 * BLOCK ** 2, P2=32 * 3 * BLOCK ** 2,
            disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_HH,
        )

    @classmethod
    def from_frame(cls, frame: np.ndarray, **kw):
        h, w = frame.shape[:2]
        return cls(src_w=w, src_h=h, **kw)

    def gpu_peak_mb(self):
        return None

    # ---- disparity -----------------------------------------------------
    def _prep(self, rgb):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        if USE_CLAHE:
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return lab

    def compute_clean_disparity(self, img_l, img_r):
        # OpenCV leaves the left NUM_DISP columns of the left-referenced map invalid; padding recovers them
        # (far points at the left edge have small disparity and match fine).
        pad = NUM_DISP
        L = cv2.copyMakeBorder(img_l, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        R = cv2.copyMakeBorder(img_r, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        d = self.sgbm.compute(L, R)[:, pad:].astype(np.float32) / 16.0
        # a horizontal baseline only constrains horizontal gradients -> gate on window-mean |dI/dx|
        gx = np.abs(cv2.Sobel(img_l, cv2.CV_32F, 1, 0, ksize=3)).max(axis=2)
        tex = cv2.boxFilter(gx, -1, (BLOCK, BLOCK))
        ok = (d >= self.d_lo) & (d <= self.d_hi) & (tex >= MIN_TEXTURE)
        return np.where(ok, d, 0.0).astype(np.float32)

    # ---- geometry ------------------------------------------------------
    def points(self, disp):
        v, u = np.nonzero(disp > 0)
        z = self.fb / disp[v, u].astype(np.float64)
        k = (z >= MIN_RANGE) & (z <= MAX_RANGE)
        u, v, z = u[k], v[k], z[k]
        pts = np.stack([(u - self.cx) * z / self.fx, -(v - self.cy) * z / self.fy, z], axis=1)  # x right, y up, z fwd
        return pts, z, v

    def estimate_ground(self, pts, z, v, sig, cand=None):
        """
        (n, d) of the floor in camera coords. Fits are blended over time because the mount is rigid.
        cand: optional indices of the points to fit (the deep mapper passes the semantic-floor pixels);
              default = lower image half within RANSAC_MAX_Z.
        """
        if not USE_RANSAC_GROUND:                                   # oracle: known pitch + known height
            p = math.radians(self.oracle_pitch)
            return np.array([0.0, math.cos(p), -math.sin(p)]), CAMERA_HEIGHT
        if cand is None:
            cand = np.flatnonzero((v >= RANSAC_ROW_FRAC * self.H) & (z <= RANSAC_MAX_Z))
        fit = None
        if len(cand) >= RANSAC_MIN_INL:
            if len(cand) > RANSAC_MAX_PTS:
                cand = self.rng.choice(cand, RANSAC_MAX_PTS, replace=False)
            fit = ransac_plane(pts[cand], sig[cand], self.rng)
        self.last["ransac_ok"] = fit is not None
        if fit is not None:
            if self.plane is None:
                self.plane = fit
            else:
                n0, d0 = self.plane
                ang = math.degrees(math.acos(float(np.clip(fit[0] @ n0, -1, 1))))
                if ang < PLANE_GATE_DEG and abs(fit[1] - d0) < PLANE_GATE_M:
                    n1 = (1 - PLANE_EMA) * n0 + PLANE_EMA * fit[0]
                    self.plane, self.bad_fits = (n1 / np.linalg.norm(n1), (1 - PLANE_EMA) * d0 + PLANE_EMA * fit[1]), 0
                else:
                    self.bad_fits += 1
                    if self.bad_fits >= 5:
                        self.plane, self.bad_fits = fit, 0
        return self.plane                                           # None until the first successful fit

    def to_local(self, pts, pitch_deg, yaw_deg, rob_xz):
        """Undo the downward pitch, then rotate/translate into the self-tracked local world frame."""
        p, a = math.radians(pitch_deg), math.radians(yaw_deg)
        z_h = pts[:, 1] * math.sin(p) + pts[:, 2] * math.cos(p)
        x_l = pts[:, 0]
        return (rob_xz[0] + x_l * math.cos(a) + z_h * math.sin(a),
                rob_xz[1] - x_l * math.sin(a) + z_h * math.cos(a))

    @staticmethod
    def robot_xz(rob_rc):
        return ((rob_rc[1] - HALF_GRID_CELLS) * GRID_SIZE, (HALF_GRID_CELLS - rob_rc[0]) * GRID_SIZE)

    # ---- evidence ------------------------------------------------------
    def default_masks(self, h, z, sig):
        """Geometry-only point labels: obstacle / floor from height above the estimated plane."""
        obs = (h >= np.maximum(OBS_MIN_H, K_OBS * sig)) & (h <= OBS_MAX_H)
        flr = (h >= -0.12) & (h <= 0.10) & (z <= 1.4)
        return obs, flr

    def cell_evidence(self, grid_shape, x, zl, h, z, sig, w=None, masks=None):
        """
        Per-cell observed obstacle / floor SURFACE AREA in m^2 (each pixel weighs (z/f)^2, so range-invariant).
        w: optional per-point weights (default (z/f)^2); masks: optional (obs, flr) point labels (default: geometry).
        """
        rows, cols = grid_shape
        if w is None:
            w = (z / self.fx) ** 2
        obs, flr = masks if masks is not None else self.default_masks(h, z, sig)
        r = np.round(HALF_GRID_CELLS - zl / GRID_SIZE).astype(int)
        c = np.round(HALF_GRID_CELLS + x / GRID_SIZE).astype(int)
        ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        idx = r * cols + c
        area = lambda m: np.bincount(idx[ok & m], w[ok & m], minlength=rows * cols).reshape(rows, cols)
        a_obs, a_flr = area(obs), area(flr)
        hit = a_obs >= AREA_OBS_M2
        floor = (a_flr >= AREA_FLOOR_M2) & (a_obs < 0.5 * AREA_OBS_M2)
        return hit, floor

    def apply_evidence(self, grid, log_odds, visited_free, rob_rc, exit_rc, hit, floor):
        """Per-frame log-odds update from the per-cell hit / floor flags, then re-threshold the planning grid."""
        mutable = log_odds < BUMP_LOCK                          # collisions are final
        mutable[rob_rc] = mutable[exit_rc] = False
        if visited_free:
            vr, vc = zip(*visited_free)
            mutable[list(vr), list(vc)] = False
        # QUIRK (kept on purpose so the classical run matches the tuned one): floor evidence is applied twice
        # -- once guarded by (log_odds < L_OCC) and once unguarded. The unguarded one also fires on cells that
        # are already confirmed obstacles and can erode them. Remove the third line to test the "fixed" version.
        log_odds[floor & mutable & (log_odds < L_OCC)] += L_FLOOR
        log_odds[hit & mutable] += L_HIT
        log_odds[floor & mutable] += L_FLOOR
        log_odds[mutable] = np.clip(log_odds[mutable], -L_MAX, L_MAX)
        threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc)

    # ---- grid update ---------------------------------------------------
    def update_grid_from_disparity(self, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, disp):
        """Method-agnostic part: any disparity map (0 = invalid) -> log-odds."""
        with timed("mapping"):
            pts, z, v = self.points(disp)
            self.last["n_points"] = len(pts)
            if len(pts) < 50:
                return
            sig = z ** 2 / self.fb * DISP_SIGMA_PX
            plane = self.estimate_ground(pts, z, v, sig)
            if plane is None:                                       # no floor model yet -> skip, never guess
                return
            n, d = plane
            h = pts @ n + d
            pitch_est = math.degrees(math.atan2(-n[2], n[1]))
            self.last.update(pitch_est=pitch_est, cam_h_est=d)

            x, zl = self.to_local(pts, pitch_est, local_yaw, self.robot_xz(rob_rc))
            hit, floor = self.cell_evidence(grid.shape, x, zl, h, z, sig)
            self.apply_evidence(grid, log_odds, visited_free, rob_rc, exit_rc, hit, floor)

    def update_grid_from_stereo(self, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, rgb_left, rgb_right):
        with timed("disparity"):
            disp = self.compute_clean_disparity(self._prep(rgb_left), self._prep(rgb_right))
        self.update_grid_from_disparity(grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, disp)
        return disp


# =====================================================================
# 3.B DEEP VISION: RAFT-STEREO + MONO METRIC DEPTH + SEMANTIC SEGMENTATION
# =====================================================================
class DeepStereoMapper(ClassicalStereoMapper):
    """
    Same geometry, same log-odds back end, same A* as the classical mapper. Only the perception changes:

      RAFT-Stereo        dense disparity from the raw RGB pair (no texture gate, no NUM_DISP cap)
      mono_fill          Depth Anything V2 (metric, indoor) is scale-aligned to stereo every frame and fills ONLY
                         the pixels where stereo cannot match (left image strip, non-finite output). Stereo wins
                         wherever it is valid. Filled pixels count DEEP_MONO_WEIGHT as much as stereo ones.
      seg_ground         SegFormer-ADE20K floor/rug pixels choose the points the floor plane is fitted to
                         (falls back to the classical lower-half candidates if the mask is too small).
      seg_evidence       obstacle / floor labels use class AND height:
                           obstacle = (above floor AND class not floor/rug/ceiling)  OR  (>= DEEP_STRONG_OBS_H above floor)
                           floor    = class floor/rug AND |height| within the depth-noise band (up to DEEP_FLOOR_MAX_Z)
                         The "strong geometric obstacle always counts" clause means a segmentation miss on
                         ProcTHOR's synthetic textures can never hide furniture and cause a collision.

    One-time RAFT-Stereo setup: `git clone https://github.com/princeton-vl/RAFT-Stereo` next to this file and
    put the checkpoint (default raftstereo-realtime.pth) in RAFT-Stereo/models/. If `bash download_models.sh`
    fails, download models.zip by hand from the links in the RAFT-Stereo README (Dropbox / Google Drive).
    Override with the RAFT_STEREO_DIR / RAFT_STEREO_CKPT environment variables.
    """
    # name -> (checkpoint file, architecture flags, default refinement iterations)
    PRESETS = {
        "middlebury": ("raftstereo-middlebury.pth", dict(), 32),
        "eth3d":      ("raftstereo-eth3d.pth", dict(), 32),
        "realtime":   ("raftstereo-realtime.pth",
                       dict(shared_backbone=True, n_downsample=3, n_gru_layers=2, slow_fast_gru=True), 7),
    }

    def __init__(self, *args, seg_ground=True, seg_evidence=True, mono_fill=True,
                 preset: str = RAFT_PRESET, num_iters=RAFT_ITERS, repo_dir=None, device=None, **kw):
        super().__init__(*args, **kw)
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seg_ground, self.seg_evidence, self.mono_fill = seg_ground, seg_evidence, mono_fill
        self.use_seg = seg_ground or seg_evidence
        self.vv, self.uu = [a.astype(np.float32) for a in np.mgrid[0:self.H, 0:self.W]]
        self.d_hi_deep = self.fb / MIN_RANGE            # RAFT has no NUM_DISP cap
        self.mono_scale = 1.0                           # last good stereo/mono scale (reused when overlap is tiny)

        self._load_raft(preset, num_iters, repo_dir)
        if self.use_seg:
            self._load_seg()
        if self.mono_fill:
            self._load_mono()
        self.cuda_sync = (lambda: torch.cuda.synchronize()) if self.device.type == "cuda" else _no_sync
        self._warmup()

    # ---- model loading -------------------------------------------------
    def _load_raft(self, preset, num_iters, repo_dir):
        import os, sys, argparse
        torch = self.torch
        ckpt_name, arch, default_iters = self.PRESETS[preset]
        self.num_iters = num_iters or default_iters

        repo = os.path.abspath(repo_dir or os.environ.get(
            "RAFT_STEREO_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAFT-Stereo")))
        ckpt = os.environ.get("RAFT_STEREO_CKPT") or os.path.join(repo, "models", ckpt_name)
        if not os.path.isdir(os.path.join(repo, "core")):
            raise FileNotFoundError(
                f"RAFT-Stereo repo not found at {repo}. Run `git clone https://github.com/princeton-vl/RAFT-Stereo` "
                "next to roomtesting.py (or set RAFT_STEREO_DIR).")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Checkpoint {ckpt} not found. Download '{ckpt_name}' (models.zip from the RAFT-Stereo README: "
                "Dropbox or Google Drive) and place it there, or set RAFT_STEREO_CKPT to its path.")

        # The repo's modules import each other as `core.*`, so its root must be on sys.path.
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder
        self._InputPadder = InputPadder

        # Same flags as RAFT-Stereo's demo.py (defaults + the preset's architecture overrides).
        use_amp = self.device.type == "cuda"
        model_args = argparse.Namespace(
            hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False,
            corr_levels=4, corr_radius=4, n_downsample=2, context_norm="batch",
            slow_fast_gru=False, n_gru_layers=3, mixed_precision=use_amp)
        for k, v in arch.items():
            setattr(model_args, k, v)

        model = RAFTStereo(model_args)
        state = torch.load(ckpt, map_location=self.device)
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state)
        self.raft = model.to(self.device).eval()
        print(f"[deep] RAFT-Stereo '{preset}' loaded from {ckpt} on {self.device} ({self.num_iters} iters)")

    @staticmethod
    def _class_lut(id2label, names):
        """bool lookup table: class id -> does its ADE20K name (';'-separated synonyms) match one of `names`?"""
        wanted = {s.lower() for s in names}
        lut = np.zeros(max(id2label) + 1, bool)
        for i, name in id2label.items():
            if {t.strip().lower() for t in re.split(r"[;,]", name)} & wanted:
                lut[i] = True
        return lut

    def _norm_consts(self, processor):
        t = self.torch
        mean = t.tensor(processor.image_mean, dtype=t.float32, device=self.device).view(1, 3, 1, 1)
        std = t.tensor(processor.image_std, dtype=t.float32, device=self.device).view(1, 3, 1, 1)
        return mean, std

    def _load_seg(self):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        proc = AutoImageProcessor.from_pretrained(SEG_MODEL_ID)
        self.seg_mean, self.seg_std = self._norm_consts(proc)
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).to(self.device).eval()
        id2label = {int(k): str(v) for k, v in self.seg_model.config.id2label.items()}
        self.seg_floor_lut = self._class_lut(id2label, SEG_FLOOR_NAMES)
        self.seg_ceil_lut = self._class_lut(id2label, SEG_CEILING_NAMES)
        if not self.seg_floor_lut.any():
            raise RuntimeError(f"None of {SEG_FLOOR_NAMES} found in the labels of {SEG_MODEL_ID}")
        print(f"[deep] SegFormer '{SEG_MODEL_ID}' loaded | floor classes: "
              f"{[id2label[i] for i in np.flatnonzero(self.seg_floor_lut)]}")

    def _load_mono(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(MONO_MODEL_ID)
        self.mono_mean, self.mono_std = self._norm_consts(proc)
        self.mono_model = AutoModelForDepthEstimation.from_pretrained(MONO_MODEL_ID).to(self.device).eval()
        print(f"[deep] Depth Anything V2 '{MONO_MODEL_ID}' loaded")

    def _warmup(self):
        """One dummy pass per network so first-call CUDA / cuDNN initialisation does not skew the timings."""
        dummy = np.zeros((self.H, self.W, 3), np.uint8)
        self._raft_raw(dummy, dummy)
        if self.use_seg:
            self._segment(dummy)
        if self.mono_fill:
            self._mono_depth(dummy)
        self.cuda_sync()

    def gpu_peak_mb(self):
        if self.device.type != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated(self.device) / 2 ** 20

    def reset_gpu_peak(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    # ---- network forward passes ----------------------------------------
    def _to_input(self, rgb, target, mult, mean, std):
        """uint8 HxWx3 -> normalised 1x3xhxw tensor, short side resized to `target` (sizes rounded to `mult`)."""
        torch, F = self.torch, self.F
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()[None] / 255.0
        scale = target / min(self.H, self.W)
        size = (max(mult, int(round(self.H * scale / mult)) * mult), max(mult, int(round(self.W * scale / mult)) * mult))
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
        return (x - mean) / std

    def _raft_raw(self, rgb_l, rgb_r):
        """RAFT-Stereo on the raw RGB pair -> float32 HxW disparity (pixels, left-referenced, may be garbage)."""
        torch = self.torch
        to_t = lambda rgb: torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None].to(self.device)
        left, right = to_t(rgb_l), to_t(rgb_r)                 # RAFT-Stereo takes float RGB in [0, 255]
        padder = self._InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        with torch.inference_mode():
            _, flow_up = self.raft(left, right, iters=self.num_iters, test_mode=True)
            flow_up = padder.unpad(flow_up)
        return (-flow_up[0, 0]).float().cpu().numpy()          # RAFT-Stereo outputs negative disparity

    def _segment(self, rgb):
        """SegFormer-ADE20K -> int64 HxW class map at the input resolution."""
        torch, F = self.torch, self.F
        x = self._to_input(rgb, SEG_INPUT, 32, self.seg_mean, self.seg_std)
        with torch.inference_mode():
            logits = self.seg_model(pixel_values=x).logits
            logits = F.interpolate(logits, size=(self.H, self.W), mode="bilinear", align_corners=False)
            lab = logits.argmax(1)[0]
        return lab.cpu().numpy().astype(np.int64)

    def _mono_depth(self, rgb):
        """Depth Anything V2 metric indoor -> float32 HxW depth in metres (treated as z-depth)."""
        torch, F = self.torch, self.F
        x = self._to_input(rgb, MONO_INPUT, 14, self.mono_mean, self.mono_std)
        with torch.inference_mode():
            d = self.mono_model(pixel_values=x).predicted_depth            # (1, h, w)
            d = F.interpolate(d[:, None], size=(self.H, self.W), mode="bilinear", align_corners=False)[0, 0]
        return d.float().cpu().numpy()

    # ---- disparity helpers ---------------------------------------------
    def _prep(self, rgb):
        return np.ascontiguousarray(rgb)                      # the networks normalise internally: no LAB, no CLAHE

    def _stereo_ok(self, d):
        """Pixels whose RAFT disparity is usable: finite, inside the [MIN_RANGE, MAX_RANGE] band, match inside the image."""
        return np.isfinite(d) & (d >= self.d_lo) & (d <= self.d_hi_deep) & ((self.uu - d) >= 0)

    def compute_clean_disparity(self, img_l, img_r):
        d = self._raft_raw(img_l, img_r)
        return np.where(self._stereo_ok(d), d, 0.0).astype(np.float32)

    def _fuse_depth(self, d_raw, z_mono):
        """
        -> (z_map, src, z_mono_aligned). src: 0 = no depth, 1 = stereo, 2 = mono fill.
        Stereo wins wherever valid. Mono is scaled to stereo (median ratio on the overlap, clipped to 0.5..2) and
        used only where stereo has no usable match (non-finite, ~zero, or the match falls outside the right image).
        Pixels that are valid-but-out-of-range for stereo (too far / too near) are NOT filled.
        """
        st_ok = self._stereo_ok(d_raw)
        d = np.where(np.isfinite(d_raw), d_raw, 0.0)
        z_map = np.zeros(d.shape, np.float32)
        z_map[st_ok] = self.fb / d[st_ok]
        src = st_ok.astype(np.uint8)
        if z_mono is None:
            return z_map, src, None

        mono_ok = np.isfinite(z_mono) & (z_mono > 0.1)
        both = st_ok & mono_ok
        if both.sum() >= DEEP_MONO_ALIGN_MIN_PX:
            self.mono_scale = float(np.clip(np.median(z_map[both] / z_mono[both]), 0.5, 2.0))
            rel = np.abs(z_map[both] - self.mono_scale * z_mono[both]) / z_map[both]
            self.last["mono_rel_err"] = float(np.median(rel))         # diagnostic: stereo vs aligned-mono disagreement
        z_m = (self.mono_scale * np.where(mono_ok, z_mono, 0.0)).astype(np.float32)
        hole = ~np.isfinite(d_raw) | (d <= 0.5) | ((self.uu - d) < 0)
        fill = hole & mono_ok & (z_m >= MIN_RANGE) & (z_m <= MAX_RANGE)
        z_map[fill] = z_m[fill]
        src[fill] = 2
        return z_map, src, z_m

    def _classify(self, h, z, sig, floor_pt, ceil_pt):
        """Per-point (obstacle, floor) labels. Geometry only, or geometry + semantic class (seg_evidence)."""
        obs, flr = self.default_masks(h, z, sig)
        if not (self.seg_evidence and floor_pt is not None):
            return obs, flr
        strong = obs & (h >= DEEP_STRONG_OBS_H)                              # geometry alone is enough here
        obs = (obs & ~floor_pt & ~ceil_pt) | strong                          # class vetoes only the marginal ones
        tol = np.maximum(0.10, K_FLOOR * sig)                                # depth-noise-aware floor band
        flr = floor_pt & (np.abs(h) <= tol) & (z <= DEEP_FLOOR_MAX_Z) & ~obs
        return obs, flr

    @staticmethod
    def _overlay(img, floor_img, ceil_img):
        """RGB with the floor class tinted green and the ceiling class tinted blue (plot only)."""
        if floor_img is None:
            return img
        ov = img.astype(np.float32)
        for mask, col in ((floor_img, (0, 200, 60)), (ceil_img, (40, 120, 255))):
            ov[mask] = 0.55 * ov[mask] + 0.45 * np.array(col, np.float32)
        return ov.astype(np.uint8)

    # ---- grid update ---------------------------------------------------
    def update_grid_from_stereo(self, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, rgb_left, rgb_right):
        img_l, img_r = self._prep(rgb_left), self._prep(rgb_right)

        with timed("disparity"):
            d_raw = self._raft_raw(img_l, img_r)
        d_clean = np.where(self._stereo_ok(d_raw), d_raw, 0.0).astype(np.float32)

        z_mono = None
        if self.mono_fill:
            with timed("mono_depth"):
                z_mono = self._mono_depth(img_l)

        floor_img = ceil_img = None
        if self.use_seg:
            with timed("segmentation"):
                lab = self._segment(img_l)
            floor_img, ceil_img = self.seg_floor_lut[lab], self.seg_ceil_lut[lab]

        ev_img = np.zeros((self.H, self.W), np.uint8)                       # plot only: 1 = floor, 2 = obstacle
        self.vis = dict(seg_overlay=self._overlay(img_l, floor_img, ceil_img), mono_depth=None, evidence=ev_img)

        with timed("fusion"):
            z_map, src, z_m = self._fuse_depth(d_raw, z_mono)
            self.vis["mono_depth"] = z_m
            v_idx, u_idx = np.nonzero(src > 0)
            z = z_map[v_idx, u_idx].astype(np.float64)
            pts = np.stack([(u_idx - self.cx) * z / self.fx, -(v_idx - self.cy) * z / self.fy, z], axis=1)
            from_mono = src[v_idx, u_idx] == 2
            sig = z ** 2 / self.fb * DISP_SIGMA_PX
            sig[from_mono] = np.maximum(sig[from_mono], DEEP_MONO_REL_SIGMA * z[from_mono])
            floor_pt = floor_img[v_idx, u_idx] if floor_img is not None else None
            ceil_pt = ceil_img[v_idx, u_idx] if ceil_img is not None else None
        self.last.update(n_points=len(pts), n_mono=int(from_mono.sum()))
        if len(pts) < 50:
            return d_clean

        # floor plane: stereo points only (mono scale error must not bias it)
        with timed("ground"):
            st = ~from_mono
            cand, ground_src = None, "rows"
            if self.seg_ground and floor_pt is not None:
                c = np.flatnonzero(floor_pt[st] & (z[st] <= RANSAC_MAX_Z))
                if len(c) >= RANSAC_MIN_INL:
                    cand, ground_src = c, "seg"
                else:
                    ground_src = "rows (seg mask too small)"
            plane = self.estimate_ground(pts[st], z[st], v_idx[st], sig[st], cand=cand)
        self.last["ground_src"] = ground_src
        if plane is None:                                                   # no floor model yet -> skip, never guess
            return d_clean
        n, d = plane
        h = pts @ n + d
        pitch_est = math.degrees(math.atan2(-n[2], n[1]))
        self.last.update(pitch_est=pitch_est, cam_h_est=d)

        with timed("mapping"):
            x, zl = self.to_local(pts, pitch_est, local_yaw, self.robot_xz(rob_rc))
            obs, flr = self._classify(h, z, sig, floor_pt, ceil_pt)
            w = (z / self.fx) ** 2
            w[from_mono] *= DEEP_MONO_WEIGHT
            ev_img[v_idx[flr], u_idx[flr]] = 1
            ev_img[v_idx[obs], u_idx[obs]] = 2
            hit, floor = self.cell_evidence(grid.shape, x, zl, h, z, sig, w=w, masks=(obs, flr))
            self.apply_evidence(grid, log_odds, visited_free, rob_rc, exit_rc, hit, floor)
        return d_clean


# ---- the pipelines that can be run (RUN_METHODS picks from these; the deep rows form an ablation ladder) ----
METHODS = {
    "classical":     dict(kind="classical", label="Classical SGBM"),
    "deep_raft":     dict(kind="deep", label="RAFT-Stereo only",
                          seg_ground=False, seg_evidence=False, mono_fill=False),
    "deep_segfloor": dict(kind="deep", label="RAFT + seg floor plane",
                          seg_ground=True, seg_evidence=False, mono_fill=False),
    "deep_segevid":  dict(kind="deep", label="RAFT + seg floor + seg evidence",
                          seg_ground=True, seg_evidence=True, mono_fill=False),
    "deep":          dict(kind="deep", label="Full deep (RAFT + seg + mono fusion)",
                          seg_ground=True, seg_evidence=True, mono_fill=True),
}


def observe_forward(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw):
    """One stereo frame in the current heading. Used after every move / turn (no 360 spin)."""
    rgb_l, rgb_r = sync_stereo_rig(controller)
    return mapper.update_grid_from_stereo(grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, rgb_l, rgb_r)


def scan_all_directions_stereo(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw):
    """4 views, back to the original heading. Only at spawn and when the planner is stuck."""
    disp = None
    for _ in range(4):
        disp = observe_forward(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw)
        controller.step(action="RotateRight")
        local_yaw = (local_yaw + 90) % 360
    return disp, local_yaw


# =====================================================================
# 4. 4-WAY MANHATTAN A* WITH SOFT OBSTACLE CLEARANCE   (shared, identical for every method)
# =====================================================================
def next_move(grid: np.ndarray):
    rows, cols = grid.shape
    robot_pts = np.argwhere(grid == ROBOT_LOCATION)
    exit_pts  = np.argwhere(grid == EXIT_LOCATION)
    if robot_pts.size == 0 or exit_pts.size == 0:
        return None, []
    start, goal = tuple(int(v) for v in robot_pts[0]), tuple(int(v) for v in exit_pts[0])

    # soft clearance: discourages hugging walls on a 0.25 m grid without hard-blocking 0.5 m doorways
    near_blocked = cv2.dilate((grid == BLOCKED_CELL).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    heuristic = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    pq = [(heuristic(start), 0.0, start)]
    cost_to_reach, came_from = {start: 0.0}, {}
    while pq:
        _, g_cost, curr = heapq.heappop(pq)
        if curr == goal:
            path = [curr]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            if len(path) < 2:
                return None, path
            delta = (path[1][0] - start[0], path[1][1] - start[1])
            return next(k for k, v in CARDINAL_MOVES.items() if v == delta), path
        if g_cost > cost_to_reach[curr]:
            continue
        for dr, dc in CARDINAL_MOVES.values():
            nb = (curr[0] + dr, curr[1] + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid[nb] == BLOCKED_CELL:
                continue
            step = 1.25 if grid[nb] == UNKNOWN_CELL else 1.0
            if near_blocked[nb] and nb != goal:
                step += 0.90
            if g_cost + step < cost_to_reach.get(nb, float("inf")):
                cost_to_reach[nb] = g_cost + step
                came_from[nb] = curr
                heapq.heappush(pq, (g_cost + step + heuristic(nb), g_cost + step, nb))
    return None, []


# =====================================================================
# 5. SELF-TRACKED MOTION EXECUTION
# =====================================================================
def turn_toward(controller, move_name, local_yaw):
    """One 90-degree turn toward the move direction (a 180 takes two calls, so we never turn blind twice)."""
    diff = (MOVE_TO_YAW[move_name] - local_yaw) % 360
    if diff == 270:
        controller.step(action="RotateLeft");  return (local_yaw - 90) % 360
    controller.step(action="RotateRight");     return (local_yaw + 90) % 360


def execute_move(controller, move_name, grid, log_odds, rob_rc, exit_rc, local_yaw):
    """Caller has already turned to face `move_name`."""
    rows, cols = grid.shape
    dr, dc = CARDINAL_MOVES[move_name]
    nr, nc = rob_rc[0] + dr, rob_rc[1] + dc
    if not (0 <= nr < rows and 0 <= nc < cols):
        return False, rob_rc, local_yaw
    with timed("capture"):
        event = controller.step(action="MoveAhead")
    if event.metadata["lastActionSuccess"]:
        grid[rob_rc] = FREE_CELL
        log_odds[rob_rc] = VISIT_LOCK
        grid[nr, nc] = ROBOT_LOCATION
        return True, (nr, nc), local_yaw
    log_odds[nr, nc] = BUMP_LOCK            # hard evidence from a physical collision
    grid[nr, nc] = BLOCKED_CELL
    return False, rob_rc, local_yaw


# =====================================================================
# 6. EVALUATION (ground truth; evaluation + visualisation only, never seen by the navigator)
# =====================================================================
def _to_local(spawn, x, z):
    sx, sz, yaw = spawn
    dx, dz = x - sx, z - sz
    return dx * math.cos(yaw) - dz * math.sin(yaw), dx * math.sin(yaw) + dz * math.cos(yaw)


def capture_gt(controller):
    """gt_free[r, c]: robot centre can stand there (AI2-THOR reachable positions). Call right after spawn."""
    ag = controller.last_event.metadata["agent"]
    spawn = (ag["position"]["x"], ag["position"]["z"], math.radians(ag["rotation"]["y"]))
    reach = controller.step(action="GetReachablePositions").metadata["actionReturn"] or []
    n = 2 * HALF_GRID_CELLS + 1
    free = np.zeros((n, n), bool)
    for p in reach:
        lx, lz = _to_local(spawn, p["x"], p["z"])
        r, c = round(HALF_GRID_CELLS - lz / GRID_SIZE), round(HALF_GRID_CELLS + lx / GRID_SIZE)
        if 0 <= r < n and 0 <= c < n:
            free[r, c] = True
    return dict(spawn=spawn, gt_free=free)


def _eval_masks(grid):
    free_p = np.isin(grid, (FREE_CELL, ROBOT_LOCATION, EXIT_LOCATION))
    blk = grid == BLOCKED_CELL
    return free_p, blk, blk, free_p | blk


# categories of the error map
E_UNK, E_UNK_REACH, E_TRUE_FREE, E_TRUE_BLK, E_FALSE_FREE, E_PHANTOM = range(6)


def error_map(grid, gt_free):
    free_p, blk, blk_cs, known = _eval_masks(grid)
    pred_blk, pred_free = blk_cs & known, known & ~(blk_cs & known)
    e = np.full(grid.shape, E_UNK, int)
    e[~known & gt_free] = E_UNK_REACH
    e[pred_free & gt_free] = E_TRUE_FREE
    e[pred_blk & ~gt_free] = E_TRUE_BLK
    e[pred_free & ~gt_free] = E_FALSE_FREE      # believes free, robot cannot stand there -> collision
    e[pred_blk & gt_free] = E_PHANTOM           # believes blocked, reachable -> detour / dead end
    return e


def grid_quality(grid, gt_free):
    """Only cells the method labelled are scored (UNKNOWN is not penalised); coverage is reported separately."""
    e = error_map(grid, gt_free)
    tp, fp = int((e == E_TRUE_BLK).sum()), int((e == E_PHANTOM).sum())
    fn, tn = int((e == E_FALSE_FREE).sum()), int((e == E_TRUE_FREE).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    free_p = _eval_masks(grid)[0]
    return dict(blocked_precision=prec, blocked_recall=rec, blocked_f1=div(2 * prec * rec, prec + rec),
                false_free_rate=div(fn, fn + tn),     # of predicted-FREE cells, share the robot cannot stand on
                phantom_rate=div(fp, tp + fp),        # = 1 - precision
                coverage=float((free_p & gt_free).sum() / max(int(gt_free.sum()), 1)))


def bfs_cells(free, start, goal):
    """Shortest 4-connected path in cells on the GT free mask (same action space as the robot)."""
    if not (free[start] and free[goal]):
        return None
    dist, q = {start: 0}, deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return dist[cur]
        for dr, dc in CARDINAL_MOVES.values():
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < free.shape[0] and 0 <= nb[1] < free.shape[1] and free[nb] and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return None


def score_episode(controller, gt, grid, exit_rc, moves_ok, collisions, rotations, frames, steps):
    pos = controller.last_event.metadata["agent"]["position"]
    lx, lz = _to_local(gt["spawn"], pos["x"], pos["z"])
    goal_x, goal_z = (exit_rc[1] - HALF_GRID_CELLS) * GRID_SIZE, (HALF_GRID_CELLS - exit_rc[0]) * GRID_SIZE
    success = math.hypot(lx - goal_x, lz - goal_z) <= 0.5 * GRID_SIZE        # GT pose, not dead reckoning
    L = moves_ok * GRID_SIZE
    cells = bfs_cells(gt["gt_free"], (HALF_GRID_CELLS, HALF_GRID_CELLS), exit_rc)
    L_star = None if cells is None else cells * GRID_SIZE
    spl = float(success) * L_star / max(L, L_star) if L_star else None
    out = dict(success=success, path_m=L, optimal_m=L_star, spl=spl, collisions=collisions, rotations=rotations,
               nav_steps=steps, frames=frames, **grid_quality(grid, gt["gt_free"]))
    for stage in PERCEPTION_STAGES + ("planning",):
        if TIMES[stage]:
            out[f"{stage}_ms_per_call"] = 1e3 * float(np.mean(TIMES[stage]))
    out["perception_ms_per_frame"] = 1e3 * sum(sum(TIMES[s]) for s in PERCEPTION_STAGES) / max(frames, 1)
    return out


# =====================================================================
# 7. LIVE PLOT: occupancy grid | grid vs ground truth | left camera | disparity   (+ 3 deep panels)
# =====================================================================
class LiveStereoPlotter:
    def __init__(self, grid, work_w, work_h, gt_free=None, method_label="Classical SGBM",
                 deep=False, seg_on=False, mono_on=False):
        plt.ion()
        self.gt_free, self.deep, self.method_label = gt_free, deep, method_label
        self.disp_name = "RAFT-Stereo" if deep else "SGBM"
        if deep:        # grid | error | disparity  /  RGB+seg | mono depth | per-pixel evidence
            self.fig, ((self.ax_grid, self.ax_err, self.ax_disp), (self.ax_rgb, self.ax_mono, self.ax_ev)) = \
                plt.subplots(2, 3, figsize=(19, 11.5))
        else:
            self.fig, ((self.ax_grid, self.ax_err), (self.ax_rgb, self.ax_disp)) = plt.subplots(2, 2, figsize=(13, 11.5))
        if self.fig.canvas.manager:
            self.fig.canvas.manager.set_window_title(f"{method_label} run")
        self.ax_err.sharex(self.ax_grid)
        self.ax_err.sharey(self.ax_grid)

        self.cmap = ListedColormap(["#2b83ba", "#fdae61", "#ffffff", "#2c3e50", "#bdc3c7"])   # robot, exit, free, blocked, unknown
        self.im_grid = self.ax_grid.imshow(grid, cmap=self.cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
        self.path_line, = self.ax_grid.plot([], [], color="#00a86b", linewidth=1.8, marker=".", markersize=4, label="A* planned path")
        self.taken_line, = self.ax_grid.plot([], [], color="#d62728", linewidth=2, marker=".", markersize=4, label="Path taken")
        self.ax_grid.legend(loc="upper left", fontsize=8)
        cbar = self.fig.colorbar(self.im_grid, ax=self.ax_grid, ticks=[-2, -1, 0, 1, 2], shrink=0.7)
        cbar.ax.set_yticklabels(["-2: Robot", "-1: Exit", "0: Free", "1: Blocked", "2: Unknown"])

        err_colors = ["#d9d9d9", "#c6dbef", "#b8e6b8", "#2c3e50", "#e41a1c", "#ff9f1c"]
        self.err_cmap = ListedColormap(err_colors)
        self.im_err = self.ax_err.imshow(np.zeros(grid.shape, int), cmap=self.err_cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
        self.ax_err.legend(handles=[Patch(color=c, label=l) for c, l in zip(
            err_colors, ["unknown", "reachable, not yet mapped", "free: correct", "blocked: correct",
                         "FALSE-FREE (collision risk)", "PHANTOM (blocked, reachable)"])], loc="upper left", fontsize=8)
        for ax in (self.ax_grid, self.ax_err):
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#95a5a6", linestyle="-", linewidth=0.3)
        if gt_free is None:
            self.ax_err.axis("off")

        self.im_rgb = self.ax_rgb.imshow(np.zeros((work_h, work_w, 3), dtype=np.uint8))
        self.ax_rgb.axis("off")
        self.ax_rgb.set_title("Left camera + segmentation (green = floor, blue = ceiling)" if (deep and seg_on)
                              else "Left head camera (RGB)", fontsize=10)
        self.im_disp = self.ax_disp.imshow(np.zeros((work_h, work_w), dtype=np.float32), cmap="plasma", vmin=0, vmax=NUM_DISP)
        self.ax_disp.axis("off")
        self.ax_disp.set_title(f"{self.disp_name} disparity (black = rejected)")

        if deep:
            self.im_mono = self.ax_mono.imshow(np.zeros((work_h, work_w), dtype=np.float32), cmap="viridis", vmin=0, vmax=MONO_VMAX)
            self.ax_mono.axis("off")
            self.ax_mono.set_title("Monocular metric depth, scale-aligned to stereo [m]" if mono_on
                                   else "Monocular depth (disabled in this run)", fontsize=10)
            ev_colors = ["#d9d9d9", "#2ca02c", "#d62728"]
            self.im_ev = self.ax_ev.imshow(np.zeros((work_h, work_w), int), cmap=ListedColormap(ev_colors),
                                           vmin=-0.5, vmax=2.5, interpolation="nearest")
            self.ax_ev.axis("off")
            self.ax_ev.set_title("Per-pixel evidence fed to the grid")
            self.ax_ev.legend(handles=[Patch(color=c, label=l) for c, l in zip(
                ev_colors, ["none", "floor", "obstacle"])], loc="lower right", fontsize=8)
        plt.tight_layout()

    def update(self, grid, path, rgb_frame, disp_frame, step_idx, status, taken_path=(), extra="", vis=None):
        self.im_grid.set_data(grid)
        known_r, known_c = np.nonzero(grid != UNKNOWN_CELL)                 # auto-zoom: 81x81 stays readable
        if len(known_r):
            pad = 4
            self.ax_grid.set_xlim(max(-0.5, known_c.min() - pad), min(grid.shape[1] - 0.5, known_c.max() + pad))
            self.ax_grid.set_ylim(min(grid.shape[0] - 0.5, known_r.max() + pad), max(-0.5, known_r.min() - pad))
        self.path_line.set_data([p[1] for p in path], [p[0] for p in path])
        self.taken_line.set_data([p[1] for p in taken_path], [p[0] for p in taken_path])
        self.ax_grid.set_title(f"Occupancy grid - {self.method_label} (step {step_idx})\n{status}", fontsize=10)
        if self.gt_free is not None:
            self.im_err.set_data(error_map(grid, self.gt_free))
            q = grid_quality(grid, self.gt_free)
            self.ax_err.set_title(f"Grid vs ground truth (robot-radius C-space)\nprec={q['blocked_precision']:.2f}  "
                                  f"false-free={q['false_free_rate']:.2f}  coverage={q['coverage']:.2f}", fontsize=10)

        rgb_to_show = rgb_frame
        if self.deep and vis:
            if vis.get("seg_overlay") is not None:
                rgb_to_show = vis["seg_overlay"]            # the frame the networks actually processed
            if vis.get("mono_depth") is not None:
                self.im_mono.set_data(np.clip(vis["mono_depth"], 0, MONO_VMAX))
            if vis.get("evidence") is not None:
                self.im_ev.set_data(vis["evidence"])
        if rgb_to_show is not None:
            self.im_rgb.set_data(rgb_to_show)
        if disp_frame is not None:
            self.im_disp.set_data(np.clip(disp_frame, 0, NUM_DISP))
        if extra:
            self.ax_disp.set_title(f"{self.disp_name} disparity (black = rejected)\n{extra}", fontsize=9)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def save(self, path="occupancy_final.png"):
        self.fig.savefig(path, dpi=140)
        print(f"[plot] saved {path}")


# =====================================================================
# 8. ONE FULL NAVIGATION EPISODE WITH A CHOSEN VISION METHOD
# =====================================================================
def run_navigation(method_key):
    """Runs one episode (fresh simulator, fresh mapper, fresh TIMES) and returns the results dict."""
    global GPU_SYNC
    spec = METHODS[method_key]
    label = spec["label"]
    print(f"\n{'=' * 72}\n RUN: {label}  [{method_key}]\n{'=' * 72}")

    controller, occ_grid, log_odds, rob_rc, exit_rc, pitch_deg = init_environment()
    try:
        gt = capture_gt(controller)                      # evaluation + error panel only

        left0, _ = sync_stereo_rig(controller)
        if spec["kind"] == "deep":
            mapper = DeepStereoMapper.from_frame(left0, pitch_deg=pitch_deg, seg_ground=spec["seg_ground"],
                                                 seg_evidence=spec["seg_evidence"], mono_fill=spec["mono_fill"])
        else:
            mapper = ClassicalStereoMapper.from_frame(left0, pitch_deg=pitch_deg)   # pitch only if USE_RANSAC_GROUND=False
        deep = spec["kind"] == "deep"
        plotter = LiveStereoPlotter(occ_grid, mapper.W, mapper.H, gt["gt_free"], method_label=label, deep=deep,
                                    seg_on=deep and mapper.use_seg, mono_on=deep and mapper.mono_fill)
        GPU_SYNC = mapper.cuda_sync
        if deep:
            mapper.reset_gpu_peak()                      # peak GPU memory then covers weights + one episode
        TIMES.clear()

        local_yaw    = 0
        visited_free = {rob_rc}
        taken_path   = [rob_rc]
        frames = moves_ok = collisions = rotations = steps = 0

        last_disp, local_yaw = scan_all_directions_stereo(controller, mapper, occ_grid, log_odds, visited_free,
                                                          rob_rc, exit_rc, local_yaw)
        frames += 4; rotations += 4
        scanned = {rob_rc}

        for step in range(MAX_STEPS):
            steps = step + 1
            if rob_rc == exit_rc:
                break
            with timed("planning"):
                move_name, path = next_move(occ_grid)
                if move_name is None and not path:       # no path: plan on confident obstacles only (does NOT erase map)
                    strict = occ_grid.copy()
                    threshold_grid(strict, log_odds, visited_free, rob_rc, exit_rc, strict=True)
                    move_name, path = next_move(strict)
            frame_now = controller.last_event.frame
            ex = mapper.last
            info = ""
            if "pitch_est" in ex:
                info = f"floor plane: pitch={ex['pitch_est']:.1f} deg, cam h={ex['cam_h_est']:.2f} m"
                if "ground_src" in ex:
                    info += f" [{ex['ground_src']}]"
                if "mono_rel_err" in ex and deep and mapper.mono_fill:
                    info += f" | mono-vs-stereo rel.err={ex['mono_rel_err']:.2f}"
            plotter.update(occ_grid, path, frame_now, last_disp, step,
                           f"Robot {rob_rc} -> Goal {exit_rc} | next: {move_name}", taken_path, info, vis=mapper.vis)

            if move_name is None:
                if rob_rc not in scanned:                # stuck: one 360 scan per cell
                    last_disp, local_yaw = scan_all_directions_stereo(controller, mapper, occ_grid, log_odds,
                                                                      visited_free, rob_rc, exit_rc, local_yaw)
                    frames += 4; rotations += 4
                    scanned.add(rob_rc)
                    continue
                print(f"[Step {step}] No valid path found!")
                break

            if MOVE_TO_YAW[move_name] != local_yaw:      # turn, LOOK, then re-plan (never move after a blind turn)
                local_yaw = turn_toward(controller, move_name, local_yaw)
                rotations += 1
                last_disp = observe_forward(controller, mapper, occ_grid, log_odds, visited_free,
                                            rob_rc, exit_rc, local_yaw)
                frames += 1
                continue

            success, rob_rc, local_yaw = execute_move(controller, move_name, occ_grid, log_odds, rob_rc, exit_rc, local_yaw)
            print(f"[Step {step:03d}] {move_name:<5} ok={str(success):<5} cell={rob_rc} steps left={len(path) - 1}")
            if success:
                moves_ok += 1
                visited_free.add(rob_rc)
                taken_path.append(rob_rc)
                last_disp = observe_forward(controller, mapper, occ_grid, log_odds, visited_free,
                                            rob_rc, exit_rc, local_yaw)
                frames += 1
            else:
                collisions += 1

        threshold_grid(occ_grid, log_odds, visited_free, rob_rc, exit_rc)
        plotter.update(occ_grid, [], controller.last_event.frame, last_disp, steps,
                       "GOAL REACHED!" if rob_rc == exit_rc else "STOPPED (no path / step limit)", taken_path,
                       vis=mapper.vis)
        plotter.save(f"occupancy_final_{method_key}.png")

        res = score_episode(controller, gt, occ_grid, exit_rc, moves_ok, collisions, rotations, frames, steps)
        peak = mapper.gpu_peak_mb()
        if peak is not None:
            res["gpu_peak_mb"] = peak
        print(f"\n===== RESULT: {label} =====")
        for k, v in res.items():
            print(f"  {k:<26}{v:.3f}" if isinstance(v, float) else f"  {k:<26}{v}")
        ex = mapper.last
        if "pitch_est" in ex:
            print(f"  floor-plane estimate: pitch {ex['pitch_est']:.2f} deg vs true {pitch_deg:.1f} | "
                  f"cam height {ex['cam_h_est']:.3f} m vs {CAMERA_HEIGHT} m")
        return res
    finally:
        GPU_SYNC = _no_sync
        controller.stop()


COMPARE_ROWS = ("success", "path_m", "optimal_m", "spl", "collisions", "rotations", "nav_steps", "frames",
                "blocked_precision", "blocked_recall", "blocked_f1", "false_free_rate", "phantom_rate", "coverage",
                "perception_ms_per_frame", "disparity_ms_per_call", "mono_depth_ms_per_call",
                "segmentation_ms_per_call", "fusion_ms_per_call", "ground_ms_per_call", "mapping_ms_per_call",
                "planning_ms_per_call", "gpu_peak_mb")


def print_comparison(results):
    """Side-by-side table, one column per method that was run."""
    if len(results) < 2:
        return
    fmt = lambda v: ("-" if v is None else ("nan" if isinstance(v, float) and math.isnan(v)
                     else (f"{v:.3f}" if isinstance(v, float) else str(v))))
    keys = list(results)
    col = max(14, max(len(k) for k in keys)) + 2
    print(f"\n{'=' * (28 + col * len(keys))}\n COMPARISON\n{'=' * (28 + col * len(keys))}")
    print(f"  {'':<26}" + "".join(f"{k:>{col}}" for k in keys))
    for row in COMPARE_ROWS:
        if all(row not in r for r in results.values()):
            continue
        print(f"  {row:<26}" + "".join(f"{fmt(results[k].get(row)):>{col}}" for k in keys))
    print("  (classical 'mapping' time includes its RANSAC ground fit; compare perception_ms_per_frame)")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    all_results = {}
    for key in RUN_METHODS:                  # each run starts only after the previous one has fully finished
        all_results[key] = run_navigation(key)
    print_comparison(all_results)

    plt.ioff()
    plt.show()                               # keep every live figure open until the user closes them
