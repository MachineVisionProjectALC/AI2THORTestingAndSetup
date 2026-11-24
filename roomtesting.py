# Classical CV algorithms for robot navigation around a RoboTHOR scene
# other file allows us to test on 15 scenes and then export to pdf
# this is the main file with algorithms

"""
PERFECT INFORMATION STILL USED:
  * MoveAhead == exactly one cell, Rotate* == exactly 90 deg (dead reckoning)
  * goal given as a relative (right, ahead) offset from spawn
  * camera intrinsics / baseline (calibration, fine)
PERFECT INFORMATION REMOVED:
  * camera pitch and camera height: estimated per frame by RANSAC on the stereo points
    (USE_RANSAC_GROUND=False falls back to the known values = the oracle, for comparison).
Ground truth (reachable positions) is used for the errors and the final score.
"""
import heapq
import math
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from ai2thor.controller import Controller

# Hyperparameters/ configuration parameters / constants


SCENE_NAME      = "FloorPlan_Train6_5"
GRID_SIZE       = 0.25 # default grid size
IMG_W, IMG_H    = 300, 300 # cannot handle full rendering resolution
FOV             = 90.0     # vertical FOV (Unity convention); square image so horizontal is the same
BASELINE        = 0.12 # 12cm distance from robot, we want farther, but also dont want to look into a wall as it goes from right of the robot, robot radius is 0.2m
LOOK_DOWN_DEG   = 30       # AI2-THOR's default maximum -> floor blind spot = h / tan(30+45) = 0.42 m
CAMERA_HEIGHT   = 1.576    # ONLY used when USE_RANSAC_GROUND = False (oracle) and for the final printout
# default Global camera height
MIN_RANGE       = 0.2 # set to the robot radius
MAX_RANGE       = 2.2 # 2 meter range, depth errors grow larger as max_range increases
# try increasing this number to 5 or more and see what happens

GOAL_OFFSET_M   = (10.00, 10.50)   # (meters right, meters ahead) of spawn
# here this is unreachable which allows us to see the robot exploring the entire apartment
HALF_GRID_CELLS = 40       # 81x81 local grid (+/-10 m)
# robot spawns in the middle of the grid
MAX_STEPS       = 300
# 300 steps should be enough to explore the entire map for any robothor scene
ROBOT_RADIUS_M  = 0.20     # evaluation only, ground truth reachable cells are robot-center positions (C-space)
# configuration space in robotics is the set of all possible positions of the robot's reference point (usually its center) such that the robot does not collide with any obstacles in the environment. 
# The robot's radius is used to determine how close the robot can get to obstacles without colliding, 
# effectively "inflating" the obstacles by the robot's radius to create a safe navigation space for the robot's center point.

# ---- ablation switches ----
USE_CLAHE         = True  # per-image tiles differ between L and R, so this may hurt matching
# CLAHE went under ablation test and it seems to improve results 
# equalization helps when walls are so similar/ blank like in RoboTHOR
# adds more contrast to the image and makes it easier for the stereo algorithm to find correspondences between the left and right images.
# using with StereoSGBM openCV implementation 
USE_RANSAC_GROUND = True
# we want RANSAC ground to eventually decouple everything from ground truth data for true SLAM system
# right now RANSAC job is to estimate the ground plane from the stereo points, 
# which is used to compute the height of each point above the ground.
# in a true system with no fixed angle for turning and no fixed movement point, where robot has to estimate its own position, 
# RANSAC will estimate pitch and roll

# stereo
# these variables are used to configure the stereo matching algorithm and the subsequent processing of the disparity map to estimate the 3D structure of the scene.
# NUM_DISP is the number of disparity levels to consider, 
# BLOCK is the size of the block used for matching,
#  MIN_TEXTURE is a threshold for texture in the image, 
# and EPSILON_D is a parameter related to disparity noise.
# BLOCK is 7 because it is a good compromise between accuracy and speed. 
# A larger block size can improve matching in low-texture areas but may blur fine details,
#  while a smaller block size can capture more detail but may be more sensitive to noise.
# NUM_DISP was increased to 96 because it allows a closer range to be captured
#max disparity in pixels = min (f_x * baseline/MIN_RANGE, NUM_DISP -4 pixel buffer)
# MIN_TEXTURE is 10 because it strips out almost blank walls for horizontal stereo rig
# 
# why is epsilon_d 0.4?
# why is min texxture 10 mathematically?
# why is block 7 mathematically?
# why is num_disp 96 mathematically? empirically?
NUM_DISP        = 96 # 64
BLOCK           = 7
MIN_TEXTURE     = 10.0     # window-mean |dI/dx|; tune on Train scenes, report on Val
EPSILON_D   = 0.4      # assumed disparity noise -> height noise sigma_h(z) = z^2/(f*b) * sigma_d

# ---- RANSAC ground plane (no known pitch / height) ----
RANSAC_ITERS      = 200
# this is number of iterations to do ransac, number of plane hypotheses tested
RANSAC_ROW_FRAC   = 0.5    # candidates come from the lower half of the image, because we only care about the floor plane right now
RANSAC_MAX_Z      = 1.8    # candidates need to be relatively close to cut out noisy far points 
RANSAC_MAX_PTS    = 2000 # need plane fitting to stay fast, it is not super difficult to estimate ground plane 
RANSAC_MIN_INL    = 150  # minimum inlier count to accept a plane, stops from fitting a floor plane to flat surface like a table/chair
MIN_UP            = 0.75   # floor normal must be mostly along camera-up (camera tilted down < ~40 deg).
                           # Loose design prior: it rejects wall planes, it does not give the exact pitch.
# explain trig?
HEIGHT_RANGE      = (0.3, 3.0)   # plausible camera height [m] for an accepted plane, plane equation n p + d = 0, d is perpendicular distance from camera origin to plane/camera height
#rejects high counters under the lens, or too far away
PLANE_GATE_DEG    = 5.0    # camera mount is rigid -> floor plane in camera coords is constant across frames;
PLANE_GATE_M      = 0.15   # fits are blended (EMA)/exponential moving average if consistent, ignored if not (reset after 5 in a row)
PLANE_EMA         = 0.3 #ema= exponential moving average

# ---- point classification (heights above the estimated floor plane) ----
OBS_MIN_H       = 0.12
# object minimum height to be detected ~ 4.7 inches
# note this assumes the floors are clear. (they are in RoboTHOR)
OBS_MAX_H       = 1.50
# object maximum height to be detected ~ 1.5 meters

K_OBS  = 3.0   # obstacle: h > K_OBS*sigma_h 
# K_FLOOR is unused
AREA_OBS_M2     = 0.004      # observed obstacle surface per cell (~6% of a cell) -> counts as a hit
AREA_FLOOR_M2   = 0.006      # observed floor per cell (~10% of a cell) -> counts as seen floor

#Bayesian log-odds inverse sensor model (Thrun ch. 9), updated once per cell per frame 
def _logit(p): return math.log(p / (1.0 - p))
# use logit to represent probabilities
P_HIT, P_FLOOR, P_CLAMP = 0.82, 0.36, 0.97
P_OCC_THR, P_FREE_THR, P_STRICT = 0.80, 0.30, 0.95   # two hits cross P_OCC_THR, one floor view crosses P_FREE_THR
L_HIT, L_FLOOR = _logit(P_HIT), _logit(P_FLOOR)
L_MAX, L_OCC, L_FREE, L_STRICT = _logit(P_CLAMP), _logit(P_OCC_THR), _logit(P_FREE_THR), _logit(P_STRICT)
BUMP_LOCK       = 4.0    # log-odds written ONLY by a physical collision (> L_MAX, never touched by vision)
VISIT_LOCK      = -4.0   # log-odds of cells the robot has stood on


#occupancy grid tiles 
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

TIMES = defaultdict(list)   # stage -> seconds per call (plain perf_counter; 
#add torch.cuda.synchronize() on the DL side)
# dictionary that maps stage name to list of durations in seconds, one entry for every call

@contextmanager
def timed(stage):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        TIMES[stage].append(time.perf_counter() - t0)
# this records how long each stage of the pipeline takes
# this allows us to report times at the end and compare/contrast with a deep learning approach


# Robot stereo 
# Rig is our fixed arrangement of cameras: one at robot face, other 0.12m right of robot face
# We take stereo shot from both cameras 
# Horizontal stereo pair only
# right camera is shifted baseline=0.12
def sync_stereo_rig(controller: Controller, baseline: float = BASELINE):
    meta = controller.last_event.metadata
    #metadata dictionary from last simulator event contains camera position,rotation, camera pitch in degrees (horizon)
    # yaw_rad = yaw converted to radians
    # rx and rz are right camera's world x and z
    # event = simulator response containing both rendered images (camera 1 and 2)
    cam_pos = meta["cameraPosition"]
    agent_rot = meta["agent"]["rotation"]
    horizon = meta["agent"].get("cameraHorizon", 0.0)

    yaw_rad = math.radians(agent_rot["y"])
    rx = cam_pos["x"] + baseline * math.cos(yaw_rad)
    rz = cam_pos["z"] - baseline * math.sin(yaw_rad)
    # cam pos y stays the same (horizntal stereo pair)
    with timed("capture"):
        event = controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=dict(x=rx, y=cam_pos["y"], z=rz),
            rotation=dict(x=horizon, y=agent_rot["y"], z=0.0),
            fieldOfView=FOV,
        )
    return event.frame, event.third_party_camera_frames[0]

#initializes the controller and camera position, not full SLAM yet (snapToGrid is true, rotatestepdegrees is fixed)
def init_environment(render_depth=False):
    controller = Controller(
        scene=SCENE_NAME, gridSize=GRID_SIZE, snapToGrid=True, rotateStepDegrees=90,
        renderDepthImage=render_depth,  
          # only for the optional depth-error check; slows the simulator
        width=IMG_W, height=IMG_H, fieldOfView=FOV,
    )
    controller.step(action="LookDown", degrees=LOOK_DOWN_DEG)

    meta = controller.last_event.metadata
    cam_pos = meta["cameraPosition"]
    yaw_rad = math.radians(meta["agent"]["rotation"]["y"])
    horizon = meta["agent"].get("cameraHorizon", 0.0)

    controller.step(
        action="AddThirdPartyCamera",
        position=dict(x=cam_pos["x"] + BASELINE * math.cos(yaw_rad), y=cam_pos["y"],
                      z=cam_pos["z"] - BASELINE * math.sin(yaw_rad)),
        rotation=dict(x=horizon, y=meta["agent"]["rotation"]["y"], z=0.0),
        fieldOfView=FOV,
    )
#MAP SETUP
    size = 2 * HALF_GRID_CELLS + 1
    # +1 is the center cell the robot is in 
    grid     = np.full((size, size), UNKNOWN_CELL, dtype=int)
    # occupancy grid
    log_odds = np.zeros((size, size), dtype=np.float32)
    # separate grid for log odds of ever cell in occupancy grid
    rob_rc = (HALF_GRID_CELLS, HALF_GRID_CELLS)
    # robot row and column set to the grid center
    goal_dx, goal_dz = GOAL_OFFSET_M
    # goal offset for testing in this file
    exit_rc = (int(np.clip(HALF_GRID_CELLS - round(goal_dz / GRID_SIZE), 1, size - 2)),
               int(np.clip(HALF_GRID_CELLS + round(goal_dx / GRID_SIZE), 1, size - 2)))
    # exit row and column converted from offset + robot postion 
    #np.clip keeps this inside grid 
    log_odds[rob_rc] = VISIT_LOCK
    #start cell is a known free cell, cell has been visited so it is locked in log odds terms
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION
    return controller, grid, log_odds, rob_rc, exit_rc, float(horizon)

# converts continuous log odds into labels for the A* planner/grid
def threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc, strict=False):
#discrete planning grid from log-odds. strict=True: blocked only if very confident (no-path fallback).
# prevents false blockages (orange)
    rows, cols = grid.shape
    grid[:] = UNKNOWN_CELL # all Occupancy grid cells unknown at start
    grid[log_odds <= L_FREE] = FREE_CELL # confident for free and blocked cells
    grid[log_odds >= (L_STRICT if strict else L_OCC)] = BLOCKED_CELL
    if visited_free: # if the robot has stood on somewhere without collision it's free
        vr, vc = zip(*[(r, c) for (r, c) in visited_free if 0 <= r < rows and 0 <= c < cols])
        grid[list(vr), list(vc)] = FREE_CELL
    grid[exit_rc] = EXIT_LOCATION
    grid[rob_rc]  = ROBOT_LOCATION


#Geometry Primitives

def ransac_plane(P, sig, rng):
    # use RANSAC to estimate the floor plane
    # Guess plane from 3 random points, count inliers near it, repeat and keep best guess
    #Refit using guess's inliers
    # using the floor plane we can then know how high 3D points are later, without ground truth height
    #plane is n . p + d = 0, n is unit normal, d is scalar
    # point p, n . p + d is its signed distance from the plane
    # if n points up it's points height above the plane
    # at camera origin height is just d (camera height)
    # tilt of n away tells camera pitch
    i = rng.integers(0, len(P), (RANSAC_ITERS, 3)) # generate with set seed
    # generate 200 possible 
    nrm = np.cross(P[i[:, 1]] - P[i[:, 0]], P[i[:, 2]] - P[i[:, 0]])
    # for each of 200 trials, take 3 random point indices
    ln = np.linalg.norm(nrm, axis=1)
    # for each trial, take cross product (plane normal)
    # ln is each normal's length
    nrm /= np.maximum(ln, 1e-9)[:, None]
    nrm *= np.where(nrm[:, 1] < 0, -1.0, 1.0)[:, None]
    # makes every normal point up so the sign of n . p + d means above or below
    off = -np.einsum("ij,ij->i", nrm, P[i[:, 0]])
    # solves for d = -np, n.p + d = 0
    tol = 0.02 + 0.5 * sig
    # inlier tolerance per point is 2cm baseline plus 1/2 expected noise of the point
    # this allows far away points with larger values of expected noise to be farther from plane and still count
    dist = np.abs(P @ nrm.T + off)                                           # (N, iters) every point against every candidate plane
    score = (dist < tol[:, None]).sum(0) * ((ln > 1e-6) & (nrm[:, 1] >= MIN_UP))
    # marks inliers per point per hypothesis then counts them per hypothetical plane, also zero bad hypothesis with small length or tilted too much
    k = int(np.argmax(score))
    # pick best hypothetical plane if we can
    if score[k] < RANSAC_MIN_INL:
        return None
    inl = P[dist[:, k] < tol] # refit using all the plane's inliers
    ctr = inl.mean(0) # subtract the centroid
    n = np.linalg.svd(inl - ctr, full_matrices=False)[2][-1] # take SVD, right singular vector with smallest singular value is plane normal
    # this is best fit 
    n = n if n[1] >= 0 else -n # re orient upward
    d = float(-n @ ctr) # plane passes thru centroid
    ok = n[1] >= MIN_UP and HEIGHT_RANGE[0] <= d <= HEIGHT_RANGE[1]
    # sanity check if angle is floor like and camera height is valid
    return (n, d) if ok else None


# Main Pipeline for Classical Vision using StereoSGBM, RANSAC for ground plane, CLAHE

class ClassicalStereoMapper:
    def __init__(self, src_w: int, src_h: int, pitch_deg=None, fov=FOV, baseline=BASELINE, seed=0):
        """pitch_deg is used ONLY if USE_RANSAC_GROUND is False (oracle plane)."""
        # right now we use the RANSAC plane and will in the future so this is deprecated  can be remvoed
        self.H, self.W = src_h, src_w
        self.baseline = baseline
        self.oracle_pitch = pitch_deg
        self.fy = (self.H / 2.0) / math.tan(math.radians(fov) / 2.0)
        # focal length in pixels
        self.fx = self.fy
        # fx = fy because the pixels are square
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        # image center = principal point
        self.fb = self.fx * baseline
        # focal length times baseline, converts pixel disparity into depth
        # point at depth z appears in different columns in L and R images, col difference is disparity d
        # d = f * b / z --> z = fb/d
        self.d_lo, self.d_hi = self.fb / MAX_RANGE, min(self.fb / MIN_RANGE, NUM_DISP - 4)
        # disparity limits to search with SGBM
        self.rng = np.random.default_rng(seed) # allows us to reproduce RANSAC with set seed
        self.plane, self.bad_fits = None, 0 # smoothed floor plane 
        self.last = {}                                    # diagnostics of the last frame (shown in the plot title)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # run CLAHE algorithm, brings out texture in low contrast regions like blank walls
        #allows sgbm to find more reliable matches
        # we tested this with ablation and it improved significantly results
        # divide image into tiles
        # analyze each tile's histogram
        #capping contrast. here we use a value of 2 which works for natural images
        # image divided into 64 tiles
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=NUM_DISP, blockSize=BLOCK,
            P1=8 * 3 * BLOCK ** 2, P2=32 * 3 * BLOCK ** 2,
            disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_HH,
        )
        # Semi Global Matching (with Block)
        #StereoSGBM is OpenCV's implementation of Hirschmüller's original SGM algorithm
        #The original algorithm uses pixel-wise aggregation cost, 
        # while StereoSGBM allows matching blocks. If the block size is set to 1,
        #  it's the same as working on pixels
        # here we use a block size greater than 1
        #https://docs.opencv.org/3.4.1/d2/d85/classcv_1_1StereoSGBM.html
        # mindisparity=0 - minimum possible horizontal pixel shift is 0
        # numdisparities = NUM_DISP, positive and divisible by 16, tune this
        # blocksize= block- square mathcing window for SGBM, odd number typically 3 to 11
        # P1 is penalty applied when neighboring pixel disparity changes by +=1 pixel
        #P2 is penalty applied when disparity jumps by more than 1 pixel 
        # disp12MaxDiff is the max allowed distance in integer pixels in the left right consistency check
        # SGBM computes disparity L to R and R to L, they have to agree by +- 1 pixel
        #uniquenessRatio	Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        #speckleWindowSize	Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        #speckleRange	Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        # 3 channel RGB input image



    @classmethod # reads height and width from image
    def from_frame(cls, frame: np.ndarray, **kw):
        h, w = frame.shape[:2]
        return cls(src_w=w, src_h=h, **kw)

    # disparity
    # converts RGB to LAB (luminance, and a and b channels, a is green red and b is blue yellow)
    # https://stackoverflow.com/questions/25008458/how-to-apply-clahe-on-rgb-color-images

    def _prep(self, rgb):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        if USE_CLAHE:
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return lab
    # allows for CLAHE to work better, split brightness from color

    #
    # can't match the first NUM_DISP columns since pixel at column x looks at col x-d in right image
    
    def compute_clean_disparity(self, img_l, img_r):
        # OpenCV leaves the left NUM_DISP columns of the left-referenced map invalid; padding recovers them
        # (far points at the left edge have small disparity and match fine).
        pad = NUM_DISP
        L = cv2.copyMakeBorder(img_l, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        R = cv2.copyMakeBorder(img_r, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        d = self.sgbm.compute(L, R)[:, pad:].astype(np.float32) / 16.0
        # we divide by 16 since SGBM returns fixed point disparity with 4 fractional bits
        # a horizontal baseline only constrains horizontal gradients -> gate on window-mean |dI/dx|
        gx = np.abs(cv2.Sobel(img_l, cv2.CV_32F, 1, 0, ksize=3)).max(axis=2)
        # dispairty code gates on horizontal gradients, not vertical ones
        # this code means: 
        # horizontal baseline matches features along rows only
        # .max(axis=2) takes the strongest channel, boxFilter averages over 7x7 window
        #blank wall regions dont have a horizontal gradient so they get rejected
        # then we keep pixel if disparity is within the disparity range and texture exists
        #https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html
        tex = cv2.boxFilter(gx, -1, (BLOCK, BLOCK))
        ok = (d >= self.d_lo) & (d <= self.d_hi) & (tex >= MIN_TEXTURE)
        return np.where(ok, d, 0.0).astype(np.float32)

    # geometry, back projection
    # calculate camera coordinates, X, Y, Z relative to camera
    #z = fb/d, d is disparity, fb = 18
    # d=18 is at 1m, d=9 is at 2m
    # u and v are pinhole camera model, projecting 3d point
    #u-cx is pixels point is to right of image center, negative = left of center
    #/fx converts pixels to a ratio
    # multiply by z to scale angle by distance to get meters
    def points(self, disp):
        v, u = np.nonzero(disp > 0)
        z = self.fb / disp[v, u].astype(np.float64)
        k = (z >= MIN_RANGE) & (z <= MAX_RANGE)
        # filter points by min range and max range
        u, v, z = u[k], v[k], z[k]
        pts = np.stack([(u - self.cx) * z / self.fx, -(v - self.cy) * z / self.fy, z], axis=1)  # x right, y up, z fwd
        return pts, z, v
    # return z and pixel row v, need row for RANSAC to restrict to lower half of image for floor plane

    def estimate_ground(self, pts, z, v, sig):
        #z is depth of each point in meters
        #v is image row of each point
        #sig is estimated depth noise per point in meters
        #Output:
        # n is the floor plane unit normal
        # d is the plane offset/camera height above floor in meters
        if not USE_RANSAC_GROUND:                                   # oracle: known pitch + known height
            p = math.radians(self.oracle_pitch)
            return np.array([0.0, math.cos(p), -math.sin(p)]), CAMERA_HEIGHT
        cand = np.flatnonzero((v >= RANSAC_ROW_FRAC * self.H) & (z <= RANSAC_MAX_Z))
        # cand are points in lower half of image and within 1.8m
        # subsample to 2000 points
        fit = None
        if len(cand) >= RANSAC_MIN_INL:
            if len(cand) > RANSAC_MAX_PTS:
                cand = self.rng.choice(cand, RANSAC_MAX_PTS, replace=False)
            fit = ransac_plane(pts[cand], sig[cand], self.rng)
            # fit with ransac plane
        self.last["ransac_ok"] = fit is not None
        if fit is not None:
            if self.plane is None:
                self.plane = fit
                # accept first fit as is
            else:
                # ignore inconsistent fits, but after 5 in a row replace the plane
                # prevents first one from being wrong one
            
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
    # camera frame to map frame
    # undo downward tilt, then use R and t (rotate / translate) into self tracked local world frame
    def to_local(self, pts, pitch_deg, yaw_deg, rob_xz):
        p, a = math.radians(pitch_deg), math.radians(yaw_deg)
        # pitch and yaw
        z_h = pts[:, 1] * math.sin(p) + pts[:, 2] * math.cos(p)
        x_l = pts[:, 0]
        return (rob_xz[0] + x_l * math.cos(a) + z_h * math.sin(a),
                rob_xz[1] - x_l * math.sin(a) + z_h * math.cos(a))
    
    def cell_evidence(self, grid_shape, x, zl, h, z, sig): # Complex function with a lot of tuned variables
        # for each cell, did the frame see an obstacle, or did it see floor
        # range invariant
        rows, cols = grid_shape
        w = (z / self.fx) ** 2
        # w is the physical area one pixel covers at depth z
        # evidence is independent of range
        # weight by area instead of counting pixels to make far surfaces important in calculation
        obs = (h >= np.maximum(OBS_MIN_H, K_OBS * sig)) & (h <= OBS_MAX_H)
        # height above the floor between max (0.12, 3 * sigma) and 1.5 m
        # far floor points measured too high not trusted as an obstacle
        # upper bound ignores high up points
        # floor = height within -0.12 to 0.1m and z <= 1.4, only claim at close range
        flr = (h >= -0.12) & (h <= 0.10) & (z <= 1.4)
        r = np.round(HALF_GRID_CELLS - zl / GRID_SIZE).astype(int)
        c = np.round(HALF_GRID_CELLS + x / GRID_SIZE).astype(int)
        # which cell does each point land in
        ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        #x and zl are the points metric positions in the local map frame
        # divide by 0.25 to convert to cell units (grid_size=0.25)
        # add half-grid_cells to shift origin to grid center
        # rounding picks cell
        # row has a minus bc rows increase downward in an array while forward is up on the map
        # ok throws away anything off the occupancy grid
        idx = r * cols + c
        # sum area per cell, idx flattens row col into one integer per point, np.bincount(indices,weights) adds up weights
        # for every point that shares an index
        # m is a mask so lambda is total area of masked points per cell
        # minlength=rows*cols lets output cover every cell
        # .reshape turns back into grid
        area = lambda m: np.bincount(idx[ok & m], w[ok & m], minlength=rows * cols).reshape(rows, cols)
        a_obs, a_flr = area(obs), area(flr)
        # hit requires obstacle surface, and floor requires enough floor surface and almost no obstacle surface
        # cell with chair leg in it shouldn't be declared free just because floor around it is visible
        hit = a_obs >= AREA_OBS_M2
        floor = (a_flr >= AREA_FLOOR_M2) & (a_obs < 0.5 * AREA_OBS_M2)
        return hit, floor

    # update grid
    def update_grid_from_disparity(self, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, disp):
        """Method-agnostic part: any disparity map (0 = invalid) -> log-odds. 
        The RAFT side can call this directly."""
        with timed("mapping"):
            pts, z, v = self.points(disp)
            self.last["n_points"] = len(pts)
            if len(pts) < 50:
                # if fewer than 50 points, skip the frame
                return
            sig = z ** 2 / self.fb * EPSILON_D
            # stereo error 

            plane = self.estimate_ground(pts, z, v, sig)
            if plane is None:                                       # no floor model yet -> skip, never guess
                return
            n, d = plane
            h = pts @ n + d
                        # get the plane then h = pts @ n + d
            pitch_est = math.degrees(math.atan2(-n[2], n[1]))
            self.last.update(pitch_est=pitch_est, cam_h_est=d)

            rob_xz = ((rob_rc[1] - HALF_GRID_CELLS) * GRID_SIZE, (HALF_GRID_CELLS - rob_rc[0]) * GRID_SIZE)
            x, zl = self.to_local(pts, pitch_est, local_yaw, rob_xz)
            hit, floor = self.cell_evidence(grid.shape, x, zl, h, z, sig)
            # get robot metric position from its cell, project points to the map, and get hit / floor
            # mutable: cells vision is not allowed to change are cells with collisions, robot cell, goal, 
            # and cells robot drove on
            mutable = log_odds < BUMP_LOCK                          # collisions are final
            mutable[rob_rc] = mutable[exit_rc] = False
            if visited_free:
                vr, vc = zip(*visited_free)
                mutable[list(vr), list(vc)] = False
            log_odds[floor & mutable & (log_odds < L_OCC)] += L_FLOOR
            #add log odds evidence

#    pass previously confirmed obstacles so rays stop instead of erasing them
            log_odds[hit & mutable] += L_HIT
            # single hit marks cell blocked
            log_odds[floor & mutable] += L_FLOOR # updating twice crosses the threshold P_FREE_THR

            log_odds[mutable] = np.clip(log_odds[mutable], -L_MAX, L_MAX)
            # clip to L_MAX, then re threshold and update grid
            threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc)

    def update_grid_from_stereo(self, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, rgb_left, rgb_right):
        with timed("disparity"):
            disp = self.compute_clean_disparity(self._prep(rgb_left), self._prep(rgb_right))
        self.update_grid_from_disparity(grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, disp)
        return disp
    # updates grid


def observe_forward(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw):
    """One stereo frame in the current heading. Used after every move / turn (no 360 spin)."""
    rgb_l, rgb_r = sync_stereo_rig(controller)
    return mapper.update_grid_from_stereo(grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw, rgb_l, rgb_r)
# look once, reposition right camera, return L and R RGB images from current pose
# then run pipeline on that pair, disparity between images, disparity points, floor plane calc,
# cell evidence, log odds update, re threshold 
# returns disparity map for plot 


def scan_all_directions_stereo(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw):
    """4 views, back to the original heading. Only at spawn and when the planner is stuck."""
    disp = None
    for _ in range(4):
        disp = observe_forward(controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw)
        controller.step(action="RotateRight")
        local_yaw = (local_yaw + 90) % 360
    return disp, local_yaw
# do a 360 degree survey, used at spawn once so the robot starts with map of all 4 directions
# when stuck, also does 360 degree survey (move_name is None)



# A* algorithm, 4 way movement, Manhattan distance, soft obstacle clearance
def next_move(grid: np.ndarray):  # the robot size is 1x1 so we can treat it as a point in the grid
    rows, cols = grid.shape
    robot_pts = np.argwhere(grid == ROBOT_LOCATION)
    exit_pts  = np.argwhere(grid == EXIT_LOCATION)
    if robot_pts.size == 0 or exit_pts.size == 0:
        return None, []  # we return none for no path
    start, goal = tuple(int(v) for v in robot_pts[0]), tuple(int(v) for v in exit_pts[0])

    # soft clearance: discourages hugging walls on a 0.25 m grid without hard-blocking 0.5 m doorways
    near_blocked = cv2.dilate((grid == BLOCKED_CELL).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    heuristic = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
  # heuristic is h(n) = Manhattan distance from the current cell to the exit cell
    # heuristic can never overestimate the cost to reach the goal, so we use Manhattan distance
    pq = [(heuristic(start), 0.0, start)]  # priority queue is a list of tuples (estimated total cost, cost to reach, coordinate)
    cost_to_reach, came_from = {start: 0.0}, {}  # cost_to_reach is a dictionary that maps coordinates to the cost to reach that coordinate from the start coordinate
    # came_from is a dictionary that maps coordinates to the coordinate that we came from to reach that coordinate
    while pq:
        _, g_cost, curr = heapq.heappop(pq)  # pop the coordinate with the lowest estimated total cost from the priority queue
        # the first variable is the estimated total cost, we don't need it
        if curr == goal:  # we have reached the exit, reconstruct the path
            path = [curr]
            while path[-1] != start:  # reconstruct the path by following the came_from dictionary from the exit to the start
                path.append(came_from[path[-1]])
            path.reverse()
            if len(path) < 2:  # if the path is only 1 step long, we are already at the exit, so there is no next move
                return None, path
            delta = (path[1][0] - start[0], path[1][1] - start[1])  # here we calculate the delta between the first step and the start coordinate to determine the move name
            # the name of the move is used to turn the robot in the right direction to reach the exit
            return next(k for k, v in CARDINAL_MOVES.items() if v == delta), path
        if g_cost > cost_to_reach[curr]:  # there exists a better path, skip this one
            continue
        for dr, dc in CARDINAL_MOVES.values():   # calculate the neighbor coordinate by adding the delta to the current coordinate
            nb = (curr[0] + dr, curr[1] + dc)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid[nb] == BLOCKED_CELL:
                continue  # check if the neighbor coordinate is within the grid bounds and not blocked
            step = 1.25 if grid[nb] == UNKNOWN_CELL else 1.0
            if near_blocked[nb] and nb != goal:
                step += 0.90  # calculate the cost to reach the neighbor coordinate
            # small penalty (+0.25) is added if the neighbor cell is unknown, to encourage exploration of known free cells first
              # additional penalty (+0.90) is added if the cell is adjacent to a blocked cell (unless it is the goal)
            if g_cost + step < cost_to_reach.get(nb, float("inf")):
                 # if the tentative cost to reach the neighbor coordinate is less than the previously recorded cost, update the cost and path
                 # we push the neighbor coordinate onto the priority queue with its estimated total cost and tentative cost
                # priority queue is a min-heap, so the coordinate with the lowest estimated total cost will be popped first
                cost_to_reach[nb] = g_cost + step
                came_from[nb] = curr
                heapq.heappush(pq, (g_cost + step + heuristic(nb), g_cost + step, nb))
    return None, [] # the function returns None if there is no path to the exit, and an empty list for the path


# Self Tracked Motion Execution
def turn_toward(controller, move_name, local_yaw):
    #One 90-degree turn toward the move direction (a 180 takes two calls, so we never turn blind twice)
    diff = (MOVE_TO_YAW[move_name] - local_yaw) % 360
    if diff == 270:
        controller.step(action="RotateLeft");  return (local_yaw - 90) % 360
    controller.step(action="RotateRight");     return (local_yaw + 90) % 360


def execute_move(controller, move_name, grid, log_odds, rob_rc, exit_rc, local_yaw):
    #Caller has already turned to face `move_name`
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



# EVALUATION (ground truth; evaluation + visualisation only, never seen by the navigator)

def _to_local(spawn, x, z):
    sx, sz, yaw = spawn
    dx, dz = x - sx, z - sz
    return dx * math.cos(yaw) - dz * math.sin(yaw), dx * math.sin(yaw) + dz * math.cos(yaw)

# Capture the ground truth
def capture_gt(controller):
    #gt_free[r, c]: robot centre can stand there (AI2-THOR reachable positions)
    #  Call right after spawn
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
    #Shortest 4-connected path in cells on the GT free mask (same action space as the robot)
    # use to compare path length
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
    for stage in ("disparity", "mapping", "planning"):
        if TIMES[stage]:
            out[f"{stage}_ms_per_call"] = 1e3 * float(np.mean(TIMES[stage]))
    out["perception_ms_per_frame"] = 1e3 * (sum(TIMES["disparity"]) + sum(TIMES["mapping"])) / max(frames, 1)
    return out



#LIVE PLOT: occupancy grid
# grid vs ground truth
# left camera
#  disparity

class LiveStereoPlotter:
    def __init__(self, grid, work_w, work_h, gt_free=None):
        plt.ion()
        self.gt_free = gt_free
        self.fig, ((self.ax_grid, self.ax_err), (self.ax_rgb, self.ax_disp)) = plt.subplots(2, 2, figsize=(13, 11.5))
        self.ax_err.sharex(self.ax_grid)
        self.ax_err.sharey(self.ax_grid)

        self.cmap = ListedColormap(["#2b83ba", "#fdae61", "#ffffff", "#2c3e50", "#bdc3c7"])   # robot, exit, free, blocked, unknown
        self.im_grid = self.ax_grid.imshow(grid, cmap=self.cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
        self.path_line, = self.ax_grid.plot([], [], color="#00a86b", linewidth=1.8, marker=".", markersize=4, label="A* planned path")
        self.taken_line, = self.ax_grid.plot([], [], color="#d62728", linewidth=2, marker=".", markersize=4, label="Path taken")
        self.ax_grid.legend(loc="upper left", fontsize=8)
        cbar = self.fig.colorbar(self.im_grid, ax=self.ax_grid, ticks=[-2, -1, 0, 1, 2], shrink=0.7)
        cbar.ax.set_yticklabels(["-2: Robot", "-1: Exit", "0: Free", "1: Blocked", "2: Unknown"])

        self.err_cmap = ListedColormap(["#d9d9d9", "#c6dbef", "#b8e6b8", "#2c3e50", "#e41a1c", "#ff9f1c"])
        self.im_err = self.ax_err.imshow(np.zeros(grid.shape, int), cmap=self.err_cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
        self.ax_err.legend(handles=[Patch(color=c, label=l) for c, l in zip(
            ["#d9d9d9", "#c6dbef", "#b8e6b8", "#2c3e50", "#e41a1c", "#ff9f1c"],
            ["unknown", "reachable, not yet mapped", "free: correct", "blocked: correct",
             "FALSE-FREE (collision risk)", "PHANTOM (blocked, reachable)"])], loc="upper left", fontsize=8)
        for ax in (self.ax_grid, self.ax_err):
            ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which="minor", color="#95a5a6", linestyle="-", linewidth=0.3)
        if gt_free is None:
            self.ax_err.axis("off")

        self.im_rgb = self.ax_rgb.imshow(np.zeros((work_h, work_w, 3), dtype=np.uint8))
        self.ax_rgb.axis("off")
        self.ax_rgb.set_title("Left head camera (RGB)")
        self.im_disp = self.ax_disp.imshow(np.zeros((work_h, work_w), dtype=np.float32), cmap="plasma", vmin=0, vmax=NUM_DISP)
        self.ax_disp.axis("off")
        self.ax_disp.set_title("SGBM disparity (black = rejected)")
        plt.tight_layout()

    def update(self, grid, path, rgb_frame, disp_frame, step_idx, status, taken_path=(), extra=""):
        self.im_grid.set_data(grid)
        known_r, known_c = np.nonzero(grid != UNKNOWN_CELL)                 # auto-zoom: 81x81 stays readable
        if len(known_r):
            pad = 4
            self.ax_grid.set_xlim(max(-0.5, known_c.min() - pad), min(grid.shape[1] - 0.5, known_c.max() + pad))
            self.ax_grid.set_ylim(min(grid.shape[0] - 0.5, known_r.max() + pad), max(-0.5, known_r.min() - pad))
        self.path_line.set_data([p[1] for p in path], [p[0] for p in path])
        self.taken_line.set_data([p[1] for p in taken_path], [p[0] for p in taken_path])
        self.ax_grid.set_title(f"Occupancy grid (step {step_idx})\n{status}", fontsize=10)
        if self.gt_free is not None:
            self.im_err.set_data(error_map(grid, self.gt_free))
            q = grid_quality(grid, self.gt_free)
            self.ax_err.set_title(f"Grid vs ground truth (robot-radius C-space)\nprec={q['blocked_precision']:.2f}  "
                                  f"false-free={q['false_free_rate']:.2f}  coverage={q['coverage']:.2f}", fontsize=10)
        if rgb_frame is not None:
            self.im_rgb.set_data(rgb_frame)
        if disp_frame is not None:
            self.im_disp.set_data(np.clip(disp_frame, 0, NUM_DISP))
        if extra:
            self.ax_disp.set_title(f"SGBM disparity (black = rejected)\n{extra}", fontsize=9)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)

    def save(self, path="occupancy_final.png"):
        self.fig.savefig(path, dpi=140)
        print(f"[plot] saved {path}")



# MAIN iterates through finding the next best move and executing
# using our provided A* algorithm and vision algorithms
# TODO: implement full SLAM with no ground truth pose and movement
# requires loop closure, etc.

if __name__ == "__main__":
    controller, occ_grid, log_odds, rob_rc, exit_rc, pitch_deg = init_environment()
    gt = capture_gt(controller)                      # evaluation + error panel only

    left0, _ = sync_stereo_rig(controller)
    stereo_mapper = ClassicalStereoMapper.from_frame(left0, pitch_deg=pitch_deg)   # pitch used only if USE_RANSAC_GROUND=False
    plotter = LiveStereoPlotter(occ_grid, stereo_mapper.W, stereo_mapper.H, gt["gt_free"])
    TIMES.clear()

    local_yaw    = 0
    visited_free = {rob_rc}
    taken_path   = [rob_rc]
    frames = moves_ok = collisions = rotations = steps = 0

    last_disp, local_yaw = scan_all_directions_stereo(controller, stereo_mapper, occ_grid, log_odds, visited_free,
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
        ex = stereo_mapper.last
        info = (f"RANSAC floor: pitch={ex['pitch_est']:.1f} deg, cam h={ex['cam_h_est']:.2f} m" if "pitch_est" in ex else "")
        plotter.update(occ_grid, path, frame_now, last_disp, step,
                       f"Robot {rob_rc} -> Goal {exit_rc} | next: {move_name}", taken_path, info)

        if move_name is None:
            if rob_rc not in scanned:                # stuck: one 360 scan per cell
                last_disp, local_yaw = scan_all_directions_stereo(controller, stereo_mapper, occ_grid, log_odds,
                                                                  visited_free, rob_rc, exit_rc, local_yaw)
                frames += 4; rotations += 4
                scanned.add(rob_rc)
                continue
            print(f"[Step {step}] No valid path found!")
            break

        if MOVE_TO_YAW[move_name] != local_yaw:      # turn, LOOK, then re-plan (never move after a blind turn)
            local_yaw = turn_toward(controller, move_name, local_yaw)
            rotations += 1
            last_disp = observe_forward(controller, stereo_mapper, occ_grid, log_odds, visited_free,
                                        rob_rc, exit_rc, local_yaw)
            frames += 1
            continue

        success, rob_rc, local_yaw = execute_move(controller, move_name, occ_grid, log_odds, rob_rc, exit_rc, local_yaw)
        print(f"[Step {step:03d}] {move_name:<5} ok={str(success):<5} cell={rob_rc} steps left={len(path) - 1}")
        if success:
            moves_ok += 1
            visited_free.add(rob_rc)
            taken_path.append(rob_rc)
            last_disp = observe_forward(controller, stereo_mapper, occ_grid, log_odds, visited_free,
                                        rob_rc, exit_rc, local_yaw)
            frames += 1
        else:
            collisions += 1

    threshold_grid(occ_grid, log_odds, visited_free, rob_rc, exit_rc)
    plotter.update(occ_grid, [], controller.last_event.frame, last_disp, steps,
                   "GOAL REACHED!" if rob_rc == exit_rc else "STOPPED (no path / step limit)", taken_path)
    plotter.save("occupancy_final.png")

    res = score_episode(controller, gt, occ_grid, exit_rc, moves_ok, collisions, rotations, frames, steps)
    print("\n===== RESULT =====")
    for k, v in res.items():
        print(f"  {k:<26}{v:.3f}" if isinstance(v, float) else f"  {k:<26}{v}")
    ex = stereo_mapper.last
    if "pitch_est" in ex:
        print(f"  RANSAC pitch {ex['pitch_est']:.2f} deg vs true {pitch_deg:.1f} | "
              f"RANSAC cam height {ex['cam_h_est']:.3f} m vs {CAMERA_HEIGHT} m")

    plt.ioff()
    plt.show()
    controller.stop()
