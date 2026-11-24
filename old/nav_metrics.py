"""
nav_metrics.py  --  modular, READ-ONLY metrics tracking for the classical stereo navigator.

Design rules
  * Nothing in the navigator's logic changes. Functions are wrapped from the outside
    (timing + counting only) and return exactly what they returned before.
  * Ground-truth queries (optimal path, reachable cells, depth map) are used for
    EVALUATION ONLY. The planner never sees them. Disable with use_ground_truth=False.
  * Method-agnostic: a deep-learning pipeline can reuse the same tracker via
    `with metrics.timer("inference"): ...` and `MetricsTracker(cuda_sync=torch.cuda.synchronize,
    compute_stages=("inference", "planning"))`.

Optional dependencies (all degrade gracefully when missing):
    pip install psutil nvidia-ml-py
"""
from __future__ import annotations

import csv
import functools
import glob
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime

import numpy as np

import GroundTruthBaseline as gt_baseline

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None
try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None
try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

__all__ = ["MetricsTracker", "ResourceSampler"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()),
                std=float(v.std()))


def _mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _get(d, dotted, default=None):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def _f(x, spec=".1f", none="n/a"):
    return none if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# background hardware sampler: CPU / RAM / GPU / power
# ---------------------------------------------------------------------------
class ResourceSampler:
    """
    Samples at a fixed interval in a daemon thread.
      * 'py'  = this Python process  (your algorithm: OpenCV, NumPy, torch, ...)
      * 'sim' = child processes      (the AI2-THOR Unity renderer)
    CPU % is "% of one core" (can exceed 100). cpu_sys_pct is whole-machine, 0-100.
    GPU numbers come from NVML; power from NVML (GPU) and Intel/AMD RAPL (CPU package, Linux).
    """

    def __init__(self, interval=0.2, gpu_index=0, rapl_dir="/sys/class/powercap/intel-rapl:0"):
        self.interval = interval
        self.samples = []
        self.phase = "run"      # "idle" while calibrating a baseline
        self.step = -1
        self.t0 = None
        self.info = {}
        self._stop = threading.Event()
        self._thread = None
        self._procs = {}
        self._py = None
        self._gpu = None
        self._rapl = None
        self._init_cpu()
        self._init_gpu(gpu_index)
        self._init_rapl(rapl_dir)

    # -- init -------------------------------------------------------------
    def _init_cpu(self):
        if psutil is None:
            print("[metrics] psutil not installed -> CPU/RAM metrics disabled")
            return
        self._py = psutil.Process(os.getpid())
        self._py.cpu_percent(None)
        psutil.cpu_percent(None)
        self.info["cpu_cores_logical"] = psutil.cpu_count(logical=True)

    def _init_gpu(self, idx):
        if pynvml is None:
            print("[metrics] nvidia-ml-py not installed -> GPU metrics disabled")
            return
        try:
            pynvml.nvmlInit()
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(self._gpu)
            self.info["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
        except Exception as e:
            print(f"[metrics] NVML unavailable ({e}) -> GPU metrics disabled")
            self._gpu = None

    def _init_rapl(self, d):
        try:
            with open(os.path.join(d, "energy_uj")) as f:
                e = int(f.read())
            try:
                with open(os.path.join(d, "max_energy_range_uj")) as f:
                    mx = int(f.read())
            except OSError:
                mx = 2 ** 32
            self._rapl = dict(dir=d, max=mx, last_t=time.perf_counter(), last_e=e)
            self.info["cpu_power_source"] = "RAPL"
        except (OSError, ValueError):
            self._rapl = None  # not Linux / no permission (needs: sudo chmod -R a+r /sys/class/powercap/intel-rapl*)

    # -- control ----------------------------------------------------------
    def start(self):
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        nxt = time.perf_counter()
        first = True
        while not self._stop.is_set():
            try:
                sample = self._tick()
                if not first:                 # first tick's CPU % covers an undefined interval -> discard
                    self.samples.append(sample)
                first = False
            except Exception:
                pass
            nxt += self.interval
            self._stop.wait(max(0.0, nxt - time.perf_counter()))

    # -- one sample -------------------------------------------------------
    def _tick(self):
        now = time.perf_counter()
        s = dict(t=now - self.t0, phase=self.phase, step=self.step)

        if self._py is not None:
            s["cpu_py_pct"] = self._py.cpu_percent(None)
            s["rss_py_mb"] = self._py.memory_info().rss / 2 ** 20
            sim_cpu = sim_rss = 0.0
            live = {}
            for ch in self._py.children(recursive=True):
                p = self._procs.get(ch.pid, ch)   # reuse Process objects so cpu_percent has history
                try:
                    sim_cpu += p.cpu_percent(None)
                    sim_rss += p.memory_info().rss
                    live[ch.pid] = p
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self._procs = live
            s["cpu_sim_pct"] = sim_cpu
            s["rss_sim_mb"] = sim_rss / 2 ** 20
            s["cpu_sys_pct"] = psutil.cpu_percent(None)

        if self._gpu is not None:
            try:
                s["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu
                s["gpu_mem_used_mb"] = pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 2 ** 20
            except Exception:
                pass
            try:
                s["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0
            except Exception:
                pass
            pids = {os.getpid(), *self._procs}
            used = 0
            for fn in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    for p in getattr(pynvml, fn)(self._gpu):
                        if p.pid in pids and p.usedGpuMemory:
                            used += p.usedGpuMemory
                except Exception:
                    pass
            s["gpu_mem_proc_mb"] = used / 2 ** 20

        if self._rapl is not None:
            try:
                with open(os.path.join(self._rapl["dir"], "energy_uj")) as f:
                    e = int(f.read())
                de = e - self._rapl["last_e"]
                if de < 0:
                    de += self._rapl["max"]
                dt = now - self._rapl["last_t"]
                if dt > 0:
                    s["cpu_power_w"] = de / 1e6 / dt
                self._rapl.update(last_t=now, last_e=e)
            except (OSError, ValueError):
                pass
        return s


# ---------------------------------------------------------------------------
# main tracker
# ---------------------------------------------------------------------------
class MetricsTracker:
    RES_KEYS = ["cpu_py_pct", "cpu_sim_pct", "cpu_sys_pct", "rss_py_mb", "rss_sim_mb",
                "gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_proc_mb", "gpu_power_w", "cpu_power_w"]

    def __init__(self, method_name, scene, grid_size, half_grid, out_dir="metrics_out",
                 sample_interval=0.2, gpu_index=0, cuda_sync=None, success_radius_m=0.30,
                 use_ground_truth=True, depth_range=(0.30, 2.2),
                 compute_stages=("mapping", "planning"), sim_stages=("stereo_capture", "execute_move"),
                 cell_codes=None, episode_id=None, extra_meta=None, verbose=True, idle_baseline=None):
        self.method_name, self.scene = method_name, scene
        self.episode_id = episode_id
        self.extra_meta = dict(extra_meta or {})
        self.verbose = verbose
        self.idle_baseline = idle_baseline        # reuse an idle measurement from an earlier episode
        self.last_idle_measured = None
        self._step_backup = None
        self._wrapped_controller = None
        self.grid_size, self.half_grid = grid_size, half_grid
        self.out_dir = out_dir
        self.success_radius_m = success_radius_m
        self.use_gt = use_ground_truth
        self.depth_range = depth_range
        self.compute_stages = tuple(compute_stages)   # "algorithm" time (what you compare against DL)
        self.sim_stages = tuple(sim_stages)           # simulator latency (NOT part of the algorithm)
        self.cell = dict(robot=-2, exit=-1, free=0, blocked=1, unknown=2)
        if cell_codes:
            self.cell.update(cell_codes)
        self._sync = cuda_sync or (lambda: None)

        self.sampler = ResourceSampler(sample_interval, gpu_index)
        self.call_times = defaultdict(list)           # stage -> [seconds per call]
        self.steps, self.vision, self.map_series = [], [], []
        self.counters = Counter()
        self.sim_actions = Counter()
        self.collision_log = []
        self.first_plan_len_cells = None

        self._cur = None
        self._step_idx = -1
        self._t_ep0 = None
        self._controller = None
        self._spawn = None
        self.goal_local = None
        self.goal_world = None
        self.optimal_path_m = None
        self.ai2thor_shortest_path_m = None
        self.ground_truth_astar_m = None
        self.gt_free = None
        self._gt_dist = 0.0
        self._gt_last = None

    # ======================= setup =========================================
    def attach(self, controller, goal_offset_m):
        """Call once after init_environment(). Reads spawn pose, queries GT (eval only), counts sim actions."""
        self._controller = controller
        pos = controller.last_event.metadata["agent"]["position"]
        yaw = math.radians(controller.last_event.metadata["agent"]["rotation"]["y"])
        self._spawn = (pos["x"], pos["y"], pos["z"], yaw)
        self._gt_last = (pos["x"], pos["z"])
        G = self.grid_size
        self.goal_local = (round(goal_offset_m[0] / G) * G, round(goal_offset_m[1] / G) * G)
        gx, gz = self._local_to_world(*self.goal_local)
        self.goal_world = (gx, pos["y"], gz)

        if self.use_gt:
            self._setup_ground_truth(controller, pos)

        # count every simulator call (installed AFTER GT queries so those aren't counted)
        self._wrapped_controller = controller
        self._step_backup = controller.__dict__.get("step")
        orig_step = controller.step
        tracker = self

        def counted(*a, **kw):
            name = kw.get("action") or (a[0] if a else "unknown")
            tracker.sim_actions[name] += 1
            t0 = time.perf_counter()
            try:
                return orig_step(*a, **kw)
            finally:
                tracker._add_time("sim_step", time.perf_counter() - t0)

        controller.step = counted

    def detach(self):
        """Remove the sim-action counter so multiple episodes can reuse one controller."""
        c = self._wrapped_controller
        if c is None:
            return
        if self._step_backup is None:
            c.__dict__.pop("step", None)
        else:
            c.step = self._step_backup
        self._wrapped_controller = None

    def _setup_ground_truth(self, controller, pos):
        gx, gy, gz = self.goal_world
        try:  # optimal geodesic distance, start -> goal
            ev = controller.step(action="GetShortestPathToPoint", target=dict(x=gx, y=gy, z=gz))
            ret = ev.metadata.get("actionReturn")
            corners = ret.get("corners") if isinstance(ret, dict) else ret
            if ev.metadata.get("lastActionSuccess") and corners:
                pts = [(pos["x"], pos["z"])] + [(p["x"], p["z"]) for p in corners]
                self.ai2thor_shortest_path_m = float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))
        except Exception as e:
            print(f"[metrics] optimal path unavailable ({e})")
        try:  # reachable cells -> GT free-space mask in the robot's local grid
            ev = controller.step(action="GetReachablePositions")
            rp = ev.metadata.get("actionReturn") or []
            H, G = self.half_grid, self.grid_size
            size = 2 * H + 1
            mask = np.zeros((size, size), dtype=bool)
            for p in rp:
                lx, lz = self._world_to_local(p["x"], p["z"])
                r, c = int(round(H - lz / G)), int(round(H + lx / G))
                if 0 <= r < size and 0 <= c < size:
                    mask[r, c] = True
            self.gt_free = mask if mask.any() else None
        except Exception as e:
            print(f"[metrics] reachable-position GT unavailable ({e})")

        if self.gt_free is not None:
            start = (H, H)
            goal = (H - round(self.goal_local[1] / G), H + round(self.goal_local[0] / G))
            if all(0 <= v < size for v in goal) and self.gt_free[start] and self.gt_free[goal]:
                grid = np.where(self.gt_free, gt_baseline.FREE_CELL, gt_baseline.BLOCKED_CELL)
                path = gt_baseline.astar(grid, start, goal)
                if path is not None:
                    self.ground_truth_astar_m = (len(path) - 1) * G
                    self.optimal_path_m = self.ground_truth_astar_m

    def start(self, baseline_s=0.0):
        """Start the hardware sampler. baseline_s>0 first records idle hardware draw (sim idle, algorithm not running)."""
        self.sampler.start()
        if baseline_s > 0:
            self.sampler.phase = "idle"
            time.sleep(baseline_s)
            self.sampler.phase = "run"

    # ======================= coordinate helpers ============================
    def _world_to_local(self, x, z):
        sx, _, sz, yaw = self._spawn
        dx, dz = x - sx, z - sz
        return dx * math.cos(yaw) - dz * math.sin(yaw), dx * math.sin(yaw) + dz * math.cos(yaw)

    def _local_to_world(self, lx, lz):
        sx, _, sz, yaw = self._spawn
        return sx + lx * math.cos(yaw) + lz * math.sin(yaw), sz - lx * math.sin(yaw) + lz * math.cos(yaw)

    # ======================= timing / wrapping =============================
    def _add_time(self, stage, dt):
        self.call_times[stage].append(dt)
        if self._cur is not None:
            self._cur["stages"][stage] += dt

    @contextmanager
    def timer(self, stage):
        """with metrics.timer('inference'): ...   (use cuda_sync=torch.cuda.synchronize for GPU models)"""
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._add_time(stage, time.perf_counter() - t0)

    def wrap(self, stage, fn):
        @functools.wraps(fn)
        def inner(*a, **kw):
            with self.timer(stage):
                return fn(*a, **kw)
        return inner

    def wrap_planner(self, fn, stage="planning"):
        timed = self.wrap(stage, fn)

        @functools.wraps(fn)
        def inner(*a, **kw):
            move, path = timed(*a, **kw)
            n = max(len(path) - 1, 0) if path else None
            if self._cur is not None:
                self._cur["plan_len_cells"] = n
            if path and self.first_plan_len_cells is None:
                self.first_plan_len_cells = n
            if move is None and not path:
                self.counters["no_path"] += 1
            return move, path
        return inner

    def wrap_executor(self, fn, stage="execute_move"):
        """Detects collisions: a MoveAhead the simulator rejected (path was physically blocked)."""
        timed = self.wrap(stage, fn)

        @functools.wraps(fn)
        def inner(controller, move_name, *a, **kw):
            before = controller.last_event
            out = timed(controller, move_name, *a, **kw)
            after = controller.last_event
            self.counters["moves_attempted"] += 1
            if out[0]:
                self.counters["moves_ok"] += 1
            elif after is not before:
                m = after.metadata
                if m.get("lastAction") == "MoveAhead" and not m.get("lastActionSuccess", True):
                    self.counters["collisions"] += 1
                    self.collision_log.append(dict(step=self._step_idx, move=move_name,
                                                   error=str(m.get("errorMessage", ""))[:160]))
            self._update_gt_travel(after)
            return out
        return inner

    def wrap_mapper(self, mapper):
        """Instance-level wrap of the stereo mapper (class itself is untouched)."""
        mapper.compute_clean_disparity = self.wrap("disparity", mapper.compute_clean_disparity)
        mapper.match_sift_epipolar = self.wrap("sift_match", mapper.match_sift_epipolar)
        timed_update = self.wrap("mapping", mapper.update_grid_from_stereo)

        @functools.wraps(mapper.update_grid_from_stereo)
        def update(*a, **kw):
            disp, pts_sift = timed_update(*a, **kw)
            self._record_vision(mapper, disp, pts_sift)      # outside the timed region
            return disp, pts_sift

        mapper.update_grid_from_stereo = update

    def note(self, name, n=1):
        self.counters[name] += n

    # ======================= per-step bookkeeping ==========================
    def begin_step(self, step, grid=None):
        """First line of the main loop. Automatically closes the previous step."""
        now = time.perf_counter()
        self._close_step(now)
        if self._t_ep0 is None:
            self._t_ep0 = now
        self._step_idx = step
        self.sampler.step = step
        if grid is not None:
            m = self._map_metrics(grid)
            if m:
                self.map_series.append(dict(step=step, **m))
        self._cur = dict(step=step, t0=now, stages=defaultdict(float), plan_len_cells=None)

    def _close_step(self, now):
        cur = self._cur
        if cur is None:
            return
        st = cur["stages"]
        rec = dict(step=cur["step"], wall_ms=(now - cur["t0"]) * 1000,
                   compute_ms=1000 * sum(st.get(k, 0.0) for k in self.compute_stages),
                   sim_ms=1000 * sum(st.get(k, 0.0) for k in self.sim_stages),
                   plot_ms=1000 * st.get("plotting", 0.0),
                   plan_len_cells=cur["plan_len_cells"])
        rec.update({f"ms_{k}": v * 1000 for k, v in st.items()})
        self.steps.append(rec)
        self._cur = None

    def _update_gt_travel(self, event):
        if not self.use_gt or self._gt_last is None:
            return
        p = event.metadata["agent"]["position"]
        self._gt_dist += math.hypot(p["x"] - self._gt_last[0], p["z"] - self._gt_last[1])
        self._gt_last = (p["x"], p["z"])

    # ======================= vision metrics ================================
    def _record_vision(self, mapper, disp, pts_sift):
        rec = dict(step=self._step_idx, disp_density=float((disp > 0).mean()), sift_matches=int(len(pts_sift)))
        ev = self._controller.last_event if self._controller is not None else None
        depth = getattr(ev, "depth_frame", None) if ev is not None else None
        if self.use_gt and depth is not None:
            rec.update(self._depth_errors(mapper, disp, depth))
        self.vision.append(rec)

    def _depth_errors(self, mapper, disp, depth):
        """Needs renderDepthImage=True on the Controller. Assumes depth_frame is z-depth in metres."""
        lo, hi = self.depth_range
        if depth.shape != disp.shape and cv2 is not None:
            depth = cv2.resize(depth, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        fb = mapper.fx * mapper.baseline
        gt_ok = np.isfinite(depth) & (depth >= lo) & (depth <= hi)
        valid = disp > 0
        z = np.zeros_like(disp)
        z[valid] = fb / disp[valid]
        both = valid & gt_ok
        out = dict(depth_coverage=float(both.sum() / max(int(gt_ok.sum()), 1)))
        if both.sum() >= 50:
            g, p = depth[both], z[both]
            out.update(depth_mae_m=float(np.mean(np.abs(p - g))),
                       depth_rmse_m=float(np.sqrt(np.mean((p - g) ** 2))),
                       depth_absrel=float(np.mean(np.abs(p - g) / g)),
                       depth_delta1=float(np.mean(np.maximum(p / g, g / p) < 1.25)),
                       disp_bad1=float(np.mean(np.abs(disp[both] - fb / g) > 1.0)))
        return out

    def _map_metrics(self, grid):
        if self.gt_free is None or grid.shape != self.gt_free.shape:
            return None
        c = self.cell
        free = (grid == c["free"]) | (grid == c["robot"])
        blocked = grid == c["blocked"]
        gt = self.gt_free
        nf, nb, ng = int(free.sum()), int(blocked.sum()), int(gt.sum())
        tp = int((free & gt).sum())
        phantom = int((blocked & gt).sum())
        return dict(n_free=nf, n_blocked=nb,
                    free_precision=tp / nf if nf else None,                 # FREE cells that are truly reachable
                    false_free_cells=int((free & ~gt).sum()),               # dangerous: believes free, is not
                    phantom_obstacle_cells=phantom,                         # believes blocked, is reachable
                    phantom_obstacle_rate=phantom / nb if nb else None,
                    reachable_coverage=tp / ng if ng else None)             # share of reachable space mapped

    # ======================= finish & report ===============================
    def finish(self, success, final_rc=None, exit_rc=None, grid=None, max_steps=None, save=True, save_csv=True):
        now = time.perf_counter()
        self._close_step(now)
        self.sampler.stop()
        wall = (now - self._t_ep0) if self._t_ep0 else 0.0
        G, c = self.grid_size, self.counters

        nav = dict(
            success=bool(success),
            termination="goal" if success else ("max_steps" if max_steps and len(self.steps) >= max_steps
                                                else "no_path_or_aborted"),
            nav_steps=len(self.steps), moves_attempted=c["moves_attempted"], moves_ok=c["moves_ok"],
            collisions=c["collisions"], collision_rate=c["collisions"] / max(c["moves_attempted"], 1),
            no_path_events=c["no_path"], replan_retries=c["replan_retry"],
            rotations=self.sim_actions["RotateLeft"] + self.sim_actions["RotateRight"],
            path_length_m=c["moves_ok"] * G,
            first_plan_length_m=None if self.first_plan_len_cells is None else self.first_plan_len_cells * G,
            sim_actions=dict(self.sim_actions), collision_log=self.collision_log,
        )
        if self.use_gt and self._controller is not None and self.goal_world is not None:
            pos = self._controller.last_event.metadata["agent"]["position"]
            dist = math.hypot(pos["x"] - self.goal_world[0], pos["z"] - self.goal_world[2])
            ok_gt = dist <= self.success_radius_m
            used, opt = self._gt_dist, self.optimal_path_m
            nav.update(success_gt=bool(ok_gt), final_goal_dist_gt_m=dist, path_length_gt_m=used,
                       optimal_path_m=opt,
                       optimal_path_source="ground_truth_astar",
                       ai2thor_shortest_path_m=self.ai2thor_shortest_path_m,
                       ground_truth_astar_m=self.ground_truth_astar_m,
                       start_goal_euclid_m=math.hypot(*self.goal_local),
                       path_efficiency=(opt / used) if (opt and used > 0) else None,
                       spl=(float(ok_gt) * opt / max(used, opt)) if opt else None)
            if final_rc is not None:
                lx, lz = self._world_to_local(pos["x"], pos["z"])
                gt_rc = (round(self.half_grid - lz / G), round(self.half_grid + lx / G))
                nav["pose_error_m"] = math.hypot(gt_rc[0] - final_rc[0], gt_rc[1] - final_rc[1]) * G

        S = self.steps
        timing = dict(
            total_wall_s=wall,
            per_step_wall_ms=_stats([s["wall_ms"] for s in S]),
            per_step_compute_ms=_stats([s["compute_ms"] for s in S]),
            per_step_sim_ms=_stats([s["sim_ms"] for s in S]),
            per_step_plot_ms=_stats([s["plot_ms"] for s in S]),
            total_compute_s=sum(s["compute_ms"] for s in S) / 1000,
            total_sim_s=sum(s["sim_ms"] for s in S) / 1000,
            total_plot_s=sum(s["plot_ms"] for s in S) / 1000,
            compute_stages=list(self.compute_stages),
            per_call_ms={k: _stats([v * 1000 for v in vs]) for k, vs in self.call_times.items()},
        )
        timing["compute_fraction_of_wall"] = timing["total_compute_s"] / wall if wall else None

        res = self._resource_summary(nav["path_length_m"], len(S))
        vis = dict(frames=len(self.vision),
                   **{k: _stats([v.get(k) for v in self.vision])
                      for k in ("disp_density", "sift_matches", "depth_coverage", "depth_mae_m", "depth_rmse_m",
                                "depth_absrel", "depth_delta1", "disp_bad1")})
        mp = None
        if grid is not None:
            mp = dict(final=self._map_metrics(grid),
                      max_phantom_obstacle_cells=max((m["phantom_obstacle_cells"] for m in self.map_series), default=None),
                      max_false_free_cells=max((m["false_free_cells"] for m in self.map_series), default=None))

        summary = dict(meta=dict(method=self.method_name, scene=self.scene, grid_size=G,
                                 episode_id=self.episode_id, **self.extra_meta,
                                 goal_local_m=self.goal_local, goal_world=self.goal_world,
                                 timestamp=datetime.now().isoformat(timespec="seconds"),
                                 hardware=self.sampler.info),
                       navigation=nav, timing=timing, resources=res, vision=vis, map=mp)
        if self.verbose:
            self._print_report(summary)
        if save:
            self._save(summary, save_csv)
        self.detach()
        return summary

    def _resource_summary(self, path_m, n_steps):
        rows = self.sampler.samples
        run = [r for r in rows if r["phase"] == "run"]
        idle = [r for r in rows if r["phase"] == "idle"]
        col = lambda rs, k: [r[k] for r in rs if r.get(k) is not None]
        out = {k: _stats(col(run, k)) for k in self.RES_KEYS}
        measured = {k: _mean(col(idle, k)) for k in self.RES_KEYS}
        if any(v is not None for v in measured.values()):
            self.last_idle_measured = measured
        out["idle_mean"] = self.idle_baseline or measured
        out["n_samples"] = len(run)
        out["sample_interval_s"] = self.sampler.interval

        def energy(key):
            pts = [(r["t"], r[key]) for r in run if r.get(key) is not None]
            if len(pts) < 2:
                return None, None
            gross = sum(0.5 * (a[1] + b[1]) * (b[0] - a[0]) for a, b in zip(pts, pts[1:]))
            idle_w = out["idle_mean"].get(key)
            net = gross - idle_w * (pts[-1][0] - pts[0][0]) if idle_w is not None else None
            return float(gross), (None if net is None else float(net))

        g, gn = energy("gpu_power_w")
        cp, cpn = energy("cpu_power_w")
        out["gpu_energy_j"], out["gpu_energy_net_j"] = g, gn
        out["cpu_energy_j"], out["cpu_energy_net_j"] = cp, cpn
        tot = (g or 0) + (cp or 0)
        out["total_energy_j"] = tot if (g is not None or cp is not None) else None
        out["energy_per_m_j"] = tot / path_m if (out["total_energy_j"] and path_m > 0) else None
        out["energy_per_step_j"] = tot / n_steps if (out["total_energy_j"] and n_steps) else None
        out["note"] = ("gpu_* includes the Unity renderer; compare against cpu_py/rss_py/gpu_mem_proc "
                       "for the algorithm's own footprint")
        return out

    def _print_report(self, s):
        n, t, r, v, m = s["navigation"], s["timing"], s["resources"], s["vision"], s["map"]
        mean = lambda d, k="mean": None if d is None else d.get(k)
        print(f"\n===== METRICS [{s['meta']['method']} | {s['meta']['scene']}] =====")
        print(f"Outcome    : {'SUCCESS' if n['success'] else 'FAIL'} ({n['termination']})  "
              f"steps={n['nav_steps']}  wall={_f(t['total_wall_s'])}s")
        print(f"Navigation : path={_f(n['path_length_m'], '.2f')}m  optimal={_f(n.get('optimal_path_m'), '.2f')}m  "
              f"SPL={_f(n.get('spl'), '.2f')}  collisions={n['collisions']} (rate {n['collision_rate']:.2f})  "
              f"no-path={n['no_path_events']}  rotations={n['rotations']}")
        if "success_gt" in n:
            print(f"Ground truth: success_gt={n['success_gt']}  goal_dist={_f(n['final_goal_dist_gt_m'], '.2f')}m  "
                  f"pose_err={_f(n.get('pose_error_m'), '.2f')}m")
        print(f"Time       : compute={_f(mean(t['per_step_compute_ms']))} ms/step "
              f"(p95 {_f(mean(t['per_step_compute_ms'], 'p95'))})  sim={_f(mean(t['per_step_sim_ms']))} ms/step  "
              f"plot={_f(mean(t['per_step_plot_ms']))} ms/step")
        print(f"CPU        : python={_f(mean(r['cpu_py_pct']), '.0f')}% (peak {_f(mean(r['cpu_py_pct'], 'max'), '.0f')})  "
              f"sim={_f(mean(r['cpu_sim_pct']), '.0f')}%  system={_f(mean(r['cpu_sys_pct']), '.0f')}%  (% of one core)")
        print(f"Memory     : python RSS={_f(mean(r['rss_py_mb']), '.0f')} MB (peak {_f(mean(r['rss_py_mb'], 'max'), '.0f')})  "
              f"sim RSS={_f(mean(r['rss_sim_mb']), '.0f')} MB")
        print(f"GPU        : util={_f(mean(r['gpu_util_pct']), '.0f')}%  mem={_f(mean(r['gpu_mem_used_mb']), '.0f')} MB  "
              f"(this process {_f(mean(r['gpu_mem_proc_mb']), '.0f')} MB)")
        print(f"Power      : GPU={_f(mean(r['gpu_power_w']))} W  CPU={_f(mean(r['cpu_power_w']))} W  "
              f"energy={_f(r['total_energy_j'], '.0f')} J  ({_f(r['energy_per_m_j'], '.0f')} J/m)")
        print(f"Vision     : disp density={_f(mean(v['disp_density']), '.3f')}  SIFT/frame={_f(mean(v['sift_matches']), '.0f')}  "
              f"depth AbsRel={_f(mean(v['depth_absrel']), '.3f')}  d<1.25={_f(mean(v['depth_delta1']), '.3f')}  "
              f"coverage={_f(mean(v['depth_coverage']), '.3f')}")
        if m and m.get("final"):
            fm = m["final"]
            print(f"Map vs GT  : free precision={_f(fm['free_precision'], '.3f')}  "
                  f"phantom obstacles={fm['phantom_obstacle_cells']} ({_f(fm['phantom_obstacle_rate'], '.2f')})  "
                  f"false-free={fm['false_free_cells']}  reachable coverage={_f(fm['reachable_coverage'], '.3f')}")
        print("=" * 60)

    def _save(self, summary, save_csv=True):
        os.makedirs(self.out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eid = self.episode_id if self.episode_id else self.scene
        base = os.path.join(self.out_dir, f"episode_{self.method_name}_{eid}_{stamp}")
        with open(base + ".json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        if save_csv:
            _write_csv(base + "_steps.csv", self.steps)
            _write_csv(base + "_samples.csv", self.sampler.samples)
            _write_csv(base + "_vision.csv", self.vision)
            _write_csv(base + "_map.csv", self.map_series)
        if self.verbose:
            print(f"[metrics] saved -> {base}.json" + (" (+ csv)" if save_csv else ""))

    # ======================= multi-episode aggregation =====================
    @staticmethod
    def aggregate(out_dir="metrics_out", group_by="method"):
        """Success rate, avg successful path length, etc. across every episode_*.json in out_dir."""
        runs = []
        for p in sorted(glob.glob(os.path.join(out_dir, "episode_*.json"))):
            with open(p) as f:
                runs.append(json.load(f))
        by_method = defaultdict(list)
        for r in runs:
            by_method[str(r["meta"].get(group_by))].append(r)

        results = {}
        for method, rs in by_method.items():
            ok = [r for r in rs if r["navigation"]["success"]]
            col = lambda key, rr=rs: [_get(r, key) for r in rr if _get(r, key) is not None]
            results[method] = dict(
                episodes=len(rs),
                success_rate=len(ok) / len(rs),
                success_rate_gt=_mean([r["navigation"].get("success_gt") for r in rs
                                       if r["navigation"].get("success_gt") is not None]),
                avg_successful_path_m=_mean([r["navigation"]["path_length_m"] for r in ok]),
                mean_spl=_mean(col("navigation.spl")),
                mean_collisions=_mean(col("navigation.collisions")),
                mean_compute_ms_per_step=_mean(col("timing.per_step_compute_ms.mean")),
                mean_cpu_py_pct=_mean(col("resources.cpu_py_pct.mean")),
                peak_rss_py_mb=max(col("resources.rss_py_mb.max"), default=None),
                mean_gpu_util_pct=_mean(col("resources.gpu_util_pct.mean")),
                mean_gpu_power_w=_mean(col("resources.gpu_power_w.mean")),
                mean_energy_j=_mean(col("resources.total_energy_j")),
                mean_depth_absrel=_mean(col("vision.depth_absrel.mean")),
                mean_phantom_obstacles=_mean(col("map.final.phantom_obstacle_cells")),
            )
        if results:
            with open(os.path.join(out_dir, f"aggregate_by_{group_by}.json"), "w") as f:
                json.dump(results, f, indent=2, default=_json_default)
        for method, a in results.items():
            print(f"\n--- {method}: {a['episodes']} episode(s) ---")
            for k, val in a.items():
                if k != "episodes":
                    print(f"  {k:<28}{_f(val, '.3f')}")
        return results


if __name__ == "__main__":
    import sys
    MetricsTracker.aggregate(sys.argv[1] if len(sys.argv) > 1 else "metrics_out",
                             sys.argv[2] if len(sys.argv) > 2 else "method")
