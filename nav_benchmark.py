"""
nav_benchmark.py -- run the classical stereo navigator over RoboTHOR scenes / dataset start poses
and log nav_metrics for every instance. Imports your navigator module; does not modify it.

An "instance" = one start pose (position + heading) in one scene. The goal is the reachable
point with the largest straight-line (Euclidean) distance from the start whose shortest path
also stays inside the robot's local grid. The planner only ever receives the relative
(right, ahead) offset -- the simulator's reachable-position list is used just to pick the task.

Examples
  # all 1800 val episodes' start poses (needs the robothor-challenge dataset folder)
  python nav_benchmark.py --module roomtesting --split val --dataset ./robothor-challenge/dataset
  # quick pass: 10 evenly spaced start poses per scene
  python nav_benchmark.py --module roomtesting --split val --dataset ./robothor-challenge/dataset --episodes-per-scene 10
  # no dataset: one instance per scene at its default spawn
  python nav_benchmark.py --module roomtesting --split val
  # only write the instance list (start pose + goal) for the deep-learning side
  python nav_benchmark.py --module roomtesting --split val --dataset ... --manifest-only
Runs are resumable: finished instances are skipped on re-run (use --no-resume to redo).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib
import json
import math
import os
import time
import traceback
from collections import OrderedDict

import numpy as np

from nav_metrics import MetricsTracker

HOOKED = ("sync_stereo_rig", "scan_all_directions_stereo", "next_move", "execute_move")


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
def default_scenes(split):
    if split == "train":
        return [f"FloorPlan_Train{a}_{b}" for a in range(1, 13) for b in range(1, 6)]
    if split == "val":
        return [f"FloorPlan_Val{a}_{b}" for a in range(1, 4) for b in range(1, 6)]
    raise ValueError("--dataset is required for split 'test' / 'debug' (scene names come from its files)")


def snap90(yaw):
    return int(round(yaw / 90.0) * 90) % 360


def load_instances(split, dataset_dir, per_scene):
    instances = []
    if dataset_dir:
        files = sorted(glob.glob(os.path.join(dataset_dir, split, "episodes", "*.json.gz")))
        if not files:
            raise FileNotFoundError(f"no episode files under {dataset_dir}/{split}/episodes/")
        for f in files:
            with gzip.open(f, "rt") as fh:
                eps = json.load(fh)
            seen, scene_inst = set(), []
            for e in eps:
                pos = e["initial_position"]
                yaw = snap90(e.get("initial_orientation", 0))   # navigator assumes axis-aligned headings
                key = (round(pos["x"], 2), round(pos["z"], 2), yaw)
                if key in seen:
                    continue
                seen.add(key)
                scene_inst.append(dict(id=e["id"], scene=e["scene"], position=pos, yaw=yaw, source="dataset"))
            if per_scene and len(scene_inst) > per_scene:
                idx = sorted(set(np.linspace(0, len(scene_inst) - 1, per_scene).round().astype(int)))
                scene_inst = [scene_inst[i] for i in idx]
            instances += scene_inst
    else:
        for sc in default_scenes(split):
            instances.append(dict(id=f"{sc}_default", scene=sc, position=None, yaw=None, source="scene_default"))
    return instances


# ---------------------------------------------------------------------------
# goal selection: farthest Euclidean reachable point, path kept inside the local grid
# ---------------------------------------------------------------------------
def farthest_goal(controller, grid_size, half_grid, max_tries=300):
    ag = controller.last_event.metadata["agent"]
    sx, sz = ag["position"]["x"], ag["position"]["z"]
    yaw = math.radians(ag["rotation"]["y"])
    c, s = math.cos(yaw), math.sin(yaw)
    limit = (half_grid - 1) * grid_size            # +/-9.75 m for the default 81x81 grid

    def to_local(x, z):                            # world -> (right, ahead) in the spawn frame
        dx, dz = x - sx, z - sz
        return dx * c - dz * s, dx * s + dz * c

    reach = controller.step(action="GetReachablePositions").metadata["actionReturn"] or []
    cands = []
    for p in reach:
        lx, lz = to_local(p["x"], p["z"])
        if abs(lx) <= limit and abs(lz) <= limit:
            cands.append((math.hypot(lx, lz), lx, lz, p))
    cands.sort(key=lambda t: (-round(t[0], 6), t[3]["x"], t[3]["z"]))    # deterministic order

    for euclid, lx, lz, p in cands[:max_tries]:
        ev = controller.step(action="GetShortestPathToPoint", target=dict(x=p["x"], y=p["y"], z=p["z"]))
        ret = ev.metadata.get("actionReturn")
        corners = ret.get("corners") if isinstance(ret, dict) else ret
        if not ev.metadata.get("lastActionSuccess") or not corners:
            continue
        pts = [(sx, sz)] + [(q["x"], q["z"]) for q in corners]
        if not all(abs(v) <= limit for q in pts for v in to_local(*q)):
            continue                               # path leaves the local grid -> not usable
        geo = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        offset = (round(lx / grid_size) * grid_size, round(lz / grid_size) * grid_size)
        return offset, dict(euclid_m=euclid, geodesic_m=geo, ratio=geo / max(euclid, 1e-6),
                            n_candidates=len(cands))
    return None, None


def fresh_state(nav, goal_offset):
    """Same grid setup as the tail of init_environment(), for a new goal."""
    H, G = nav.HALF_GRID_CELLS, nav.GRID_SIZE
    size = 2 * H + 1
    grid = np.full((size, size), nav.UNKNOWN_CELL, dtype=int)
    log_odds = np.zeros((size, size), dtype=np.float32)
    rob_rc = (H, H)
    exit_rc = (int(np.clip(H - round(goal_offset[1] / G), 1, size - 2)),
               int(np.clip(H + round(goal_offset[0] / G), 1, size - 2)))
    log_odds[rob_rc] = -4.0
    grid[exit_rc] = nav.EXIT_LOCATION
    grid[rob_rc] = nav.ROBOT_LOCATION
    return grid, log_odds, rob_rc, exit_rc


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------
def run_instance(nav, orig, controller, pitch_deg, inst, args, out_dir, idle_baseline):
    if inst["position"] is not None:
        ev = controller.step(action="Teleport", position=inst["position"],
                             rotation=dict(x=0, y=inst["yaw"], z=0), horizon=pitch_deg)
        if not ev.metadata["lastActionSuccess"]:
            return dict(error="teleport_failed: " + str(ev.metadata.get("errorMessage", ""))[:120])

    goal_offset, ginfo = farthest_goal(controller, nav.GRID_SIZE, nav.HALF_GRID_CELLS)
    if goal_offset is None:
        return dict(error="no_valid_goal")

    ag = controller.last_event.metadata["agent"]
    with open(os.path.join(out_dir, "manifest.jsonl"), "a") as f:
        f.write(json.dumps(dict(id=inst["id"], scene=inst["scene"], start_position=ag["position"],
                                start_yaw=ag["rotation"]["y"], goal_offset_m=goal_offset, **ginfo)) + "\n")
    if args.manifest_only:
        return dict(manifest_only=True, goal_offset=goal_offset, **ginfo)

    grid, log_odds, rob_rc, exit_rc = fresh_state(nav, goal_offset)
    mapper = nav.ClassicalStereoMapper.from_frame(controller.last_event.frame, pitch_deg=pitch_deg)

    metrics = MetricsTracker(
        method_name=args.method_name, scene=inst["scene"], grid_size=nav.GRID_SIZE,
        half_grid=nav.HALF_GRID_CELLS, out_dir=out_dir, depth_range=(0.30, nav.MAX_RANGE),
        episode_id=inst["id"], verbose=False, idle_baseline=idle_baseline,
        extra_meta=dict(split=args.split, source=inst["source"], start_yaw_snapped=inst["yaw"],
                        goal_selection="farthest_euclid", goal_info=ginfo, max_steps=args.max_steps))
    try:
        metrics.attach(controller, goal_offset)
        metrics.start(baseline_s=args.baseline_s if idle_baseline is None else 0.0)

        # the navigator looks these names up as module globals -> hook them on the module
        nav.sync_stereo_rig = metrics.wrap("stereo_capture", orig["sync_stereo_rig"])
        nav.scan_all_directions_stereo = metrics.wrap("scan_total", orig["scan_all_directions_stereo"])
        nav.next_move = metrics.wrap_planner(orig["next_move"])
        nav.execute_move = metrics.wrap_executor(orig["execute_move"])
        metrics.wrap_mapper(mapper)

        # ---- same loop as your __main__ (minus the live plot) ----
        local_yaw, visited_vision, visited_free, retries = 0, set(), {rob_rc}, 0
        for step in range(args.max_steps):
            metrics.begin_step(step, grid)
            if rob_rc not in visited_vision:
                _, _, local_yaw = nav.scan_all_directions_stereo(
                    controller, mapper, grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw)
                visited_vision.add(rob_rc)

            move_name, path = nav.next_move(grid)
            if move_name is None:
                if retries < 6:
                    retries += 1
                    metrics.note("replan_retry")
                    weak = (log_odds > 0) & (log_odds < nav.BUMP_LOCK)
                    log_odds[weak] *= 0.35
                    nav.threshold_grid(grid, log_odds, visited_free, rob_rc, exit_rc)
                    continue
                break
            retries = 0

            success, rob_rc, local_yaw = nav.execute_move(
                controller, move_name, grid, log_odds, rob_rc, exit_rc, local_yaw)
            if success:
                visited_free.add(rob_rc)
            if rob_rc == exit_rc:
                break

        summary = metrics.finish(success=(rob_rc == exit_rc), final_rc=rob_rc, exit_rc=exit_rc,
                                 grid=grid, max_steps=args.max_steps, save_csv=not args.no_csv)
        return dict(summary=summary, idle=metrics.last_idle_measured)
    finally:
        for k, fn in orig.items():
            setattr(nav, k, fn)
        metrics.detach()
        metrics.sampler.stop()


def already_done(out_dir, method, inst):
    return bool(glob.glob(os.path.join(out_dir, f"episode_{method}_{inst['id']}_*.json")))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="roomtesting", help="your navigator module (file name without .py)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "debug"])
    ap.add_argument("--dataset", default=None, help="path to robothor-challenge/dataset")
    ap.add_argument("--episodes-per-scene", type=int, default=0, help="0 = every start pose")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="metrics_out")
    ap.add_argument("--method-name", default="classical_stereo")
    ap.add_argument("--baseline-s", type=float, default=5.0, help="idle hardware baseline, measured once")
    ap.add_argument("--no-csv", action="store_true", help="JSON only (recommended for large runs)")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    nav = importlib.import_module(args.module)
    orig = {k: getattr(nav, k) for k in HOOKED}
    instances = load_instances(args.split, args.dataset, args.episodes_per_scene)
    by_scene = OrderedDict()
    for i in instances:
        by_scene.setdefault(i["scene"], []).append(i)

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    if args.manifest_only:
        open(os.path.join(out_dir, "manifest.jsonl"), "w").close()
    print(f"{len(instances)} instances in {len(by_scene)} scenes | split={args.split} | out={out_dir}")

    idle, done, t_start = None, 0, time.time()
    for scene, insts in by_scene.items():
        todo = [i for i in insts if args.manifest_only or args.no_resume
                or not already_done(out_dir, args.method_name, i)]
        if not todo:
            done += len(insts)
            continue
        nav.SCENE_NAME = scene
        try:
            ret = nav.init_environment()
            controller, pitch_deg = ret[0], ret[5]
        except Exception as e:
            print(f"[skip scene] {scene}: could not load ({type(e).__name__}: {str(e)[:100]})")
            continue
        if getattr(controller.last_event, "depth_frame", None) is not None:
            print("[warn] renderDepthImage is on: depth is rendered every step, so GPU/time/power "
                  "numbers are inflated. Set renderDepthImage=False for benchmark runs.")
        consecutive_errors = 0
        try:
            for inst in todo:
                done += 1
                t0 = time.time()
                try:
                    res = run_instance(nav, orig, controller, pitch_deg, inst, args, out_dir, idle)
                except Exception:
                    res = dict(error="exception")
                    with open(os.path.join(out_dir, "errors.log"), "a") as f:
                        f.write(f"--- {inst['id']}\n{traceback.format_exc()}\n")
                if res.get("error"):
                    consecutive_errors += 1
                    print(f"[{done}/{len(instances)}] {inst['id']}: SKIPPED ({res['error']})")
                    if consecutive_errors >= 3:
                        print(f"[skip rest of scene] {scene}: 3 consecutive errors (see errors.log)")
                        break
                    continue
                consecutive_errors = 0
                if res.get("manifest_only"):
                    print(f"[{done}/{len(instances)}] {inst['id']}: goal {res['goal_offset']} "
                          f"(euclid {res['euclid_m']:.1f} m, geodesic {res['geodesic_m']:.1f} m)")
                    continue
                idle = res["idle"] or idle
                n = res["summary"]["navigation"]
                spl = n.get("spl")
                print(f"[{done}/{len(instances)}] {inst['id']}: {'OK ' if n['success'] else 'FAIL'} "
                      f"({n['termination']}) steps={n['nav_steps']} path={n['path_length_m']:.1f} m "
                      f"coll={n['collisions']} SPL={'n/a' if spl is None else format(spl, '.2f')} "
                      f"[{time.time() - t0:.0f}s]")
        finally:
            try:
                controller.stop()
            except Exception:
                pass

    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min")
    if not args.manifest_only:
        MetricsTracker.aggregate(out_dir, "method")
        print(f"\nPer-scene breakdown:  python nav_metrics.py {out_dir} scene")


if __name__ == "__main__":
    main()