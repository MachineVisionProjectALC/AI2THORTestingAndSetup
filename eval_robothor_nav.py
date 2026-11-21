"""
eval_robothor_val.py -- Batch evaluator for roomtesting.py on RoboTHOR Val1_1..Val3_5.
Goal per scene: farthest Euclidean-distance reachable point from spawn (4-connected to start).
Generates a 4-page PDF benchmark report (classical stereo baseline vs. deep learning comparison)
and a companion CSV with per-scene metrics and exact goal offsets.
"""
import csv
import math
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless rendering; must be set before importing roomtesting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import roomtesting as rt

# =====================================================================
# BENCHMARK CONFIGURATION
# =====================================================================
SCENES = [f"FloorPlan_Val{i}_{j}" for i in range(1, 4) for j in range(1, 6)]
OUT_PDF = "classical_stereo_val_report.pdf"
OUT_CSV = "classical_stereo_val_metrics.csv"

ERR_COLORS = ["#d9d9d9", "#c6dbef", "#b8e6b8", "#2c3e50", "#e41a1c", "#ff9f1c"]
ERR_LABELS = [
    "unknown", "reachable unmapped", "free: correct",
    "blocked: correct", "FALSE-FREE (collision risk)", "PHANTOM (blocked, reachable)",
]


# =====================================================================
# 1. GOAL SELECTION & SINGLE-SCENE HEADLESS RUNNER
# =====================================================================
def select_farthest_reachable_goal(gt_free: np.ndarray, start_rc: tuple[int, int]):
    """
    Finds the reachable grid cell in the same 4-connected component as start_rc
    that maximizes Euclidean distance from start_rc.
    Returns (goal_rc, goal_dx_m, goal_dz_m, goal_euclid_m).
    """
    rows, cols = gt_free.shape
    sr, sc = start_rc
    visited = {start_rc}
    q = deque([start_rc])
    while q:
        r, c = q.popleft()
        for dr, dc in rt.CARDINAL_MOVES.values():
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and gt_free[nb] and nb not in visited:
                visited.add(nb)
                q.append(nb)

    candidates = [rc for rc in visited if rc != start_rc]
    if not candidates:  # fallback if spawn cell had no 4-connected GT neighbors
        candidates = [
            (int(r), int(c)) for r, c in np.argwhere(gt_free) if (int(r), int(c)) != start_rc
        ]

    # Deterministic tie-breaking via (dist_sq, r, c)
    goal_rc = max(candidates, key=lambda rc: ((rc[0] - sr) ** 2 + (rc[1] - sc) ** 2, rc[0], rc[1]))
    goal_dx_m = (goal_rc[1] - sc) * rt.GRID_SIZE
    goal_dz_m = (sr - goal_rc[0]) * rt.GRID_SIZE
    goal_euclid_m = math.hypot(goal_dx_m, goal_dz_m)
    return goal_rc, goal_dx_m, goal_dz_m, goal_euclid_m


def run_scene(scene_name: str):
    rt.SCENE_NAME = scene_name
    controller, occ_grid, log_odds, rob_rc, default_exit_rc, pitch_deg = rt.init_environment()
    try:
        gt = rt.capture_gt(controller)

        # Override fixed GOAL_OFFSET_M with farthest Euclidean reachable point from spawn
        if default_exit_rc != rob_rc:
            occ_grid[default_exit_rc] = rt.UNKNOWN_CELL
        exit_rc, goal_dx_m, goal_dz_m, goal_euclid_m = select_farthest_reachable_goal(
            gt["gt_free"], rob_rc
        )
        occ_grid[exit_rc] = rt.EXIT_LOCATION

        left0, _ = rt.sync_stereo_rig(controller)
        mapper = rt.ClassicalStereoMapper.from_frame(left0, pitch_deg=pitch_deg)
        rt.TIMES.clear()

        local_yaw = 0
        visited_free = {rob_rc}
        frames = moves_ok = collisions = rotations = steps = 0

        _, local_yaw = rt.scan_all_directions_stereo(
            controller, mapper, occ_grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw
        )
        frames += 4
        rotations += 4
        scanned = {rob_rc}
        stop_reason = "step_limit"

        for step in range(rt.MAX_STEPS):
            steps = step + 1
            if rob_rc == exit_rc:
                stop_reason = "reached"
                break

            with rt.timed("planning"):
                move_name, path = rt.next_move(occ_grid)
                if move_name is None and not path:
                    strict = occ_grid.copy()
                    rt.threshold_grid(strict, log_odds, visited_free, rob_rc, exit_rc, strict=True)
                    move_name, path = rt.next_move(strict)

            if move_name is None:
                if rob_rc not in scanned:
                    _, local_yaw = rt.scan_all_directions_stereo(
                        controller, mapper, occ_grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw
                    )
                    frames += 4
                    rotations += 4
                    scanned.add(rob_rc)
                    continue
                stop_reason = "no_path"
                break

            if rt.MOVE_TO_YAW[move_name] != local_yaw:
                local_yaw = rt.turn_toward(controller, move_name, local_yaw)
                rotations += 1
                rt.observe_forward(controller, mapper, occ_grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw)
                frames += 1
                continue

            dr, dc = rt.CARDINAL_MOVES[move_name]
            target_rc = (rob_rc[0] + dr, rob_rc[1] + dc)
            success, rob_rc, local_yaw = rt.execute_move(
                controller, move_name, occ_grid, log_odds, rob_rc, exit_rc, local_yaw
            )
            if success:
                moves_ok += 1
                visited_free.add(rob_rc)
                rt.observe_forward(controller, mapper, occ_grid, log_odds, visited_free, rob_rc, exit_rc, local_yaw)
                frames += 1
            else:
                collisions += 1
                if target_rc == exit_rc:
                    stop_reason = "goal_blocked"
                    break

        rt.threshold_grid(occ_grid, log_odds, visited_free, rob_rc, exit_rc)
        res = rt.score_episode(controller, gt, occ_grid, exit_rc, moves_ok, collisions, rotations, frames, steps)
        ex = mapper.last
        res["scene"] = scene_name
        res["goal_dx_m"] = goal_dx_m
        res["goal_dz_m"] = goal_dz_m
        res["goal_euclid_m"] = goal_euclid_m
        res["stop_reason"] = "reached" if res["success"] else stop_reason
        res["ransac_pitch_deg"] = ex.get("pitch_est", float("nan"))
        res["pitch_err_deg"] = abs(ex["pitch_est"] - pitch_deg) if "pitch_est" in ex else float("nan")
        res["ransac_cam_h_m"] = ex.get("cam_h_est", float("nan"))
        res["cam_h_err_m"] = abs(ex["cam_h_est"] - rt.CAMERA_HEIGHT) if "cam_h_est" in ex else float("nan")
        err_img = rt.error_map(occ_grid, gt["gt_free"])
        return res, err_img
    finally:
        controller.stop()


# =====================================================================
# 2. METRIC AGGREGATION & PDF REPORT GENERATION
# =====================================================================
def _valid(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(float(v))
    return out


def _ms(rows, key, fmt="{:.3f}"):
    v = _valid(rows, key)
    return f"{fmt.format(np.mean(v))} ± {fmt.format(np.std(v))}" if v else "n/a"


def _fmt(v, fmt="{:.2f}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return fmt.format(v)


def save_csv(rows, path=OUT_CSV):
    keys = [
        "scene", "goal_dx_m", "goal_dz_m", "goal_euclid_m", "success", "stop_reason",
        "spl", "path_m", "optimal_m", "collisions", "rotations", "nav_steps", "frames",
        "blocked_precision", "blocked_recall", "blocked_f1",
        "false_free_rate", "phantom_rate", "coverage",
        "disparity_ms_per_call", "mapping_ms_per_call", "planning_ms_per_call",
        "perception_ms_per_frame", "ransac_pitch_deg", "pitch_err_deg",
        "ransac_cam_h_m", "cam_h_err_m",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_pdf(rows, err_maps, out_pdf=OUT_PDF):
    n_all = len(rows)
    succ_all = sum(1 for r in rows if r["success"])

    with PdfPages(out_pdf) as pdf:
        # ---- PAGE 1: OVERVIEW, FILE 2 SPEC & AGGREGATE SUMMARY ----
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            "Classical Stereo Navigation Benchmark — RoboTHOR Val1_1 to Val3_5",
            fontsize=15, fontweight="bold", y=0.96,
        )
        fig.text(
            0.5, 0.918,
            "Source: roomtesting.py  |  Goal: Farthest Euclidean Reachable Point  |  Baseline vs. Deep Learning Approach",
            ha="center", fontsize=9.8, color="#34495e",
        )

        # Left column: Benchmark context + What's in File 2
        ax_txt = fig.add_axes([0.04, 0.05, 0.46, 0.83])
        ax_txt.axis("off")
        spec_text = (
            "ACTIVE ABLATION & RUN SETTINGS\n"
            f"   USE_CLAHE={rt.USE_CLAHE}  |  USE_RANSAC_GROUND={rt.USE_RANSAC_GROUND}  |  USE_RAYCAST={rt.USE_RAYCAST}\n"
            f"   GOAL=Farthest Point Reachable  |  MAX_STEPS={rt.MAX_STEPS}"
        )
        ax_txt.text(
            0.0, 1.0, spec_text, va="top", ha="left", fontsize=8.1, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f9fb", edgecolor="#cbd5e1"),
        )

        # Right column: Aggregate metrics table
        ax_kpi = fig.add_axes([0.53, 0.08, 0.44, 0.80])
        ax_kpi.axis("off")
        ax_kpi.set_title("Aggregate Metrics (15 RoboTHOR Val Scenes)", fontsize=11, fontweight="bold", pad=8)

        kpi_data = [
            ["Success Rate", f"{succ_all}/{n_all} ({100*succ_all/max(n_all,1):.1f}%)"],
            ["SPL (Success weighted by Path Len)", _ms(rows, "spl")],
            ["Goal Euclidean Distance (m)", _ms(rows, "goal_euclid_m", "{:.2f}")],
            ["Optimal GT Path Length (m)", _ms(rows, "optimal_m", "{:.2f}")],
            ["Robot Path Length (m)", _ms(rows, "path_m", "{:.2f}")],
            ["Collisions (Mean ± Std | Total)", f"{_ms(rows, 'collisions', '{:.1f}')} | {sum(r['collisions'] for r in rows)}"],
            ["Rotations / Episode", _ms(rows, "rotations", "{:.1f}")],
            ["Nav Steps / Episode", _ms(rows, "nav_steps", "{:.1f}")],
            ["Stereo Frames / Episode", _ms(rows, "frames", "{:.1f}")],
            ["Blocked Precision (C-space)", _ms(rows, "blocked_precision")],
            ["Blocked Recall (C-space)", _ms(rows, "blocked_recall")],
            ["Blocked F1 (C-space)", _ms(rows, "blocked_f1")],
            ["False-Free Rate (Collision Risk)", _ms(rows, "false_free_rate")],
            ["Phantom Obstacle Rate", _ms(rows, "phantom_rate")],
            ["Reachable Coverage", _ms(rows, "coverage")],
            ["RANSAC |ΔPitch| vs True (deg)", _ms(rows, "pitch_err_deg", "{:.2f}")],
            ["RANSAC |ΔCam Height| vs True (m)", _ms(rows, "cam_h_err_m", "{:.3f}")],
            ["Disparity Latency (ms / call)", _ms(rows, "disparity_ms_per_call", "{:.2f}")],
            ["Mapping Latency (ms / call)", _ms(rows, "mapping_ms_per_call", "{:.2f}")],
            ["Planning Latency (ms / call)", _ms(rows, "planning_ms_per_call", "{:.2f}")],
            ["Total Perception (ms / frame)", _ms(rows, "perception_ms_per_frame", "{:.2f}")],
        ]
        tbl = ax_kpi.table(
            cellText=kpi_data, colLabels=["Metric (File 2)", "Classical Stereo Result"],
            loc="center", cellLoc="left", colWidths=[0.58, 0.42],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.42)
        for (r_i, c_i), cell in tbl.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if r_i == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i % 2 == 0:
                cell.set_facecolor("#f8fafc")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- PAGE 2: PER-SCENE METRICS TABLE ----
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(
            "Per-Scene Evaluation Results — RoboTHOR Val1_1 to Val3_5 (Farthest Euclidean Goal)",
            fontsize=13, fontweight="bold", pad=12,
        )

        cols = [
            "Scene", "Dist(m)", "Succ", "Stop", "SPL", "Path/Opt",
            "Coll", "Rot", "Step", "Frm", "Prec", "Rec", "F1",
            "F-Free", "Phant", "Cov", "ms/fr", "|Δp|°", "|Δh|m",
        ]
        body = []
        for r in rows:
            short = r["scene"].replace("FloorPlan_", "")
            path_opt = f"{_fmt(r['path_m'], '{:.1f}')}/{_fmt(r['optimal_m'], '{:.1f}')}"
            body.append([
                short,
                _fmt(r["goal_euclid_m"], "{:.2f}"),
                "PASS" if r["success"] else "FAIL",
                r["stop_reason"][:10],
                _fmt(r["spl"], "{:.2f}"),
                path_opt,
                str(r["collisions"]),
                str(r["rotations"]),
                str(r["nav_steps"]),
                str(r["frames"]),
                _fmt(r["blocked_precision"]),
                _fmt(r["blocked_recall"]),
                _fmt(r["blocked_f1"]),
                _fmt(r["false_free_rate"]),
                _fmt(r["phantom_rate"]),
                _fmt(r["coverage"]),
                _fmt(r["perception_ms_per_frame"], "{:.1f}"),
                _fmt(r["pitch_err_deg"], "{:.1f}"),
                _fmt(r["cam_h_err_m"], "{:.2f}"),
            ])

        m = lambda k, f="{:.2f}": _fmt(np.mean(_valid(rows, k)) if _valid(rows, k) else None, f)
        body.append([
            "MEAN/TOT", m("goal_euclid_m"), f"{succ_all}/{n_all}", "—",
            m("spl"), f"{m('path_m', '{:.1f}')}/{m('optimal_m', '{:.1f}')}",
            str(sum(r["collisions"] for r in rows)),
            m("rotations", "{:.0f}"), m("nav_steps", "{:.0f}"), m("frames", "{:.0f}"),
            m("blocked_precision"), m("blocked_recall"), m("blocked_f1"),
            m("false_free_rate"), m("phantom_rate"), m("coverage"),
            m("perception_ms_per_frame", "{:.1f}"),
            m("pitch_err_deg", "{:.1f}"), m("cam_h_err_m", "{:.2f}"),
        ])

        t2 = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(7.4)
        t2.scale(1.0, 1.65)
        for (r_i, c_i), cell in t2.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if r_i == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r_i == len(body):
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(fontweight="bold")
            else:
                if r_i % 2 == 0:
                    cell.set_facecolor("#f8fafc")
                if c_i == 2:
                    txt = cell.get_text().get_text()
                    cell.set_facecolor("#dcfce7" if txt == "PASS" else "#fee2e2")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- PAGE 3: DIAGNOSTIC CHARTS (2x2) ----
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        labels = [r["scene"].replace("FloorPlan_", "") for r in rows]
        x = np.arange(len(rows))

        # (1) SPL & Outcome
        ax = axes[0, 0]
        spls = [r["spl"] if r["spl"] is not None else 0.0 for r in rows]
        bar_c = ["#2ecc71" if r["success"] else "#e74c3c" for r in rows]
        ax.bar(x, spls, color=bar_c, edgecolor="#2c3e50", linewidth=0.5)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("SPL"); ax.set_title("SPL by Scene (Green=Pass, Red=Fail)")
        ax.grid(axis="y", alpha=0.3)

        # (2) Occupancy Grid Quality (C-space)
        ax = axes[0, 1]
        for key, lbl, clr in [
            ("blocked_f1", "Blocked F1", "#2b83ba"),
            ("blocked_precision", "Precision", "#00a86b"),
            ("false_free_rate", "False-Free", "#e41a1c"),
            ("coverage", "Coverage", "#ff9f1c"),
        ]:
            ys = [r[key] if r[key] is not None else np.nan for r in rows]
            ax.plot(x, ys, marker="o", markersize=4, linewidth=1.6, label=lbl, color=clr)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Score / Rate"); ax.set_title("Occupancy Grid Quality (C-space)")
        ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)

        # (3) Collisions & Nav Steps
        ax = axes[1, 0]
        ax.bar(x - 0.18, [r["collisions"] for r in rows], width=0.36, label="Collisions", color="#e74c3c")
        ax2 = ax.twinx()
        ax2.bar(x + 0.18, [r["nav_steps"] for r in rows], width=0.36, label="Nav Steps", color="#3498db", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Collisions", color="#c0392b")
        ax2.set_ylabel("Nav Steps", color="#2980b9")
        ax.set_title("Safety (Collisions) & Episode Length (Nav Steps)")
        ax.grid(axis="y", alpha=0.3)

        # (4) CPU Stage Latency Breakdown (for DL Comparison)
        ax = axes[1, 1]
        stages = [
            ("Disparity\n(ms/call)", "disparity_ms_per_call"),
            ("Mapping\n(ms/call)", "mapping_ms_per_call"),
            ("Planning\n(ms/call)", "planning_ms_per_call"),
            ("Perception\n(ms/frame)", "perception_ms_per_frame"),
        ]
        means = [np.mean(_valid(rows, k)) if _valid(rows, k) else 0.0 for _, k in stages]
        stds = [np.std(_valid(rows, k)) if _valid(rows, k) else 0.0 for _, k in stages]
        ax.bar(range(len(stages)), means, yerr=stds, capsize=4,
               color=["#6366f1", "#0ea5e9", "#10b981", "#f59e0b"], edgecolor="#2c3e50", linewidth=0.6)
        ax.set_xticks(range(len(stages))); ax.set_xticklabels([s[0] for s in stages], fontsize=8.5)
        ax.set_ylabel("Milliseconds (CPU perf_counter)")
        ax.set_title("Compute Latency Breakdown (Classical Baseline)")
        ax.grid(axis="y", alpha=0.3)
        for i, val in enumerate(means):
            ax.text(i, val + stds[i] + 0.5, f"{val:.1f} ms", ha="center", va="bottom", fontsize=8)

        fig.suptitle("Diagnostic Breakdown — Classical Stereo on RoboTHOR Val1_1 to Val3_5",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ---- PAGE 4: 3x5 FINAL ERROR-MAP GALLERY ----
        fig, axes = plt.subplots(3, 5, figsize=(11, 8.5))
        cmap = ListedColormap(ERR_COLORS)
        for idx, (r, err) in enumerate(zip(rows, err_maps)):
            ax = axes[idx // 5, idx % 5]
            ax.imshow(err, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
            kr, kc = np.nonzero(err != rt.E_UNK)
            if len(kr):
                pad = 2
                ax.set_xlim(max(-0.5, kc.min() - pad), min(err.shape[1] - 0.5, kc.max() + pad))
                ax.set_ylim(min(err.shape[0] - 0.5, kr.max() + pad), max(-0.5, kr.min() - pad))
            tag = "OK" if r["success"] else "FAIL"
            ax.set_title(
                f"{r['scene'].replace('FloorPlan_', '')} [{tag}] d={r['goal_euclid_m']:.1f}m",
                fontsize=8, color=("#15803d" if tag == "OK" else "#b91c1c"),
            )
            ax.set_xticks([]); ax.set_yticks([])

        handles = [Patch(color=c, label=l) for c, l in zip(ERR_COLORS, ERR_LABELS)]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=True)
        fig.suptitle("Final Occupancy Grid vs. Ground Truth (C-space) Across All 15 Validation Scenes",
                     fontsize=13, fontweight="bold", y=0.97)
        fig.tight_layout(rect=[0, 0.07, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    all_rows, all_errs = [], []
    for i, scene in enumerate(SCENES, 1):
        print(f"[{i:02d}/{len(SCENES)}] Evaluating {scene} ...", flush=True)
        res, err_img = run_scene(scene)
        all_rows.append(res)
        all_errs.append(err_img)
        print(
            f"   -> goal={res['goal_euclid_m']:.2f}m (dx={res['goal_dx_m']:+.2f}, dz={res['goal_dz_m']:+.2f}) "
            f"succ={res['success']} spl={_fmt(res['spl'])} coll={res['collisions']} "
            f"F1={_fmt(res['blocked_f1'])} perc={_fmt(res['perception_ms_per_frame'], '{:.1f}')}ms/fr"
        )

    save_csv(all_rows, OUT_CSV)
    build_pdf(all_rows, all_errs, OUT_PDF)
    print(f"\nSaved PDF report -> {OUT_PDF}")
    print(f"Saved raw CSV    -> {OUT_CSV}")
