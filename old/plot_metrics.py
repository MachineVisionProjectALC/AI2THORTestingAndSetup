"""
plot_metrics.py -- turn nav_metrics results into one multi-page PDF of charts.

  python plot_metrics.py metrics_out/val
  python plot_metrics.py metrics_out/val metrics_dl/val --out compare.pdf   # several folders -> compare methods
  python plot_metrics.py metrics_out/val --scene-pages                      # (optional flag, see bottom)

Reads every episode_*.json in the folder(s); methods are told apart by meta.method.
If a *_samples.csv exists next to the JSONs (run without --no-csv), a time-series page is added.
Needs only: matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

PAGE = (11, 8.5)
COL = {}


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def get(d, dotted, default=None):
    for k in dotted.split("."):
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def load(dirs):
    runs = defaultdict(list)
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
            with open(p) as f:
                r = json.load(f)
            # Earlier runs with an A* baseline still stored simulator-based ratios.
            nav = r["navigation"]
            opt, used = nav.get("ground_truth_astar_m"), nav.get("path_length_gt_m")
            nav["path_efficiency"] = opt / used if opt is not None and used else None
            nav["spl"] = (float(nav["success_gt"]) * opt / max(used, opt)
                          if opt and used is not None and nav.get("success_gt") is not None else None)
            r["_path"] = p
            runs[r["meta"]["method"]].append(r)
    return runs


def vals(rs, key, scale=1.0):
    out = []
    for r in rs:
        v = get(r, key)
        if v is not None and np.isfinite(float(v)):
            out.append(float(v) * scale)
    return out


def mean_of(rs, key, scale=1.0):
    v = vals(rs, key, scale)
    return float(np.mean(v)) if v else None


def fmt(x, spec=".2f"):
    return "n/a" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def nodata(ax, title, msg="no data"):
    ax.set_title(title, fontsize=9)
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def box(ax, R, key, title, ylabel="", scale=1.0, msg="no data"):
    """One box + jittered points per method, using one value per episode."""
    names, data = [], []
    for m, rs in R.items():
        v = vals(rs, key, scale)
        if v:
            names.append(m)
            data.append(v)
    if not data:
        return nodata(ax, title, msg)
    pos = range(1, len(data) + 1)
    bp = ax.boxplot(data, positions=pos, showfliers=False, patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    rng = np.random.default_rng(0)
    for i, (m, v) in enumerate(zip(names, data)):
        bp["boxes"][i].set(facecolor=COL[m], alpha=0.35)
        ax.scatter(rng.normal(i + 1, 0.05, len(v)), v, s=6, alpha=0.35, color=COL[m])
    ax.set_xticks(list(pos))
    ax.set_xticklabels([f"{n}\n(n={len(d)})" for n, d in zip(names, data)], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)


def success_bars(ax, R):
    ms = list(R)
    x = np.arange(len(ms))
    w = 0.38
    for i, m in enumerate(ms):
        s = np.mean([bool(r["navigation"]["success"]) for r in R[m]])
        g = vals(R[m], "navigation.success_gt")
        ax.bar(i - w / 2, s, w, color=COL[m], alpha=0.9, label="Agent-reported" if i == 0 else None)
        ax.text(i - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=7)
        if g:
            ax.bar(i + w / 2, np.mean(g), w, color=COL[m], alpha=0.45, hatch="//",
                   label="Simulator-verified" if i == 0 else None)
            ax.text(i + w / 2, np.mean(g) + 0.01, f"{np.mean(g):.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={len(R[m])})" for m in ms], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("SLAM success rate", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def termination_bars(ax, R):
    cats = ["goal", "max_steps", "no_path_or_aborted"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    ms = list(R)
    bottom = np.zeros(len(ms))
    for c, colr in zip(cats, colors):
        frac = np.array([np.mean([r["navigation"]["termination"] == c for r in R[m]]) for m in ms])
        ax.bar(range(len(ms)), frac, bottom=bottom, color=colr, label=c)
        bottom += frac
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, fontsize=8)
    ax.set_title("How episodes ended", fontsize=9)
    ax.legend(fontsize=7)


def path_scatter(ax, R):
    title = "SLAM path length vs. ground-truth A*"
    count = 0
    max_distance = 0
    for m, rs in R.items():
        pts = [(n["ground_truth_astar_m"], n["path_length_gt_m"], n["success"])
               for n in (r["navigation"] for r in rs)
               if n.get("ground_truth_astar_m") is not None and n.get("path_length_gt_m") is not None]
        if not pts:
            continue
        count += len(pts)
        opt, used, ok = (np.array(v) for v in zip(*pts))
        max_distance = max(max_distance, opt.max(), used.max())
        ok = ok.astype(bool)
        ax.scatter(opt[ok], used[ok], s=12, color=COL[m], alpha=0.6, label=f"{m} success")
        ax.scatter(opt[~ok], used[~ok], s=14, facecolors="none", edgecolors=COL[m], alpha=0.8, label=f"{m} fail")
    if not count:
        return nodata(ax, title, "No A* baseline data")
    lim = max(1, max_distance * 1.05)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Equal distance")
    ax.set(xlim=(0, lim), ylim=(0, lim))
    ax.set_xlabel("Ground-truth A* path (m)", fontsize=8)
    ax.set_ylabel("SLAM distance traveled (m)", fontsize=8)
    total = sum(len(rs) for rs in R.values())
    ax.set_title(title if count == total else f"{title}\n{count}/{total} episodes plotted", fontsize=9)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)


def timing_split(ax, R):
    ms = list(R)
    keys = [("compute", "timing.per_step_compute_ms.mean", "#1f77b4"),
            ("simulator", "timing.per_step_sim_ms.mean", "#7f7f7f"),
            ("plotting", "timing.per_step_plot_ms.mean", "#e377c2")]
    bottom = np.zeros(len(ms))
    for name, key, colr in keys:
        v = np.array([mean_of(R[m], key) or 0.0 for m in ms])
        ax.bar(range(len(ms)), v, bottom=bottom, color=colr, label=name)
        bottom += v
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("ms per step", fontsize=8)
    ax.set_title("Where step time goes", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def stage_bars(ax, R):
    stages = sorted({s for rs in R.values() for r in rs for s in (get(r, "timing.per_call_ms") or {})})
    if not stages:
        return nodata(ax, "Per-call time by stage")
    ms = list(R)
    h = 0.8 / len(ms)
    for i, m in enumerate(ms):
        v = [mean_of(R[m], f"timing.per_call_ms.{s}.mean") or 0.0 for s in stages]
        ax.barh(np.arange(len(stages)) + i * h, v, h, color=COL[m], label=m)
    ax.set_yticks(np.arange(len(stages)) + h * (len(ms) - 1) / 2)
    ax.set_yticklabels(stages, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("mean ms per call (log)", fontsize=8)
    ax.set_title("Per-call time by stage\n(mapping includes disparity+sift; scan_total includes capture+mapping)",
                 fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.3)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def grid_page(pdf, title, nrows, ncols, panels):
    fig, axes = plt.subplots(nrows, ncols, figsize=PAGE)
    axes = np.atleast_1d(axes).ravel()
    for ax, fn in zip(axes, panels):
        fn(ax)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def summary_page(pdf, R, dirs):
    ms = list(R)
    rows = [
        ("Episodes", lambda rs: str(len(rs))),
        ("Success rate (self-tracked)", lambda rs: fmt(np.mean([bool(r["navigation"]["success"]) for r in rs]))),
        ("Success rate (ground truth)", lambda rs: fmt(mean_of(rs, "navigation.success_gt"))),
        ("Avg successful path (m)", lambda rs: fmt(np.mean([r["navigation"]["path_length_m"] for r in rs
                                                          if r["navigation"]["success"]]) if any(
            r["navigation"]["success"] for r in rs) else None)),
        ("Mean SPL", lambda rs: fmt(mean_of(rs, "navigation.spl"))),
        ("Collisions / episode", lambda rs: fmt(mean_of(rs, "navigation.collisions"))),
        ("Compute (ms / step)", lambda rs: fmt(mean_of(rs, "timing.per_step_compute_ms.mean"), ".1f")),
        ("Wall time / episode (s)", lambda rs: fmt(mean_of(rs, "timing.total_wall_s"), ".1f")),
        ("CPU python (% of 1 core)", lambda rs: fmt(mean_of(rs, "resources.cpu_py_pct.mean"), ".0f")),
        ("Python RSS peak (MB)", lambda rs: fmt(mean_of(rs, "resources.rss_py_mb.max"), ".0f")),
        ("GPU util (%)", lambda rs: fmt(mean_of(rs, "resources.gpu_util_pct.mean"), ".0f")),
        ("GPU power (W)", lambda rs: fmt(mean_of(rs, "resources.gpu_power_w.mean"), ".0f")),
        ("Energy per metre (J/m)", lambda rs: fmt(mean_of(rs, "resources.energy_per_m_j"), ".0f")),
        ("Depth AbsRel", lambda rs: fmt(mean_of(rs, "vision.depth_absrel.mean"), ".3f")),
        ("Map free-cell precision", lambda rs: fmt(mean_of(rs, "map.final.free_precision"), ".3f")),
        ("Phantom obstacle cells", lambda rs: fmt(mean_of(rs, "map.final.phantom_obstacle_cells"), ".1f")),
    ]
    fig, ax = plt.subplots(figsize=PAGE)
    ax.axis("off")
    hw = get(next(iter(R.values()))[0], "meta.hardware") or {}
    fig.suptitle("Navigation metrics report", fontsize=16, fontweight="bold", y=0.97)
    ax.text(0.5, 1.0, f"generated {datetime.now():%Y-%m-%d %H:%M}   |   {', '.join(dirs)}"
            + (f"   |   GPU: {hw['gpu']}" if hw.get("gpu") else ""),
            ha="center", va="top", fontsize=8, color="gray", transform=ax.transAxes)
    cells = [[fn(R[m]) for m in ms] for _, fn in rows]
    tbl = ax.table(cellText=cells, rowLabels=[r[0] for r in rows], colLabels=ms, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dddddd")
            cell.set_text_props(fontweight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def scene_page(pdf, R):
    scenes = sorted({r["meta"].get("scene") for rs in R.values() for r in rs})
    if len(scenes) < 2:
        return
    fig, ax = plt.subplots(figsize=PAGE)
    ms = list(R)
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        rate = []
        for s in scenes:
            sub = [r for r in R[m] if r["meta"].get("scene") == s]
            rate.append(np.mean([bool(r["navigation"]["success"]) for r in sub]) if sub else np.nan)
        ax.bar(np.arange(len(scenes)) + i * w, rate, w, color=COL[m], label=m)
    ax.set_xticks(np.arange(len(scenes)) + w * (len(ms) - 1) / 2)
    ax.set_xticklabels(scenes, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Success rate per scene", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def timeseries_page(pdf, R):
    """Uses one *_samples.csv per method (only exists if the run was NOT started with --no-csv)."""
    found = {}
    for m, rs in R.items():
        for r in rs:
            p = r["_path"].replace(".json", "_samples.csv")
            if os.path.exists(p):
                found[m] = p
                break
    if not found:
        return

    def read(p):
        with open(p) as f:
            rows = [r for r in csv.DictReader(f) if r.get("phase", "run") == "run"]
        col = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])
        return col

    fig, axes = plt.subplots(3, 1, figsize=PAGE, sharex=True)
    for m, p in found.items():
        col = read(p)
        t = col("t")
        axes[0].plot(t, col("cpu_py_pct"), color=COL[m], label=f"{m} python")
        axes[0].plot(t, col("cpu_sim_pct"), color=COL[m], ls=":", label=f"{m} simulator")
        axes[1].plot(t, col("rss_py_mb"), color=COL[m], label=f"{m} python RSS")
        axes[1].plot(t, col("gpu_mem_proc_mb"), color=COL[m], ls="--", label=f"{m} GPU mem (process)")
        axes[2].plot(t, col("gpu_power_w"), color=COL[m], label=f"{m} GPU power (W)")
        axes[2].plot(t, col("cpu_power_w"), color=COL[m], ls=":", label=f"{m} CPU power (W)")
    for ax, yl in zip(axes, ["CPU (% of one core)", "memory (MB)", "power (W)"]):
        ax.set_ylabel(yl, fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time since start (s)")
    fig.suptitle("Resource time series (one episode per method)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="folder(s) containing episode_*.json")
    ap.add_argument("--out", default=None, help="output PDF (default: <first dir>/metrics_report.pdf)")
    args = ap.parse_args()

    R = load(args.dirs)
    if not R:
        raise SystemExit("No episode_*.json found in: " + ", ".join(args.dirs))
    for i, m in enumerate(R):
        COL[m] = plt.cm.tab10(i % 10)
    out = args.out or os.path.join(args.dirs[0], "metrics_report.pdf")
    n_gt = "no data (needs renderDepthImage=True)"

    with PdfPages(out) as pdf:
        summary_page(pdf, R, args.dirs)

        grid_page(pdf, "Navigation", 2, 3, [
            lambda ax: success_bars(ax, R),
            lambda ax: box(ax, R, "navigation.spl", "SPL (ground-truth A*)"),
            lambda ax: box(ax, R, "navigation.collisions", "Collisions per episode"),
            lambda ax: path_scatter(ax, R),
            lambda ax: termination_bars(ax, R),
            lambda ax: box(ax, R, "navigation.path_efficiency", "Path efficiency (A* / traveled)"),
        ])

        grid_page(pdf, "Time to compute", 2, 3, [
            lambda ax: box(ax, R, "timing.per_step_compute_ms.mean", "Compute per step (mean)", "ms"),
            lambda ax: box(ax, R, "timing.per_step_compute_ms.p95", "Compute per step (p95)", "ms"),
            lambda ax: timing_split(ax, R),
            lambda ax: stage_bars(ax, R),
            lambda ax: box(ax, R, "timing.total_wall_s", "Wall time per episode", "s"),
            lambda ax: box(ax, R, "navigation.nav_steps", "Navigation steps per episode"),
        ])

        grid_page(pdf, "CPU, GPU, memory and power", 2, 4, [
            lambda ax: box(ax, R, "resources.cpu_py_pct.mean", "CPU: python (mean)", "% of one core"),
            lambda ax: box(ax, R, "resources.cpu_sim_pct.mean", "CPU: simulator (mean)", "% of one core"),
            lambda ax: box(ax, R, "resources.rss_py_mb.max", "RAM: python peak", "MB"),
            lambda ax: box(ax, R, "resources.rss_sim_mb.max", "RAM: simulator peak", "MB"),
            lambda ax: box(ax, R, "resources.gpu_util_pct.mean", "GPU utilisation (mean)", "%"),
            lambda ax: box(ax, R, "resources.gpu_mem_proc_mb.max", "GPU memory: this process, peak", "MB"),
            lambda ax: box(ax, R, "resources.gpu_power_w.mean", "GPU power (mean)", "W"),
            lambda ax: box(ax, R, "resources.energy_per_m_j", "Energy per metre travelled", "J/m"),
        ])

        grid_page(pdf, "Vision metrics", 2, 3, [
            lambda ax: box(ax, R, "vision.disp_density.mean", "Valid disparity density", "fraction of pixels"),
            lambda ax: box(ax, R, "vision.sift_matches.mean", "SIFT epipolar matches / frame"),
            lambda ax: box(ax, R, "vision.depth_coverage.mean", "Depth coverage vs ground truth", msg=n_gt),
            lambda ax: box(ax, R, "vision.depth_absrel.mean", "Depth AbsRel (lower is better)", msg=n_gt),
            lambda ax: box(ax, R, "vision.depth_delta1.mean", "Depth accuracy d<1.25 (higher is better)", msg=n_gt),
            lambda ax: box(ax, R, "vision.disp_bad1.mean", "Disparity bad-1px rate", msg=n_gt),
        ])

        grid_page(pdf, "Map quality vs ground truth", 2, 3, [
            lambda ax: box(ax, R, "map.final.free_precision", "Free-cell precision"),
            lambda ax: box(ax, R, "map.final.reachable_coverage", "Reachable space mapped"),
            lambda ax: box(ax, R, "map.final.phantom_obstacle_cells", "Phantom obstacle cells"),
            lambda ax: box(ax, R, "map.final.false_free_cells", "False-free cells (dangerous)"),
            lambda ax: box(ax, R, "map.final.phantom_obstacle_rate", "Phantom obstacle rate"),
            lambda ax: box(ax, R, "navigation.pose_error_m", "Final pose error (dead reckoning)", "m"),
        ])

        scene_page(pdf, R)
        timeseries_page(pdf, R)

    print(f"Wrote {out}  ({sum(len(v) for v in R.values())} episodes, methods: {', '.join(R)})")


if __name__ == "__main__":
    main()
