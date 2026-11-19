"""Compare experimental MI-SGM with the project's OpenCV pipeline.

Run from the project root: python stereotesting/experimental.py
Four headings at the starting position; each pair sees identical RGB and depth.
Ground truth is used only for evaluation. Use the buttons or arrow keys to change views.
"""
import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import roomtesting as room
from stereotesting.mi_sgm_algo import compute_mi_sgm

WORK_HEIGHT = 150  # Keeps the NumPy prototype manageable; both methods use this size.
NUM_DISPARITIES = 48
RESULTS = Path(__file__).resolve().parent / 'mi_sgm_comparison.csv'


def metrics(disparity, depth, fx, baseline, elapsed, method, direction):
    eligible = np.isfinite(depth) & (depth >= 0.35) & (depth <= room.MAX_RANGE)
    valid = eligible & (disparity > 0)
    errors = np.abs(fx * baseline / disparity[valid] - depth[valid])
    n = int(eligible.sum())
    return dict(direction=direction, method=method, eligible_pixels=n,
                valid_pixels=int(valid.sum()), coverage_pct=100*valid.sum()/max(n,1),
                median_depth_error_m=float(np.median(errors)) if errors.size else None,
                accurate_25cm_pct=100*np.sum(errors <= 0.25)/max(n,1), seconds=elapsed)


def main():
    controller, *_ = room.init_environment(render_depth=True)
    rows = []
    views = []
    try:
        left, _ = room.sync_stereo_rig(controller)
        mapper = room.ClassicalStereoMapper.from_frame(left, baseline=room.BASELINE, work_h=WORK_HEIGHT)
        print(f'Comparing at {mapper.W}x{mapper.H}, baseline {room.BASELINE:.3f} m', flush=True)
        for direction in range(4):
            left, right = room.sync_stereo_rig(controller)
            depth = controller.last_event.depth_frame
            if depth is None:
                raise RuntimeError('Simulator returned no ground-truth depth frame')
            depth = cv2.resize(depth, (mapper.W,mapper.H), interpolation=cv2.INTER_NEAREST)
            # Same preprocessing for both algorithms, as used by roomtesting.py.
            gl = mapper.clahe.apply(cv2.cvtColor(mapper.to_work(left), cv2.COLOR_RGB2GRAY))
            gr = mapper.clahe.apply(cv2.cvtColor(mapper.to_work(right), cv2.COLOR_RGB2GRAY))
            start = time.perf_counter()
            standard = mapper.compute_clean_disparity(gl,gr)
            t_standard = time.perf_counter()-start
            start = time.perf_counter()
            experimental = compute_mi_sgm(gl,gr,NUM_DISPARITIES)
            t_mi = time.perf_counter()-start
            for method, disp, elapsed in [('OpenCV filtered',standard,t_standard),('MI-SGM prototype',experimental,t_mi)]:
                row = metrics(disp,depth,mapper.fx,mapper.baseline,elapsed,method,direction)
                rows.append(row)
                print(f"Direction {direction} | {method}: coverage={row['coverage_pct']:.1f}% "
                      f"good/eligible={row['accurate_25cm_pct']:.1f}% median error={row['median_depth_error_m']} m "
                      f"time={elapsed:.2f}s",flush=True)
            with open(RESULTS,'w',newline='') as output:
                writer=csv.DictWriter(output,fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            valid_truth = np.isfinite(depth) & (depth >= 0.35) & (depth <= room.MAX_RANGE)
            truth = np.zeros_like(depth)
            truth[valid_truth] = mapper.fx * mapper.baseline / depth[valid_truth]
            views.append((mapper.to_work(left).copy(), standard.copy(), experimental.copy(), truth))
            if direction < 3:
                controller.step(action='RotateRight')
        print(f'Saved {RESULTS}; metrics are per view, not independent trials.')
    finally:
        controller.stop()

    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('gray')
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.subplots_adjust(bottom=0.20)
    images = [axes[0].imshow(views[0][0])]
    axes[0].set_title('Left camera')
    for ax, data, title in zip(axes[1:], views[0][1:],
                               ['OpenCV filtered', 'MI-SGM prototype', 'Ground-truth disparity']):
        images.append(ax.imshow(np.ma.masked_less_equal(data, 0), cmap=cmap,
                                vmin=0, vmax=max(NUM_DISPARITIES, mapper.num_disp)))
        ax.set_title(title)
    for ax in axes:
        ax.axis('off')
    fig.colorbar(images[-1], ax=axes, shrink=0.7, label='disparity (pixels)')
    title = fig.suptitle('Direction 1/4: gray = unavailable')
    current = 0

    def change_direction(step):
        nonlocal current
        current = (current + step) % len(views)
        for index, (image, data) in enumerate(zip(images, views[current])):
            image.set_data(data if index == 0 else np.ma.masked_less_equal(data, 0))
        title.set_text(f'Direction {current + 1}/4: gray = unavailable')
        fig.canvas.draw_idle()

    previous = Button(fig.add_axes([0.36, 0.05, 0.12, 0.08]), 'Previous')
    following = Button(fig.add_axes([0.51, 0.05, 0.12, 0.08]), 'Next')
    previous.on_clicked(lambda event: change_direction(-1))
    following.on_clicked(lambda event: change_direction(1))
    fig.canvas.mpl_connect('key_press_event', lambda event: change_direction(-1 if event.key == 'left' else 1)
                           if event.key in ('left', 'right') else None)
    plt.show()


if __name__ == '__main__':
    main()
