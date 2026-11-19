"""Show the current stereo depth estimate alongside simulator depth.

Run from the project root: python stereotesting/current.py
Use the buttons or left and right arrow keys to change direction.
"""
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import roomtesting as room


def depth_metrics(truth, estimated):
    """Compare usable estimates with eligible ground-truth pixels."""
    eligible = np.isfinite(truth)
    valid = eligible & np.isfinite(estimated)
    total = int(eligible.sum())
    count = int(valid.sum())
    if not total:
        return "No ground-truth pixels in range"
    coverage = 100 * count / total
    if not count:
        return "Coverage 0% · no valid depth estimates"
    errors = np.abs(estimated[valid] - truth[valid])
    within_25cm = 100 * np.count_nonzero(errors <= 0.25) / total
    return (f"Coverage {coverage:.1f}% · Median error {np.median(errors):.2f} m · "
            f"Within 25 cm {within_25cm:.1f}% of eligible pixels")


def main():
    controller, _, _, _, _, pitch = room.init_environment(render_depth=True)
    views = []
    try:
        for direction in range(4):
            left, right = room.sync_stereo_rig(controller)
            depth = controller.last_event.depth_frame
            if depth is None:
                raise RuntimeError("Simulator depth is unavailable")

            mapper = room.ClassicalStereoMapper.from_frame(left, pitch_deg=pitch)
            left_gray = mapper.clahe.apply(cv2.cvtColor(mapper.to_work(left), cv2.COLOR_RGB2GRAY))
            right_gray = mapper.clahe.apply(cv2.cvtColor(mapper.to_work(right), cv2.COLOR_RGB2GRAY))
            disparity = mapper.compute_clean_disparity(left_gray, right_gray)
            raw = mapper.sgbm_left.compute(left_gray, right_gray).astype(np.float32) / 16.0
            kp_l, desc_l = mapper.sift.detectAndCompute(left_gray, None)
            kp_r, desc_r = mapper.sift.detectAndCompute(right_gray, None)
            offsets = []
            if desc_l is not None and desc_r is not None and len(desc_r) >= 2:
                for pair in mapper.matcher.knnMatch(desc_l, desc_r, k=2):
                    if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                        a = kp_l[pair[0].queryIdx].pt
                        b = kp_r[pair[0].trainIdx].pt
                        offsets.append((a[0] - b[0], a[1] - b[1]))
            if offsets:
                offsets = np.asarray(offsets)
                print(f"Direction {direction}: {len(offsets)} SIFT matches; median horizontal shift "
                      f"{np.median(offsets[:, 0]):.2f}px, median absolute vertical shift "
                      f"{np.median(np.abs(offsets[:, 1])):.2f}px")
            else:
                print(f"Direction {direction}: no SIFT matches")
            truth = cv2.resize(depth, (mapper.W, mapper.H), interpolation=cv2.INTER_NEAREST)
            estimated = np.full(disparity.shape, np.nan, dtype=np.float32)
            valid = disparity > 0
            estimated[valid] = mapper.fx * mapper.baseline / disparity[valid]
            estimated[(estimated < 0.30) | (estimated > room.MAX_RANGE)] = np.nan
            truth = np.where(np.isfinite(truth) & (truth >= 0.30) & (truth <= room.MAX_RANGE), truth, np.nan)

            views.append((mapper.to_work(left).copy(), truth, estimated, raw, disparity))
            if direction < 3:
                controller.step(action="RotateRight")
    finally:
        controller.stop()

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("gray")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.subplots_adjust(bottom=0.22)
    images = [axes[0].imshow(views[0][0])]
    axes[0].set_title("Left camera")
    for ax, data, title in zip(axes[1:3], views[0][1:3], ("Simulator depth", "SGBM depth")):
        images.append(ax.imshow(data, cmap=cmap, vmin=0.30, vmax=room.MAX_RANGE))
        ax.set_title(title)
    for ax, data, title in zip(axes[3:], views[0][3:], ("Raw disparity", "Filtered disparity")):
        images.append(ax.imshow(np.ma.masked_less_equal(data, 0), cmap="plasma", vmin=0, vmax=mapper.num_disp))
        ax.set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(images[2], ax=axes[1:3], label="Depth (m)", shrink=0.8)
    fig.colorbar(images[4], ax=axes[3:], label="Disparity (pixels)", shrink=0.8)
    title = fig.suptitle("Direction 1/4 · gray = unavailable or outside range")
    stats = fig.text(0.5, 0.16, depth_metrics(*views[0][1:3]), ha="center", fontsize=9)
    current = 0

    def change_direction(step):
        nonlocal current
        current = (current + step) % len(views)
        for index, (image, data) in enumerate(zip(images, views[current])):
            image.set_data(np.ma.masked_less_equal(data, 0) if index >= 3 else data)
        title.set_text(f"Direction {current + 1}/4 · gray = unavailable or outside range")
        stats.set_text(depth_metrics(*views[current][1:3]))
        fig.canvas.draw_idle()

    previous = Button(fig.add_axes([0.36, 0.05, 0.12, 0.08]), "Previous")
    following = Button(fig.add_axes([0.51, 0.05, 0.12, 0.08]), "Next")
    previous.on_clicked(lambda event: change_direction(-1))
    following.on_clicked(lambda event: change_direction(1))

    def on_key(event):
        if event.key == "left":
            change_direction(-1)
        elif event.key == "right":
            change_direction(1)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
