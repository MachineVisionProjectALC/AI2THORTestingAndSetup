"""Compare the stereo pipeline with simulator depth without changing navigation.

Run from the project root: python stereotesting/diagnose_stereo.py
Close each figure to inspect the next direction.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Include the project root when this file is executed directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from old.box_room import ClassicalStereoMapper, init_environment, sync_stereo_pair


def diagnose(mapper, left, right, depth, direction):
    left = mapper.to_work(left)
    right = mapper.to_work(right)
    depth = cv2.resize(depth, (mapper.W, mapper.H), interpolation=cv2.INTER_NEAREST)
    gray_l = mapper.clahe.apply(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY))
    gray_r = mapper.clahe.apply(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY))
    raw = mapper.sgbm_left.compute(gray_l, gray_r).astype(np.float32) / 16
    clean = mapper.compute_clean_disparity(gray_l, gray_r)

    # AI2-THOR depth is used only to evaluate stereo, never as input to it.
    truth_valid = np.isfinite(depth) & (depth >= 0.35) & (depth <= 2.5)
    expected = np.zeros_like(depth, dtype=np.float32)
    expected[truth_valid] = mapper.fx * mapper.baseline / depth[truth_valid]
    valid = truth_valid & (clean > 0)
    error = np.abs(clean[valid] - expected[valid])

    kp_l, desc_l = mapper.sift.detectAndCompute(gray_l, None)
    kp_r, desc_r = mapper.sift.detectAndCompute(gray_r, None)
    offsets = []
    if desc_l is not None and desc_r is not None and len(desc_r) >= 2:
        for pair in mapper.matcher.knnMatch(desc_l, desc_r, k=2):
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                a, b = kp_l[pair[0].queryIdx].pt, kp_r[pair[0].trainIdx].pt
                offsets.append((a[0] - b[0], a[1] - b[1]))
    print(f'\nDirection {direction}: left {left.shape}, right {right.shape}', flush=True)
    print(f'  valid disparity: {100 * (clean > 0).mean():.1f}% of image', flush=True)
    if offsets:
        offsets = np.asarray(offsets)
        print(f'  SIFT matches: {len(offsets)}; median horizontal shift: {np.median(offsets[:, 0]):.2f}px; '
              f'median absolute vertical shift: {np.median(np.abs(offsets[:, 1])):.2f}px', flush=True)
    else:
        print('  SIFT matches: none; alignment could not be assessed', flush=True)
    if error.size:
        print(f'  ground-truth comparison: {error.size} pixels, median error {np.median(error):.2f}px, '
              f'within 1px: {100 * np.mean(error <= 1):.1f}%', flush=True)
    else:
        print('  no valid pixels in the 0.35–2.5m comparison range', flush=True)

    cmap = plt.get_cmap('plasma').copy()
    cmap.set_bad('gray')
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(left)
    axes[1].imshow(right)
    for ax, data, title in ((axes[2], np.ma.masked_less_equal(raw, 1), 'Raw SGBM'),
                            (axes[3], np.ma.masked_less_equal(clean, 0), 'Filtered SGBM'),
                            (axes[4], np.ma.masked_where(~truth_valid, expected), 'Ground truth disparity\n(0.35–2.5 m)')):
        im = ax.imshow(data, cmap=cmap, vmin=1.5, vmax=48)
        ax.set_title(title)
    axes[0].set_title('Left camera')
    axes[1].set_title('Right camera')
    for ax in axes:
        ax.axis('off')
    fig.colorbar(im, ax=axes, shrink=0.7, label='Disparity (pixels)')
    fig.suptitle(f'Direction {direction}: gray = unavailable')
    plt.show()


if __name__ == '__main__':
    controller, *_ = init_environment(render_depth=True)
    try:
        for direction in range(4):
            left, right, _ = sync_stereo_pair(controller)
            depth = controller.last_event.depth_frame
            if depth is None:
                raise RuntimeError('The simulator did not return a depth frame.')
            diagnose(ClassicalStereoMapper.from_frame(left), left, right, depth, direction)
            controller.step(action='RotateRight')
    finally:
        controller.stop()
