"""Experimental MI-cost SGM inspired by Hirschmueller (2008).

Uses OpenCV SGBM only to bootstrap MI correspondences, a smoothed joint
intensity histogram, eight path directions and parabolic subpixel fitting.
It is not a full reproduction of the paper's hierarchical implementation.
"""
import cv2
import numpy as np


def _mi_lookup(left, right, seed, bins=64):
    h, w = left.shape
    x = np.broadcast_to(np.arange(w), (h, w))
    xr = np.rint(x - seed).astype(np.int32)
    valid = (seed > 0) & (xr >= 0) & (xr < w)
    if np.count_nonzero(valid) < 100:
        raise ValueError('Too few bootstrap matches to estimate mutual information')
    li = (left[valid].astype(np.int32) * bins) // 256
    ri = (right[np.broadcast_to(np.arange(h)[:, None], (h, w))[valid], xr[valid]].astype(np.int32) * bins) // 256
    hist = np.bincount(li * bins + ri, minlength=bins * bins).reshape(bins, bins).astype(np.float32)
    joint = cv2.GaussianBlur(hist, (5, 5), 1.0) + 1e-6
    joint /= joint.sum()
    pleft, pright = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    # Negative pointwise MI; a positive offset leaves disparity decisions unchanged.
    cost = -np.log(joint / (pleft * pright))
    cost = cv2.GaussianBlur(cost.astype(np.float32), (5, 5), 1.0)
    return cost - cost.min()


def _aggregate(cost, p1, p2):
    h, w, d = cost.shape
    total = np.zeros_like(cost, dtype=np.float32)
    for dy, dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
        path = np.empty_like(cost, dtype=np.float32)
        if dy:
            for y in (range(h) if dy > 0 else range(h-1, -1, -1)):
                py = y - dy
                path[y] = cost[y]
                if not 0 <= py < h:
                    continue
                xs = np.arange(max(0, dx), min(w, w + dx))
                prev = path[py, xs - dx]
                mn = prev.min(axis=1, keepdims=True)
                lower = np.concatenate((np.full((len(xs),1), np.inf), prev[:,:-1] + p1), axis=1)
                upper = np.concatenate((prev[:,1:] + p1, np.full((len(xs),1), np.inf)), axis=1)
                path[y, xs] += np.minimum(np.minimum(prev, lower), np.minimum(upper, mn + p2)) - mn
        else:
            for x in (range(w) if dx > 0 else range(w-1, -1, -1)):
                px = x - dx
                path[:,x] = cost[:,x]
                if not 0 <= px < w:
                    continue
                prev = path[:,px]
                mn = prev.min(axis=1, keepdims=True)
                lower = np.concatenate((np.full((h,1), np.inf), prev[:,:-1] + p1), axis=1)
                upper = np.concatenate((prev[:,1:] + p1, np.full((h,1), np.inf)), axis=1)
                path[:,x] += np.minimum(np.minimum(prev, lower), np.minimum(upper, mn + p2)) - mn
        total += path
    return total


def compute_mi_sgm(left, right, num_disparities=32, p1=0.8, p2=4.0, uniqueness=0.03):
    """Return left-view disparity, with invalid pixels as zero."""
    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('Input images must be same-size uint8 grayscale arrays')
    if num_disparities % 16 or num_disparities < 16 or num_disparities >= left.shape[1]:
        raise ValueError('num_disparities must be a multiple of 16 smaller than image width')
    bootstrap = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities,
                  blockSize=5, P1=200, P2=800, uniquenessRatio=10)
    seed = bootstrap.compute(left, right).astype(np.float32) / 16
    lookup = _mi_lookup(left, right, seed)
    lq, rq = left.astype(np.int32) // 4, right.astype(np.int32) // 4
    h,w = left.shape
    cost = np.full((h,w,num_disparities), 1000, dtype=np.float32)
    for d in range(num_disparities):
        cost[:,d:,d] = lookup[lq[:,d:], rq[:,:w-d]]
    agg = _aggregate(cost, p1, p2)
    idx = np.argmin(agg, axis=2)
    best = np.take_along_axis(agg, idx[...,None], axis=2)[...,0]
    # Compare with disparities at least two pixels away, not adjacent subpixel candidates.
    candidates = agg.copy()
    for off in (-1,0,1):
        ind = np.clip(idx + off, 0, num_disparities-1)
        np.put_along_axis(candidates, ind[...,None], np.inf, axis=2)
    second = candidates.min(axis=2)
    valid = (idx > 0) & (idx < num_disparities-1) & (second - best > uniqueness * np.maximum(np.abs(best), 1))
    valid &= np.arange(w)[None,:] >= num_disparities
    lo = np.take_along_axis(agg, np.maximum(idx-1,0)[...,None], axis=2)[...,0]
    hi = np.take_along_axis(agg, np.minimum(idx+1,num_disparities-1)[...,None], axis=2)[...,0]
    denom = lo - 2*best + hi
    sub = np.divide(0.5*(lo-hi), denom, out=np.zeros_like(best), where=denom > 1e-6)
    disparity = idx.astype(np.float32) + np.clip(sub, -1, 1)
    disparity[~valid] = 0
    return disparity
