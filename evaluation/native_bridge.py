"""Frozen, fitting-only conversion of recorded (idx, ds, n) into the shared map frame."""
import numpy as np

MAX_ERROR_M = 1e-6
MAX_SCALE_DEVIATION = 1e-5
ROUTES = (("track", ""), ("pit", "_pit"))


def indices(values, size=None):
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0):
        raise ValueError("Native segment indices must be finite nonnegative integers")
    if size is not None and np.any(values >= size):
        raise ValueError("Native segment is absent from the frozen fitting-only bridge")
    return values.astype(np.int64)


def fit(state, fit_mask):
    """Fit a near-unit complex similarity per segment; no non-fitting position is read."""
    mask = np.asarray(fit_mask, bool)
    if mask.shape != state["timestamp_ns"].shape or not mask.any():
        raise ValueError("Invalid or empty fitting mask")
    world = state["x"][mask] + 1j*state["y"][mask]
    if not np.isfinite(world).all():
        raise ValueError("Non-finite fitting positions")
    frames = {}
    for route, suffix in ROUTES:
        idx = indices(state[route+"_idx"][mask])
        unique = np.unique(idx)
        if not np.array_equal(unique, np.arange(len(unique))) or len(unique) < 3:
            raise ValueError(f"{route}: fitting data must cover a contiguous closed route from index zero")
        z = state[route+"_ds"][mask] + 1j*state["n"+suffix][mask]
        if not np.isfinite(z).all():
            raise ValueError(f"{route}: non-finite fitting coordinates")
        origin, rotation = np.empty(len(unique), complex), np.empty(len(unique), complex)
        counts = np.empty(len(unique), np.int64)
        for k in unique:
            selected = idx == k
            counts[k] = selected.sum()
            if counts[k] < 3:
                raise ValueError(f"{route}: segment {k} has fewer than three fitting observations")
            a, b = z[selected], world[selected]
            da, db = a-a.mean(), b-b.mean()
            span = np.sum(np.abs(da)**2)
            if span <= 1e-8:
                raise ValueError(f"{route}: segment {k} has insufficient coordinate variation")
            # Preserve the tiny scale term: normalizing it loses the recording's numerical convention.
            rotation[k] = np.sum(db*np.conj(da))/span
            origin[k] = np.mean(b-rotation[k]*a)
            if abs(abs(rotation[k])-1) > MAX_SCALE_DEVIATION:
                raise ValueError(f"{route}: segment {k} is inconsistent with metre-valued orthogonal coordinates")
            if np.max(np.abs(origin[k]+rotation[k]*a-b)) > MAX_ERROR_M:
                raise ValueError(f"{route}: segment {k} fails fitting reconstruction")
        local_delta = (np.roll(origin, -1)-origin)/rotation
        if np.any(local_delta.real < 1e-3) or np.max(np.abs(local_delta.imag)) > 1e-5:
            raise ValueError(f"{route}: inferred frames do not form a continuous directed closed polyline")
        frames[route+"_origin"] = origin
        frames[route+"_rotation"] = rotation
        frames[route+"_fit_count"] = counts
    return frames


def reconstruct(state, frames):
    """Use only raw native coordinates and frozen frames; input x/y is intentionally unused."""
    positions = {}
    for route, suffix in ROUTES:
        origin, rotation = frames[route+"_origin"], frames[route+"_rotation"]
        if origin.ndim != 1 or origin.shape != rotation.shape or len(origin) < 3:
            raise ValueError(f"{route}: invalid bridge shape")
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all() or np.any(abs(abs(rotation)-1) > MAX_SCALE_DEVIATION):
            raise ValueError(f"{route}: non-finite or invalid bridge frames")
        idx = indices(state[route+"_idx"], len(origin))
        z = state[route+"_ds"] + 1j*state["n"+suffix]
        if not np.isfinite(z).all():
            raise ValueError(f"{route}: non-finite native coordinates")
        positions[route] = origin[idx] + rotation[idx]*z
    disagreement = np.abs(positions["track"]-positions["pit"])
    if np.max(disagreement) > MAX_ERROR_M:
        raise ValueError("Track and pit native coordinates disagree in the shared map frame")
    xy = positions["track"]
    output = {"timestamp_ns": state["timestamp_ns"], "x": xy.real, "y": xy.imag, "yaw": state["yaw"]}
    if "lane" in state:
        output["lane"] = state["lane"]
    return output, disagreement
