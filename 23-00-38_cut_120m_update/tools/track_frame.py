"""Track (s, n) frame of the racing line / pit-lane line stored in map.mcap.

Both reference lines are closed polylines (nav_msgs/Path, frame map):
  /constructor0/map_ar/racing_line  -- 'track': the line the car's state estimation computes (s, n) against
  /constructor0/map_ar/pitlane_line -- 'pit':   same loop, but going through the pit lane (pose idx ~931..2221)

The last pose of each Path repeats the first one (closed loop), so the closing segment has zero length; it is
skipped and pose idx N-1 is the same point as idx 0 (s = 0).

Definitions (identical to the car's state estimation, which projects on the polyline segment idx -> idx+1):
  s    arc length along the polyline from its first pose, metres, in [0, lap_length)
  n    signed distance from the segment, metres, positive to the LEFT of the driving direction
  epsi heading of the car minus heading of the segment, rad, wrapped to +-pi
"""
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

TOPICS = {'track': '/constructor0/map_ar/racing_line', 'pit': '/constructor0/map_ar/pitlane_line'}
WINDOW = 200         # segments (~200 m) searched around the previous projection (sequences only)
MIN_SEG = 1e-3       # shorter segments (the duplicated closing pose) are skipped
MAX_DT_NS = 1_000_000_000   # larger gap between consecutive poses -> global search again


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a)))


class Line:
    def __init__(self, xy):
        self.p = np.asarray(xy, float)                       # (N, 2) poses
        d = np.roll(self.p, -1, axis=0) - self.p              # segment k: p[k] -> p[k+1] (last closes the loop)
        self.len = np.hypot(d[:, 0], d[:, 1])
        self.valid = self.len >= MIN_SEG
        self.len[~self.valid] = 0.0
        u = np.zeros_like(d)
        u[self.valid] = d[self.valid] / self.len[self.valid, None]
        for k in np.where(~self.valid)[0]:                     # zero-length segment: direction of the previous one
            u[k] = u[(k - 1) % len(u)]
        self.u = u                                             # unit tangent
        self.heading = np.arctan2(self.u[:, 1], self.u[:, 0])
        self.s_ref = np.r_[0.0, np.cumsum(self.len[:-1])]
        self.lap = float(self.len.sum())
        self.s_ref[self.s_ref >= self.lap - 1e-9] = 0.0          # trailing duplicate of pose 0

    def _project_one(self, q, cand):
        rel = q - self.p[cand]
        t = np.clip(np.einsum('ij,ij->i', rel, self.u[cand]), 0.0, self.len[cand])
        foot = self.p[cand] + t[:, None] * self.u[cand]
        j = np.argmin(np.hypot(*(q - foot).T))
        k = cand[j]
        n = self.u[k, 0] * rel[j, 1] - self.u[k, 1] * rel[j, 0]
        return k, t[j], n

    def xy_to_sn(self, x, y, yaw=None, t_ns=None):
        """Project poses on the line. With t_ns (sorted), each pose is searched near the previous one
        (like the car does), otherwise globally. Returns s, n, epsi (nan without yaw), segment idx."""
        x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
        N, all_k = len(self.p), np.where(self.valid)[0]
        s, n, k_out = np.empty(len(x)), np.empty(len(x)), np.empty(len(x), int)
        k_prev = None
        for i, q in enumerate(np.c_[x, y]):
            if t_ns is None or k_prev is None or t_ns[i] - t_ns[i - 1] > MAX_DT_NS:
                cand = all_k
            else:
                cand = (k_prev + np.arange(-WINDOW, WINDOW + 1)) % N
                cand = cand[self.valid[cand]]
            k, t, n[i] = self._project_one(q, cand)
            s[i], k_out[i], k_prev = self._norm_s(self.s_ref[k] + t), k, k
        epsi = wrap(np.asarray(yaw) - self.heading[k_out]) if yaw is not None else np.full(len(x), np.nan)
        return s, n, epsi, k_out

    def sn_to_xy(self, s, n, epsi=None):
        """Inverse of xy_to_sn. Returns x, y, yaw (nan without epsi)."""
        s = self._norm_s(np.atleast_1d(s).astype(float))
        n = np.atleast_1d(n).astype(float)
        kv = np.where(self.valid)[0]                             # s_ref is increasing over valid segments
        k = kv[np.clip(np.searchsorted(self.s_ref[kv], s, side='right') - 1, 0, len(kv) - 1)]
        t = s - self.s_ref[k]
        xy = self.p[k] + t[:, None] * self.u[k] + n[:, None] * np.c_[-self.u[k, 1], self.u[k, 0]]
        yaw = wrap(self.heading[k] + np.asarray(epsi)) if epsi is not None else np.full(len(s), np.nan)
        return xy[:, 0], xy[:, 1], yaw

    def _norm_s(self, s):
        s = np.mod(s, self.lap)
        return np.where(s >= self.lap - 1e-9, 0.0, s)

    def from_car(self, idx, ds):
        """s from the car's own projection (sn_state.idx, sn_state.ds)."""
        return self._norm_s(self.s_ref[np.asarray(idx)] + np.asarray(ds))


def load_lines(map_mcap):
    out = {}
    with open(map_mcap, 'rb') as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, ch, _, msg in r.iter_decoded_messages(topics=list(TOPICS.values())):
            name = [k for k, v in TOPICS.items() if v == ch.topic][0]
            out[name] = Line([(q.pose.position.x, q.pose.position.y) for q in msg.poses])
    missing = set(TOPICS) - set(out)
    if missing:
        raise SystemExit(f'{map_mcap}: missing {[TOPICS[m] for m in missing]}')
    return out
