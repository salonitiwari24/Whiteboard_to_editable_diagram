"""Arrow direction (source -> target) from arrowhead ink.

For every matched edge we look at both connector ends. Near each end we sample
short lines perpendicular to the connector shaft and count ink that lies to the
*side* of the shaft (the shaft itself is excluded). An arrowhead - open "V" or
filled triangle - puts ink beside the shaft; a plain tail end does not. The edge
is oriented towards the end with clearly more side-ink.

Shape interiors and outlines are masked out so box borders and node text do not
count. When neither end clearly wins, the edge keeps its previous orientation
and is reported as low-confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from src.relationships import EdgeMatch

# ---- Tunable parameters (in connector-shaft widths unless noted) ------------
BACK_STROKES = 8.0      # how far back along the shaft to look from the endpoint
AHEAD_STROKES = 4.0     # how far past the endpoint (towards the shape) to look
BAND_STROKES = 4.0      # half-width of the side band beside the shaft
SHAFT_STROKES = 0.5     # half-width treated as shaft and ignored (+1 px margin)
MASK_DILATE_STROKES = 0.5
MIN_RATIO = 1.5         # head side needs >= 1.5x the side-ink of the tail side
MIN_HEAD_INK_STROKE2 = 1.5  # head side needs >= this * stroke^2 side pixels
SEGMENT_SNAP_STROKES = 3.0  # endpoint-to-Hough-segment snap distance
MIN_SEGMENT_COS = 0.5       # segment must be within 60 deg of the end-to-end direction
MAX_BACK_FRACTION = 0.5     # never look back more than half the connector length
SHAFT_SEARCH_STROKES = 0.25 # tolerance when locating the shaft near an endpoint


@dataclass
class DirectionInfo:
    source: int
    target: int
    side_ink_source: int
    side_ink_target: int
    confident: bool


def _shape_mask(shape_hw: tuple[int, int], contours: Iterable[np.ndarray], dilate_px: int) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for c in contours:
        if c is None or len(c) < 3:
            continue
        cv2.drawContours(mask, [np.asarray(c).astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-6 else v / n


def _local_direction(
    end: tuple[float, float],
    other_end: tuple[float, float],
    segments: list,
    snap_px: float,
) -> np.ndarray | None:
    """Unit vector pointing along the shaft *towards* ``end``.

    Uses the longest Hough segment that has an endpoint near ``end`` (handles
    L-bend connectors); falls back to the straight line from ``other_end``.
    """
    e = np.asarray(end, dtype=np.float64)
    chord = _unit(e - np.asarray(other_end, dtype=np.float64))
    best_len, best_vec = 0.0, None
    for seg in segments:
        p1 = np.asarray(seg.p1, dtype=np.float64)
        p2 = np.asarray(seg.p2, dtype=np.float64)
        for near, far in ((p1, p2), (p2, p1)):
            if np.linalg.norm(near - e) <= snap_px:
                cand = _unit(near - far)
                # Ignore segments pointing well away from the connector
                # (e.g. a shape outline or arc next to the endpoint).
                if cand is None or (chord is not None and float(cand @ chord) < MIN_SEGMENT_COS):
                    continue
                length = float(np.linalg.norm(near - far))
                if length > best_len:
                    best_len, best_vec = length, near - far
    if best_vec is not None:
        u = _unit(best_vec)
        if u is not None:
            return u
    return _unit(e - np.asarray(other_end, dtype=np.float64))


def _side_ink(
    ink: np.ndarray,
    end: tuple[float, float],
    u: np.ndarray,
    stroke: float,
    max_back: float | None = None,
) -> int:
    """Count ink beside the shaft near ``end`` that is connected to the shaft.

    Only ink in the same connected component as the shaft is counted, so
    nearby labels such as "Yes"/"No" do not look like arrowheads.
    """
    h, w = ink.shape[:2]
    n = np.array([-u[1], u[0]])
    e = np.asarray(end, dtype=np.float64)
    back, ahead = BACK_STROKES * stroke, AHEAD_STROKES * stroke
    if max_back is not None:
        back = max(stroke, min(back, max_back))
    band = BAND_STROKES * stroke
    shaft = SHAFT_STROKES * stroke + 1.0

    # Window around the sampling region.
    corners = [e - u * t + n * sg for t in (-ahead, back) for sg in (-band, band)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = max(0, int(np.floor(min(xs))) - 2), min(w, int(np.ceil(max(xs))) + 3)
    y0, y1 = max(0, int(np.floor(min(ys))) - 2), min(h, int(np.ceil(max(ys))) + 3)
    if x0 >= x1 or y0 >= y1:
        return 0
    win = ink[y0:y1, x0:x1].astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(win, connectivity=8)
    if n_lab <= 1:
        return 0

    def _grid(ts: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tt, sg = np.meshgrid(ts, ss, indexing="ij")
        px = np.rint(e[0] - u[0] * tt + n[0] * sg).astype(np.int64) - x0
        py = np.rint(e[1] - u[1] * tt + n[1] * sg).astype(np.int64) - y0
        ok = (px >= 0) & (px < x1 - x0) & (py >= 0) & (py < y1 - y0)
        return px[ok], py[ok]

    ts = np.arange(-ahead, back + 1.0, 1.0)
    # Components touching the shaft centre line.
    cband = max(1.0, round(SHAFT_SEARCH_STROKES * stroke))
    cx, cy = _grid(ts, np.arange(-cband, cband + 1.0, 1.0))
    shaft_labels = set(np.unique(labels[cy, cx]).tolist()) - {0}
    if not shaft_labels:
        return 0
    keep = np.isin(labels, list(shaft_labels))

    ss = np.concatenate([np.arange(-band, -shaft, 1.0), np.arange(shaft + 1.0, band + 1.0, 1.0)])
    sx, sy = _grid(ts, ss)
    pts = np.unique(np.stack([sx, sy], axis=1), axis=0)
    if len(pts) == 0:
        return 0
    return int(np.count_nonzero(keep[pts[:, 1], pts[:, 0]]))


def estimate_shaft_width(binary: np.ndarray, segments: list, default: float) -> float:
    """Median ink thickness measured across the middle of connector segments."""
    ink = binary > 0
    h, w = ink.shape[:2]
    widths: list[int] = []
    for seg in segments:
        p1 = np.asarray(seg.p1, dtype=np.float64)
        p2 = np.asarray(seg.p2, dtype=np.float64)
        u = _unit(p2 - p1)
        if u is None or np.linalg.norm(p2 - p1) < 20:
            continue
        n = np.array([-u[1], u[0]])
        for frac in (0.35, 0.5, 0.65):
            c = p1 + (p2 - p1) * frac
            run = 0
            for sign in (1, -1):
                for k in range(0 if sign == 1 else 1, 25):
                    x, y = int(round(c[0] + n[0] * k * sign)), int(round(c[1] + n[1] * k * sign))
                    if not (0 <= x < w and 0 <= y < h) or not ink[y, x]:
                        break
                    run += 1
            if 1 <= run < 25:
                widths.append(run)
    if len(widths) < 3:
        return float(default)
    return float(np.median(widths))


def orient_matches(
    matches: list[EdgeMatch],
    binary: np.ndarray,
    shape_contours: list[np.ndarray],
    stroke_width: float,
    segments: list | None = None,
) -> tuple[list[tuple[int, int]], list[DirectionInfo]]:
    """Return directed (source, target) pairs plus per-edge diagnostics."""
    segments = list(segments or [])
    stroke = max(2.0, estimate_shaft_width(binary, segments, float(stroke_width)))
    ink = (binary > 0).astype(np.uint8)
    mask = _shape_mask(binary.shape[:2], shape_contours, max(1, int(round(MASK_DILATE_STROKES * stroke))))
    ink[mask > 0] = 0

    directed: list[tuple[int, int]] = []
    infos: list[DirectionInfo] = []
    snap = SEGMENT_SNAP_STROKES * stroke
    min_head = MIN_HEAD_INK_STROKE2 * stroke * stroke
    for m in matches:
        ua = _local_direction(m.endpoint_a, m.endpoint_b, segments, snap)
        ub = _local_direction(m.endpoint_b, m.endpoint_a, segments, snap)
        span = float(np.hypot(m.endpoint_a[0] - m.endpoint_b[0], m.endpoint_a[1] - m.endpoint_b[1]))
        max_back = MAX_BACK_FRACTION * span
        ink_a = _side_ink(ink, m.endpoint_a, ua, stroke, max_back) if ua is not None else 0
        ink_b = _side_ink(ink, m.endpoint_b, ub, stroke, max_back) if ub is not None else 0
        if ink_b >= min_head and ink_b >= MIN_RATIO * ink_a:
            src, dst, s_ink, t_ink, ok = m.node_a, m.node_b, ink_a, ink_b, True
        elif ink_a >= min_head and ink_a >= MIN_RATIO * ink_b:
            src, dst, s_ink, t_ink, ok = m.node_b, m.node_a, ink_b, ink_a, True
        else:
            src, dst, s_ink, t_ink, ok = m.node_a, m.node_b, ink_a, ink_b, False
        directed.append((int(src), int(dst)))
        infos.append(DirectionInfo(int(src), int(dst), int(s_ink), int(t_ink), ok))
    return directed, infos
