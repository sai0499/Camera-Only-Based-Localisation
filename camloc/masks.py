"""Static ego-vehicle (bodywork) masks in canonical 960 x 464 pixels (owner: B).

A's ``valid_mask`` covers remap validity only. The front cameras also see the car's own
nose and front tyre, which move with the camera and would create false static features.
These polygons were drawn on fitting-split ``fr`` images (pit lane, straights and curves;
overlay kept locally as ``results/b/mask_inspection``) with a few pixels of margin. ``r`` showed no
bodywork in the inspected fitting images and has no polygon. Masks are part of the fitted
map's identity (``MASK_ID``); a changed polygon needs a new id and a rebuilt map.
"""

from __future__ import annotations

import cv2
import numpy as np

from camloc import contracts as c

MASK_ID = "bodywork-v1"

# Pixel polygons (u, v) of the region to EXCLUDE, per camera.
BODYWORK_POLYGONS: dict[str, tuple[tuple[int, int], ...]] = {
    "fr": (
        (0, 246), (320, 248), (560, 258), (730, 286), (742, 244), (800, 232),
        (960, 224), (960, 464), (0, 464),
    ),
}


def bodywork_mask(camera_id: str) -> np.ndarray:
    """Return a new bool (464, 960) mask, True where the pixel is not ego-vehicle bodywork."""
    c.check_camera_id(camera_id)
    usable = np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), np.uint8)
    polygon = BODYWORK_POLYGONS.get(camera_id)
    if polygon is not None:
        cv2.fillPoly(usable, [np.asarray(polygon, np.int32)], 0)
    return usable.astype(bool)
