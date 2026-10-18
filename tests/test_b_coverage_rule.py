"""B review: full-image vs usable-extent coverage and the stricter usable-only rule."""

import dataclasses

import numpy as np
import pytest

import vision_fixtures as vf
from camloc import contracts as c
from camloc import pnp


@pytest.fixture(scope="module")
def scene():
    return vf.Scene(count_u=16, count_v=10)


def _top_half_mask():
    mask = np.zeros((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool)
    mask[:232] = True  # usable extent = top half; its grid rows are 58 px tall
    return mask


def _upper_band(scene, k=0):
    """Landmarks in one full-image grid row (116..231): < 6 full cells, >= 6 usable cells."""
    pixels, visible = scene.project(k)
    keep = visible & (pixels[:, 1] >= 117) & (pixels[:, 1] < 230)
    return pixels[keep], scene.points_W[keep]


def test_solution_reports_both_grids(scene):
    mask = _top_half_mask()
    pixels, points = _upper_band(scene)
    solution = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC, valid_mask=mask)
    assert solution.full_image_cells == pnp.occupied_cells(pixels[solution.inliers])
    assert solution.occupied_cells == pnp.occupied_cells(pixels[solution.inliers], mask)
    assert solution.full_image_cells < solution.occupied_cells


def test_disabled_rule_is_the_plain_usable_extent_gate(scene):
    mask = _top_half_mask()
    pixels, points = _upper_band(scene)
    plain = dataclasses.replace(pnp.DEFAULT_GATES, min_usable_cells_without_full=0)
    solution = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC, plain, valid_mask=mask)
    assert solution.accepted
    assert solution.full_image_cells < pnp.DEFAULT_GATES.min_cells <= solution.occupied_cells


def test_stricter_rule_rejects_squeezed_support_only(scene):
    mask = _top_half_mask()
    pixels, points = _upper_band(scene)
    base = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC,
                         dataclasses.replace(pnp.DEFAULT_GATES, min_usable_cells_without_full=0),
                         valid_mask=mask)
    strict = dataclasses.replace(pnp.DEFAULT_GATES,
                                 min_usable_cells_without_full=base.occupied_cells + 1)
    squeezed = pnp.solve_pnp(pixels, points, scene.K, scene.T_BC, strict, valid_mask=mask)
    assert squeezed.reason == "poor_image_coverage"
    # Support that also covers >= 6 full-image cells is unaffected by the extra rule.
    all_pixels, visible = scene.project(3)
    full = pnp.solve_pnp(all_pixels[visible], scene.points_W[visible], scene.K, scene.T_BC,
                         strict, valid_mask=np.ones((c.CANONICAL_HEIGHT, c.CANONICAL_WIDTH), bool))
    assert full.accepted and full.full_image_cells >= pnp.DEFAULT_GATES.min_cells
