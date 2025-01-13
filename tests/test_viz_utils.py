# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from flatprot.visualization.elements.base import SmoothingMixin
from flatprot.visualization.composer import transform_to_canvas_space
from flatprot.visualization.utils import CanvasSettings


class TestSmoother(SmoothingMixin):
    pass


def test_smooth_coordinates():
    smoother = TestSmoother()

    # Test normal case with reduction
    coords = np.array([[i, i] for i in range(10)])
    result = smoother._smooth_coordinates(coords, reduction_factor=0.5)
    assert len(result) == 5
    np.testing.assert_array_equal(result[0], coords[0])  # First point preserved
    np.testing.assert_array_equal(result[-1], coords[-1])  # Last point preserved

    # Test minimum points preservation
    result = smoother._smooth_coordinates(coords, reduction_factor=0.1)
    assert len(result) == 3  # Should maintain minimum of 3 points

    # Test no reduction needed
    small_coords = np.array([[0, 0], [1, 1], [2, 2]])
    result = smoother._smooth_coordinates(small_coords)
    np.testing.assert_array_equal(result, small_coords)


def test_transform_to_canvas_space():
    # Test coordinates with known bounds
    coords = np.array(
        [
            [0, 0],  # center point
            [1, 1],  # top right
            [-1, -1],  # bottom left
            [2, -2],  # extreme point
        ]
    )

    # Create canvas with 10% padding
    canvas_settings = CanvasSettings(width=100, height=100, padding=0.1)
    pad_x = pad_y = int(100 * 0.1)  # 10 pixels padding
    available_space = 100 - 2 * pad_x  # 80 pixels

    transformed = transform_to_canvas_space(coords, canvas_settings)

    # Basic shape checks
    assert transformed.shape == coords.shape
    assert transformed.dtype == np.float64

    # Check bounds are scaled correctly with padding
    # Original bounds are [-1, 2] for x and [-2, 1] for y
    coord_width = 3  # original width (2 - (-1))
    coord_height = 3  # original height (1 - (-2))
    scale = available_space / max(
        coord_width, coord_height
    )  # Should scale to fit 80 pixels

    # Check the scaling respects padding
    max_abs_coord = np.max(np.abs(transformed))
    expected_max = available_space / 2  # 40 pixels from center (respecting padding)
    np.testing.assert_allclose(max_abs_coord, expected_max, rtol=1e-10)

    # Verify coordinates don't exceed padded area
    assert np.all(np.abs(transformed) <= (canvas_settings.width - 2 * pad_x) / 2)

    # Check aspect ratio preservation
    original_ratio = coord_width / coord_height
    transformed_width = np.max(transformed[:, 0]) - np.min(transformed[:, 0])
    transformed_height = np.max(transformed[:, 1]) - np.min(transformed[:, 1])
    transformed_ratio = transformed_width / transformed_height
    np.testing.assert_allclose(original_ratio, transformed_ratio, rtol=1e-10)

    # Check centering (mean should be at 0,0)
    center = np.mean(transformed, axis=0)
    expected_center = np.array([0, 0])
    np.testing.assert_allclose(center, expected_center, atol=1e-10)
