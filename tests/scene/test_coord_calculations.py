# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from flatprot.scene.structure.helix import calculate_zigzag_points
from flatprot.scene.structure.coil import smooth_coordinates


def test_smooth_coordinates():
    # Test normal case with reduction
    coords = np.array([[i, i] for i in range(10)])
    result = smooth_coordinates(coords, reduction_factor=0.5)
    assert len(result) == 5
    np.testing.assert_array_equal(result[0], coords[0])  # First point preserved
    np.testing.assert_array_equal(result[-1], coords[-1])  # Last point preserved

    # Test minimum points preservation
    result = smooth_coordinates(coords, reduction_factor=0.1)
    assert len(result) == 3  # Should maintain minimum of 3 points

    # Test no reduction needed
    small_coords = np.array([[0, 0], [1, 1], [2, 2]])
    result = smooth_coordinates(small_coords)
    np.testing.assert_array_equal(result, small_coords)


def test_zigzag_points():
    # Test normal case
    start = [0, 0]
    end = [10, 0]
    result = calculate_zigzag_points(start, end, thickness=2, wavelength=5, amplitude=1)

    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 2  # Each point should be 2D
    assert len(result) > 4  # Should have multiple points for ribbon effect

    # Test that first and last points form ribbon width
    top_points = result[: len(result) // 2]
    bottom_points = result[len(result) // 2 :]
    np.testing.assert_almost_equal(
        np.linalg.norm(top_points[0] - bottom_points[-1]), 2
    )  # thickness

    # Test zero-length case
    result_zero = calculate_zigzag_points(
        [1, 1], [1, 1], thickness=2, wavelength=5, amplitude=1
    )
    assert len(result_zero) == 1
    np.testing.assert_array_equal(result_zero[0], [1, 1])

    # Test diagonal direction
    start_diag = [0, 0]
    end_diag = [10, 10]
    result_diag = calculate_zigzag_points(
        start_diag, end_diag, thickness=2, wavelength=5, amplitude=1
    )
    assert isinstance(result_diag, np.ndarray)
    assert result_diag.shape[1] == 2

    # Test that points stay within expected bounds
    max_distance = np.max(np.linalg.norm(result - np.array([5, 0]), axis=1))
    assert (
        max_distance <= np.sqrt(50) + 2
    )  # Max distance from midpoint plus thickness/2
