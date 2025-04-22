# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from flatprot.scene.structure.helix import calculate_zigzag_points
from flatprot.scene.structure.coil import smooth_coordinates


def test_smooth_coordinates():
    """Test coordinate smoothing with various reduction factors."""
    # Test normal case with reduction (10 points -> 2 points)
    coords = np.array([[i, i, i] for i in range(10)])  # Use 3D coords
    result_coords, result_indices = smooth_coordinates(coords, reduction_factor=0.2)
    assert len(result_coords) == 2  # target = max(2, int(10 * 0.2)) = 2
    np.testing.assert_array_equal(result_coords[0], coords[0])  # First point preserved
    np.testing.assert_array_equal(result_coords[-1], coords[-1])  # Last point preserved
    np.testing.assert_array_equal(result_indices, [0, 9])  # Indices for target=2

    # Test minimum points preservation (10 points -> 2 points, factor 0.1)
    result_coords_min, result_indices_min = smooth_coordinates(
        coords, reduction_factor=0.1
    )
    assert (
        len(result_coords_min) == 2
    )  # Should maintain minimum of 2 points: target = max(2, int(10 * 0.1)) = 2
    np.testing.assert_array_equal(result_coords_min[0], coords[0])
    np.testing.assert_array_equal(result_coords_min[-1], coords[-1])
    np.testing.assert_array_equal(result_indices_min, [0, 9])

    # Test with slightly more points kept (10 points -> 5 points, factor 0.5)
    result_coords_mid, result_indices_mid = smooth_coordinates(
        coords, reduction_factor=0.5
    )
    target_mid = max(2, int(10 * 0.5))  # 5
    assert len(result_coords_mid) == target_mid
    np.testing.assert_array_equal(result_coords_mid[0], coords[0])
    np.testing.assert_array_equal(result_coords_mid[-1], coords[-1])
    expected_indices_mid = np.linspace(0, 9, 5, dtype=int)
    np.testing.assert_array_equal(result_indices_mid, expected_indices_mid)

    # Test no reduction needed (3 points)
    small_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    result_small_coords, result_small_indices = smooth_coordinates(small_coords)
    np.testing.assert_array_equal(result_small_coords, small_coords)
    np.testing.assert_array_equal(result_small_indices, np.arange(3))

    # Test no reduction needed (2 points)
    two_coords = np.array([[0, 0, 0], [1, 1, 1]])
    result_two_coords, result_two_indices = smooth_coordinates(two_coords)
    np.testing.assert_array_equal(result_two_coords, two_coords)
    np.testing.assert_array_equal(result_two_indices, np.arange(2))

    # Test reduction_factor >= 1.0 (no reduction)
    result_coords_no_red, result_indices_no_red = smooth_coordinates(
        coords, reduction_factor=1.0
    )
    np.testing.assert_array_equal(result_coords_no_red, coords)
    np.testing.assert_array_equal(result_indices_no_red, np.arange(10))


def test_zigzag_points():
    """Test zigzag point calculation for helix ribbons."""
    # Test normal case (horizontal)
    start = np.array([0, 0, 0])
    end = np.array([10, 0, 5])  # Add Z coords
    thickness = 2.0
    wavelength = 5.0
    amplitude = 1.0
    result = calculate_zigzag_points(start, end, thickness, wavelength, amplitude)

    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 3  # Each point should be 3D (X, Y, Depth)
    num_cycles = max(1, int(10 / wavelength))  # Length is 10
    expected_wave_points = num_cycles * 2 + 1
    assert len(result) == expected_wave_points * 2  # Top and bottom points

    # Test that first top and first bottom points form ribbon width at start
    # result[0] is first top, result[-1] is first bottom (due to [::-1])
    first_top = result[0]
    first_bottom = result[-1]
    np.testing.assert_almost_equal(np.linalg.norm(first_top - first_bottom), thickness)
    # Check interpolated Z (should be start Z)
    assert first_top[2] == start[2]
    assert first_bottom[2] == start[2]

    # Test that last top and last bottom points form ribbon width at end
    last_top_idx = expected_wave_points - 1
    last_bottom_idx = expected_wave_points  # Index of last bottom point before reversal
    last_top = result[last_top_idx]
    last_bottom = result[last_bottom_idx]
    np.testing.assert_almost_equal(np.linalg.norm(last_top - last_bottom), thickness)
    # Check interpolated Z (should be end Z)
    assert last_top[2] == end[2]
    assert last_bottom[2] == end[2]

    # Test zero-length case
    result_zero = calculate_zigzag_points(
        np.array([1, 1, 1]), np.array([1, 1, 1]), thickness, wavelength, amplitude
    )
    assert result_zero is None  # Expect None for zero length

    # Test diagonal direction
    start_diag = np.array([0, 0, -2])
    end_diag = np.array([10, 10, 8])  # Z changes
    result_diag = calculate_zigzag_points(
        start_diag, end_diag, thickness, wavelength, amplitude
    )
    assert isinstance(result_diag, np.ndarray)
    assert result_diag.shape[1] == 3
    assert len(result_diag) > 4  # Should have multiple points
    # Check Z interpolation at start/end
    assert result_diag[0][2] == start_diag[2]
    assert result_diag[-1][2] == start_diag[2]
    last_top_idx_diag = len(result_diag) // 2 - 1
    last_bottom_idx_diag = len(result_diag) // 2
    assert result_diag[last_top_idx_diag][2] == end_diag[2]
    assert result_diag[last_bottom_idx_diag][2] == end_diag[2]
