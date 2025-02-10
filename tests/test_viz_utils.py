# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from flatprot.visualization.structure.base import SmoothingMixin


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
