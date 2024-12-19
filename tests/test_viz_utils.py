# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from flatprot.visualization.elements.base import VisualizationStyle, SmoothingMixin
from flatprot.visualization.utils import CanvasSettings
from flatprot.visualization.composer import transform_to_canvas_space


class TestSmoothingMixin:
    class DummyElement(SmoothingMixin):
        pass

    def test_smooth_coordinates(self):
        element = self.DummyElement()
        coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

        # Test with window=1 (no smoothing)
        smoothed = element._smooth_coordinates(coords, window=1)
        np.testing.assert_array_equal(smoothed, coords)

        # Test with window=3
        smoothed = element._smooth_coordinates(coords, window=3)
        assert smoothed.shape == coords.shape


def test_transform_to_canvas_space():
    # Create test coordinates
    coords = np.array([[0, 0], [1, 1], [-1, -1], [2, -2]])

    canvas_settings = CanvasSettings(width=100, height=100, padding=0.1)

    transformed = transform_to_canvas_space(coords, canvas_settings)

    # Check output shape
    assert transformed.shape == coords.shape

    # Check bounds are within canvas
    assert np.all(transformed[:, 0] >= 0)
    assert np.all(transformed[:, 0] <= canvas_settings.width)
    assert np.all(transformed[:, 1] >= 0)
    assert np.all(transformed[:, 1] <= canvas_settings.height)
