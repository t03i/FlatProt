# Copyright 2025 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0


from .base import Annotation

import numpy as np

from flatprot.style import StyleType


class AreaAnnotation(Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, indices=None, **kwargs)
        self._style_type = StyleType.AREA_ANNOTATION

    def display_coordinates(self) -> np.ndarray:
        "fit an area containing all targets as closely as possible"
        coords = np.concatenate([t.display_coordinates for t in self.targets])

        # Find the convex hull using numpy
        centroid = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
        hull_points = coords[np.argsort(angles)]

        # Add padding by expanding points outward from centroid
        padding = self.style.padding
        hull_vectors = hull_points - centroid
        normalized_vectors = (
            hull_vectors / np.linalg.norm(hull_vectors, axis=1)[:, np.newaxis]
        )
        hull_points = hull_points + normalized_vectors * padding

        # Generate more points through linear interpolation
        num_points = self.style.interpolation_points
        result_points = []
        for i in range(len(hull_points) - 1):
            t = np.linspace(0, 1, num_points // (len(hull_points) - 1))[:, np.newaxis]
            segment = (1 - t) * hull_points[i] + t * hull_points[i + 1]
            result_points.append(segment)

        points = np.vstack(result_points)

        # Apply rolling average to smooth the curve
        window_size = self.style.smoothing_window
        kernel = np.ones(window_size) / window_size

        # Pad the points array for periodic boundary conditions
        padded_points = np.vstack([points[-window_size:], points, points[:window_size]])

        # Apply convolution separately for x and y coordinates
        smoothed_x = np.convolve(padded_points[:, 0], kernel, mode="valid")
        smoothed_y = np.convolve(padded_points[:, 1], kernel, mode="valid")

        return np.column_stack([smoothed_x, smoothed_y])
