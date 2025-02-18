# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from .base import StructureSceneElement


def smooth_coordinates(coords: np.ndarray, reduction_factor: float = 0.2) -> np.ndarray:
    """Reduce point complexity using uniform selection.

    Args:
        coords: Input coordinates of shape (N, 2)
        reduction_factor: Fraction of points to keep (0.0-1.0)

    Returns:
        Simplified coordinates array
    """
    n_points = len(coords)
    if n_points <= 3 or reduction_factor >= 1.0:
        return coords

    # Always keep first and last points
    target_points = max(3, int(n_points * reduction_factor))
    if target_points >= n_points:
        return coords

    # Use linear indexing for uniform point selection
    indices = np.linspace(0, n_points - 1, target_points, dtype=int)
    return coords[indices]


class CoilElement(StructureSceneElement):
    """A coil element visualization using a smooth curved line"""

    def calculate_display_coordinates(self) -> np.ndarray:
        return self._coordinates[0, :], self._coordinates[-1, :]

    def display_coordinates_at_position(self, position: int) -> np.ndarray:
        """Get smoothed display coordinates at a specific position.

        Args:
            position: Index in the original coordinate array

        Returns:
            Coordinate in the smoothed representation
        """

        # Calculate relative position
        orig_len = len(self._coordinates)
        smooth_len = len(self._display_coordinates)

        # Map original position to smoothed array index
        mapped_idx = (position * (smooth_len - 1)) / (orig_len - 1)

        # Get indices for interpolation
        idx_low = int(np.floor(mapped_idx))
        idx_high = min(idx_low + 1, smooth_len - 1)

        # Linear interpolation between adjacent smoothed points
        frac = mapped_idx - idx_low
        return (
            self._display_coordinates[idx_low] * (1 - frac)
            + self._display_coordinates[idx_high] * frac
        )
