# Copyright 2024 Tobias Olenyi.
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from .base import StructureSceneElement


class HelixElement(StructureSceneElement):
    """A helical element visualization using a zigzag style"""

    def _get_zigzag_points(self, start, end, thickness, wavelength, amplitude):
        """Calculate points for a sharp zigzag helix pattern."""
        # Convert inputs to numpy arrays
        start = np.array(start)
        end = np.array(end)

        # Calculate direction vector and length
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:  # Handle zero-length case
            return [start.tolist()]

        # Normalize direction vector and get perpendicular
        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])

        # Calculate number of complete zigzag cycles
        num_cycles = max(1, int(length / wavelength))

        # Generate zigzag points (only peaks and valleys)
        t = np.linspace(0, length, num_cycles * 2 + 1)

        # Alternate between +amplitude and -amplitude
        wave = amplitude * np.array([1 if i % 2 == 0 else -1 for i in range(len(t))])

        # Create base points along the path
        base_points = t[:, None] * direction + start

        # Apply zigzag pattern
        wave_points = base_points + (wave[:, None] * perpendicular)

        # Create ribbon effect by offsetting top and bottom
        half_thickness = thickness / 2
        top_points = wave_points + (perpendicular * half_thickness)
        bottom_points = wave_points - (perpendicular * half_thickness)

        # Combine points to form complete ribbon outline
        return np.concatenate((top_points, bottom_points[::-1]))

    def calculate_display_coordinates(self) -> np.ndarray:
        return self._get_zigzag_points(
            self.coordinates[0],
            self.coordinates[-1],
            self.style.line_width * self.style.thickness_factor,
            self.style.line_width * self.style.cross_width_factor,
            self.style.line_width * self.style.amplitude_factor,
        )
